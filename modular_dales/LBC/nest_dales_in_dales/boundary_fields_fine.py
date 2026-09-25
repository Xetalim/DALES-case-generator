import logging
from datetime import datetime
from typing import TYPE_CHECKING
from pathlib import Path
import glob

import numpy as np
import xarray as xr

from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap

from modular_dales.LBC.nest_dales_in_dales.get_all_dales_boundaries import (
    get_all_dales_boundaries,
)

if TYPE_CHECKING:
    from ..nesting_idx import NestingIndices

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@logwrap
def boundary_fields_fine(
    input_json: OpenBoundaryConfig,
    grid: GridDalesOpenBC,
    output_path,
    grid_indices: "NestingIndices",
    chunks=None,
):
    if not grid_indices:
        raise ValueError("No nesting indices provided!")

    # Get initial boundary fields from initial fields
    openboundaries = get_all_dales_boundaries(
        input_json, grid, grid_indices, chunks=chunks
    )
    if input_json.lsynturb:
        openboundaries = add_synthetic_turbulence(
            input_json, openboundaries, chunks=chunks
        )
    openboundaries = set_openboundary_attrs(input_json, openboundaries)
    # Add global attributes
    openboundaries = openboundaries.assign_attrs(
        {
            "history": f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "author": input_json.author,
            "time0": input_json.time0,
        }
    )
    return openboundaries


def _find_profiles_file(run_path: str) -> list[str]:
    pattern = (Path(run_path) / "profiles.*.nc").as_posix()
    return sorted(glob.glob(pattern))


def _interp_profile_to_zt(
    var: xr.DataArray, target_time: xr.DataArray, target_zt: xr.DataArray
):
    out = var
    if "time" in out.dims:
        out = out.sortby("time")
        out = out.interp(time=target_time, kwargs={"fill_value": "extrapolate"})

    zdim = None
    for candidate in ["zt", "zm", "z", "zh"]:
        if candidate in out.dims:
            zdim = candidate
            break

    if zdim is None:
        raise ValueError(
            f"Cannot map variable '{var.name}' to boundary zt levels: no vertical dimension found in {out.dims}."
        )

    if zdim != "zt":
        out = out.rename({zdim: "zt"})

    out = out.sortby("zt")
    out = out.interp(zt=target_zt, kwargs={"fill_value": "extrapolate"})
    return out.transpose("time", "zt")


def _clip_covariance(cov: xr.DataArray, var_a: xr.DataArray, var_b: xr.DataArray):
    limit = np.sqrt(xr.where(var_a * var_b > 0.0, var_a * var_b, 0.0))
    return cov.clip(min=-limit, max=limit)


def _project_zero_into_interval(lower: xr.DataArray, upper: xr.DataArray):
    return xr.where(lower > 0.0, lower, xr.where(upper < 0.0, upper, 0.0))


def _load_synturb_profiles(
    input_json: OpenBoundaryConfig,
    target_time: xr.DataArray,
    target_zt: xr.DataArray,
    chunks=None,
):
    profile_files = _find_profiles_file(input_json.outpath_coarse)
    if len(profile_files) == 0:
        raise FileNotFoundError(
            "Synthetic turbulence needs parent profiles.*.nc output from modgenstat, "
            f"but none was found in '{input_json.outpath_coarse}'. Enable parent "
            "StatsModule (namgenstat:lstat=.true.) and re-run the parent case."
        )

    if chunks is None:
        tchunk = input_json.tchunk
        chunks = {"time": int(tchunk)} if tchunk is not None else {"time": 1}

    with xr.open_mfdataset(
        profile_files,
        chunks=chunks,
        join="outer",
    ) as ds_prof:
        required = {
            "u2": "u2r",
            "v2": "v2r",
            "w2r": "w2r",
            "w2s": "w2s",
            "uw": "uwt",
            "vw": "vwt",
            "thl2": "thl2r",
            "wthl": "wthlt",
            "qt2": "qt2r",
            "wqt": "wqtt",
        }
        missing = [src for src in required.values() if src not in ds_prof]
        if missing:
            raise KeyError(
                "Synthetic turbulence from modgenstat needs variables "
                f"{list(required.values())} in profiles.*.nc, missing: {missing}."
            )

        u2 = _interp_profile_to_zt(ds_prof[required["u2"]], target_time, target_zt)
        v2 = _interp_profile_to_zt(ds_prof[required["v2"]], target_time, target_zt)
        w2r = _interp_profile_to_zt(ds_prof[required["w2r"]], target_time, target_zt)
        w2s = _interp_profile_to_zt(ds_prof[required["w2s"]], target_time, target_zt)
        uw = _interp_profile_to_zt(ds_prof[required["uw"]], target_time, target_zt)
        vw = _interp_profile_to_zt(ds_prof[required["vw"]], target_time, target_zt)
        thl2 = _interp_profile_to_zt(ds_prof[required["thl2"]], target_time, target_zt)
        wthl = _interp_profile_to_zt(ds_prof[required["wthl"]], target_time, target_zt)
        qt2 = _interp_profile_to_zt(ds_prof[required["qt2"]], target_time, target_zt)
        wqt = _interp_profile_to_zt(ds_prof[required["wqt"]], target_time, target_zt)

    u2 = xr.where(u2 > 0.0, u2, 0.0)
    v2 = xr.where(v2 > 0.0, v2, 0.0)
    w2 = xr.where((w2r + w2s) > 0.0, w2r + w2s, 0.0)
    thl2 = xr.where(thl2 > 0.0, thl2, 0.0)
    qt2 = xr.where(qt2 > 0.0, qt2, 0.0)
    uw = _clip_covariance(uw, u2, w2)
    vw = _clip_covariance(vw, v2, w2)
    wthl = _clip_covariance(wthl, w2, thl2)
    wqt = _clip_covariance(wqt, w2, qt2)

    # Reconstruct uv from the positive semidefinite Reynolds-stress condition.
    w2_safe = xr.where(w2 > 1e-9, w2, 1e-9)
    radicand = (uw * vw) ** 2 + w2 * (u2 * v2 * w2 - u2 * vw**2 - v2 * uw**2)
    radicand = xr.where(radicand > 0.0, radicand, 0.0)
    uv_low = (uw * vw - np.sqrt(radicand)) / w2_safe
    uv_high = (uw * vw + np.sqrt(radicand)) / w2_safe
    uv = _project_zero_into_interval(uv_low, uv_high)
    uv = _clip_covariance(uv, u2, v2)
    uv = xr.where(np.isfinite(uv), uv, 0.0)

    return {
        "u2": u2,
        "v2": v2,
        "w2": w2,
        "uv": uv,
        "uw": uw,
        "vw": vw,
        "thl2": thl2,
        "wthl": wthl,
        "qt2": qt2,
        "wqt": wqt,
    }


@logwrap
def add_synthetic_turbulence(
    input_json: OpenBoundaryConfig, openboundaries: xr.Dataset, chunks=None
) -> xr.Dataset:
    synturb_profiles = _load_synturb_profiles(
        input_json,
        openboundaries["time"],
        openboundaries["zt"],
        chunks=chunks,
    )

    synturb_vars = ["u2", "v2", "w2", "uv", "uw", "vw", "thl2", "wthl", "qt2", "wqt"]

    for boundary in ["west", "east", "south", "north"]:
        template = openboundaries[f"e12{boundary}"]
        for var in synturb_vars:
            openboundaries[f"{var}{boundary}"] = (
                (xr.ones_like(template) * synturb_profiles[var])
                .transpose(*template.dims)
                .rename(f"{var}{boundary}")
            )

    top_template = openboundaries["e12top"]
    for var in synturb_vars:
        openboundaries[f"{var}top"] = xr.zeros_like(top_template).rename(f"{var}top")

    return openboundaries


@logwrap
def set_openboundary_attrs(input_json: OpenBoundaryConfig, openboundaries):
    dts = (
        openboundaries.time.values.astype("datetime64[s]")
        - np.datetime64(input_json.time0, "s")
    ) / np.timedelta64(1, "s")
    openboundaries = openboundaries.assign_coords({"time": ("time", dts)})
    # # Adjust time variable to seconds since initial field
    # ts = openboundaries['time'].values.astype('datetime64[s]')
    # dts = (ts-np.datetime64(time0,'s'))/np.timedelta64(1, 's')
    # openboundaries = openboundaries.assign_coords({'time':('time', dts)})
    # Add variable attributes
    openboundaries["time"] = openboundaries["time"].assign_attrs(
        {"longname": "Time"}
    )  # , 'units': f"seconds since {time0}"})
    openboundaries.time.encoding["units"] = f"seconds since {input_json.time0}"
    openboundaries["xt"] = openboundaries["xt"].assign_attrs(
        {"longname": "West-East displacement of cell centers", "units": "m"}
    )
    openboundaries["xm"] = openboundaries["xm"].assign_attrs(
        {"longname": "West-East displacement of cell edges", "units": "m"}
    )
    openboundaries["yt"] = openboundaries["yt"].assign_attrs(
        {"longname": "South-North displacement of cell centers", "units": "m"}
    )
    openboundaries["ym"] = openboundaries["ym"].assign_attrs(
        {"longname": "South-North displacement of cell edges", "units": "m"}
    )
    openboundaries["zt"] = openboundaries["zt"].assign_attrs(
        {"longname": "Vertical displacement of cell centers", "units": "m"}
    )
    openboundaries["zm"] = openboundaries["zm"].assign_attrs(
        {"longname": "Vertical displacement of cell edges", "units": "m"}
    )
    variables = ["u", "v", "w", "thl", "qt", "e12"]
    if len(input_json.tracernames) > 0:
        for tracername in input_json.tracernames:
            variables.append(tracername)
    units = ["m/s", "m/s", "m/s", "K", "kg/kg", "m/s"]
    if len(input_json.tracernames) > 0:
        for tracername in input_json.tracernames:
            if len(input_json.tracernames) > 2:
                logger.warning(
                    "Unit applied to tracer %s might not be correct!", tracername
                )
            units.append("kg/kg")
    long_names = [
        "West-East velocity at ",
        "South-North velocity at ",
        "Vertical velocity at ",
        "Liquid water potential temperature at ",
        "Total water specific humidity at ",
        "Square root of turbulent kinetic energy at ",
    ]
    if len(input_json.tracernames) > 0:
        for tracername in input_json.tracernames:
            long_names.append(f"scalar field {tracername} at ")
    for ivar, var in enumerate(variables):
        unit = units[ivar]
        long_name = long_names[ivar]
        for boundary in ["West", "East", "South", "North", "top"]:
            openboundaries[var + boundary.lower()] = openboundaries[
                var + boundary.lower()
            ].assign_attrs(
                {"longname": long_name + boundary + " boundary", "units": unit}
            )

    if input_json.lsynturb:
        turb_variables = [
            "u2",
            "v2",
            "w2",
            "uv",
            "uw",
            "vw",
            "thl2",
            "qt2",
            "wthl",
            "wqt",
        ]
        turb_units = [
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "K2",
            "(kg/kg)2",
            "K m/s",
            "kg/kg m/s",
        ]
        turb_long_names = [
            "Variance of u at ",
            "Variance of v at ",
            "Variance of w at ",
            "Covariance of u and v at ",
            "Covariance of u and w at ",
            "Covariance of v and w at ",
            "Variance of thl at ",
            "Variance of qt at ",
            "Covariance of w and thl at ",
            "Covariance of w and qt at ",
        ]
        for ivar, var in enumerate(turb_variables):
            unit = turb_units[ivar]
            long_name = turb_long_names[ivar]
            for boundary in ["West", "East", "South", "North", "top"]:
                key = var + boundary.lower()
                if key in openboundaries:
                    openboundaries[key] = openboundaries[key].assign_attrs(
                        {"longname": long_name + boundary + " boundary", "units": unit}
                    )

    return openboundaries
