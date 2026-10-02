import numpy as np
import xarray as xr
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap


def _project_zero_into_interval(lower: xr.DataArray, upper: xr.DataArray):
    return xr.where(lower > 0.0, lower, xr.where(upper < 0.0, upper, 0.0))


def _clip_covariance(cov: xr.DataArray, var_a: xr.DataArray, var_b: xr.DataArray):
    limit = np.sqrt(xr.where(var_a * var_b > 0.0, var_a * var_b, 0.0))
    return cov.clip(min=-limit, max=limit)


def _interp_profile_to_zt(
    var: xr.DataArray, target_time: xr.DataArray, target_zt: xr.DataArray
):
    out = var
    out = out.interp(time=target_time, kwargs={"fill_value": "extrapolate"})

    zdim = None
    for candidate in ["zt", "zm"]:
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


def _load_synturb_profiles(
    input_json: OpenBoundaryConfig,
    target_time: xr.DataArray,
    target_zt: xr.DataArray,
):
    with xr.open_dataset(
        f"{input_json.outpath_coarse}/profiles.001.nc",  # TODO robust for expnr
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
