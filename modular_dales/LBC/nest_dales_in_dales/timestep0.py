from typing import TYPE_CHECKING
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import glob


from modular_dales.Geometry.GridDales import GridDalesOpenBC
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.LBC.nest_dales_in_dales.load_any_boundary_var import (
    load_any_boundary_var,
)
from modular_dales.logging_wrapper import logwrap
from modular_dales.LBC.nest_dales_in_dales.load_any_boundary_var import (
    get_boundary_dict,
)

if TYPE_CHECKING:
    from modular_dales.LBC.nesting_idx import NestingIndices

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@logwrap
def boundaries_timestep0(
    input_json: OpenBoundaryConfig,
    grid: GridDalesOpenBC,
    indices: "NestingIndices",
    chunks=None,
):
    """Return boundary fields for the initial timestep.

    If the host run produced ``initfields.inp.*.nc`` we use that file for
    t=0, exactly as in the original DALES nesting utilities.  If that file
    is not available, we fall back to the vertical profile file
    ``init.*.nc`` where ``zh == zt`` and construct horizontally uniform
    boundary fields for all variables and boundaries from that profile.
    """
    inpath_coarse = Path(input_json.inpath_coarse)
    crosssection_chunks = chunks
    if crosssection_chunks is None:
        tchunk = input_json.tchunk
        crosssection_chunks = (
            {"time": int(tchunk)} if tchunk is not None else {"time": 1}
        )

    if input_json.time0 == input_json.start:
        # Prefer initfields.inp.*.nc if available
        try:
            all_ls = []
            with xr.open_mfdataset(
                (inpath_coarse / "initfields.inp.*.nc").as_posix(),
                join="outer",
            ) as ds:
                for boundary in ["west", "east", "north", "south", "top"]:
                    for var in ["u", "v", "w", "thl", "qt", "e12"]:
                        all_ls.append(
                            load_any_boundary_var(
                                ds,
                                var,
                                boundary,
                                grid,
                                indices,
                                isel=True,
                                expand_dims=True,
                                expand_dims_time0=input_json.time0,
                                var_postfix="0",
                            )
                        )
                    if len(input_json.tracernames) > 0:
                        sv_boundary = []
                        for tracername in input_json.tracernames:
                            sv_boundary.append(
                                xr.zeros_like(all_ls[-1]).rename(
                                    f"{tracername}{boundary}"
                                )
                            )
                        all_ls.append(xr.merge(sv_boundary))
            return xr.merge(all_ls)
        except (OSError, FileNotFoundError, ValueError) as exc:
            # xarray raises ValueError("no files to open") when the glob is empty.
            logger.warning(
                "initfields.inp.*.nc not found or not readable (%s); "
                "falling back to init.*.nc profile for initial boundaries.",
                exc,
            )

            # Fallback: build horizontally uniform boundaries from init.*.nc,
            # which contains 1D profiles along zh == zt.
            init_ds = xr.open_mfdataset(
                (inpath_coarse / "init.*.nc").as_posix(),
                join="outer",
            )

            # Mapping from openboundary variable names to candidate variable
            # names in init.*.nc. We pick the first one that exists.
            profile_var_candidates = {
                "u": ["ua"],
                "v": ["va"],
                "w": ["wa"],
                "thl": ["thetal"],
                "qt": ["qt"],
                "e12": ["tke"],
            }

            zh = init_ds["zh"].values

            def _get_profile(var: str) -> np.ndarray:
                if var in profile_var_candidates:
                    if var == "w":
                        return np.zeros_like(zh, dtype=float)
                    for cand in profile_var_candidates[var]:
                        if cand in init_ds:
                            return np.asarray(init_ds[cand].values, dtype=float)
                    raise KeyError(
                        f"None of {profile_var_candidates[var]} found in init.*.nc for '{var}'"
                    )
                # tracers: expect the name itself to be present
                if var not in init_ds:
                    raise KeyError(f"Tracer '{var}' not found in init.*.nc")
                return np.asarray(init_ds[var].values, dtype=float)

            # Dimension mapping identical to load_any_boundary_var
            var_dims_dic = {
                "u": {"x": "xm", "y": "yt", "z": "zt"},
                "v": {"x": "xt", "y": "ym", "z": "zt"},
                "w": {"x": "xt", "y": "yt", "z": "zm"},
                "default": {"x": "xt", "y": "yt", "z": "zt"},
            }

            boundary_var_assign_dic = {
                "west": {"default": ["z", "y"]},
                "east": {"default": ["z", "y"]},
                "south": {"default": ["z", "x"]},
                "north": {"default": ["z", "x"]},
                "top": {"default": ["y", "x"]},
            }

            assign_grid_dic = {
                "xt": grid.xt,
                "xm": grid.xm,
                "yt": grid.yt,
                "ym": grid.ym,
                "zt": grid.zt,
                "zm": grid.zm,
            }

            def _build_uniform_boundary(
                profile_zt: np.ndarray, var: str, boundary: str
            ):
                # Determine base dims for this boundary and variable
                if var in var_dims_dic:
                    var_dims = var_dims_dic[var]
                else:
                    var_dims = var_dims_dic["default"]

                if var in boundary_var_assign_dic[boundary]:
                    base_dims = boundary_var_assign_dic[boundary][var]
                else:
                    base_dims = boundary_var_assign_dic[boundary]["default"]

                dims = [var_dims[d] for d in base_dims]

                # Identify vertical dimension, if any
                vertical_dim = None
                for d in ("zt", "zm"):
                    if d in dims:
                        vertical_dim = d
                        break

                if vertical_dim is None:
                    # Top boundary: use value at model top
                    fill_value = float(profile_zt[-1])
                    shape = [len(assign_grid_dic[d]) for d in dims]
                    data = np.full(shape, fill_value, dtype=float)
                else:
                    # Map profile to the vertical coordinate used
                    if vertical_dim == "zt":
                        # Interpolate from zh to the current grid zt
                        prof_on_vert = np.interp(assign_grid_dic["zt"], zh, profile_zt)
                        vert_coords = assign_grid_dic["zt"]
                    else:  # zm: interpolate from zh/zt -> zm
                        prof_on_vert = np.interp(assign_grid_dic["zm"], zh, profile_zt)
                        vert_coords = assign_grid_dic["zm"]

                    shape = [len(assign_grid_dic[d]) for d in dims]
                    data = np.zeros(shape, dtype=float)
                    vert_index = dims.index(vertical_dim)
                    for k in range(len(vert_coords)):
                        idx = [slice(None)] * len(shape)
                        idx[vert_index] = k
                        data[tuple(idx)] = prof_on_vert[k]

                coords = {d: assign_grid_dic[d] for d in dims}
                da = xr.DataArray(data, coords=coords, dims=dims)
                # Add time dimension at time0
                da = da.expand_dims({"time": [np.datetime64(input_json.time0)]}, axis=0)
                da.name = f"{var}{boundary}"
                return da

            all_ls = []
            base_vars = ["u", "v", "w", "thl", "qt", "e12"]
            tracers = list(input_json.tracernames)

            for var in base_vars + tracers:
                profile_zt = _get_profile(var)
                for boundary in ["west", "east", "south", "north", "top"]:
                    all_ls.append(_build_uniform_boundary(profile_zt, var, boundary))

            return xr.merge(all_ls)

    # Get initial boundary fields from previous simulation, specifically,
    # the last time step in the output of the previous simulation
    else:

        boundary_dict = get_boundary_dict(input_json.outpath_coarse_old, grid, indices)

        all_ls = []
        for boundary, (boundaryfile, sel_index) in boundary_dict.items():
            with xr.open_mfdataset(
                glob.glob(boundaryfile.as_posix()),
                chunks=crosssection_chunks,
                join="outer",
            ) as ds:
                for var in [
                    "u",
                    "v",
                    "w",
                    "thl",
                    "qt",
                    "e12",
                    *input_json.tracernames,
                ]:
                    if var == "e12":
                        var_postfix = "0"
                    else:
                        var_postfix = ""
                    all_ls.append(
                        load_any_boundary_var(
                            ds.sel(sel_index),
                            var,
                            boundary=boundary,
                            grid=grid,
                            indices=indices,
                            isel={"time": -1},
                            expand_dims=True,
                            expand_dims_time0=input_json.start,
                            var_postfix=var_postfix,
                        )
                    )

    return xr.merge(all_ls)
