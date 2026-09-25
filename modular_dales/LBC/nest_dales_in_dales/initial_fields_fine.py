import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@logwrap
def initial_fields_fine(
    input_json: OpenBoundaryConfig, grid: GridDalesOpenBC, output_path
):
    """
    Docstring for initial_fields_fine

    :param input_json: Configuration dict for LBC
    :param grid: Grid object describing the output grid for open boundaries, which is different from the normal DALES grid.
    :type grid: GridDalesOpenBC
    :param output_path: Where the data should be output.

    Gets the initial fields for the current simulation from the host
    ``initfields.inp.*.nc``. If that file is not available, fall back to the
    vertical-profile file ``init.*.nc`` (where zh == zt) and build a
    horizontally uniform 3D initial field from those profiles.
    """
    inpath = Path(input_json.inpath)

    # Load data: prefer initfields.inp.*.nc, fall back to init.*.nc
    try:
        with xr.open_mfdataset(
            (inpath / "initfields.inp.*.nc").as_posix(),
            join="outer",
        ) as ds:
            initfields_fine = ds.interp(
                xt=grid.xt,
                xm=grid.xm,
                yt=grid.yt,
                ym=grid.ym,
                zt=grid.zt,
                zm=grid.zm,
                assume_sorted=False,
            )
            initfields_fine = initfields_fine.assign_coords(
                {
                    "xt": grid.xt,
                    "xm": grid.xm,
                    "yt": grid.yt,
                    "ym": grid.ym,
                }
            )
    except (OSError, FileNotFoundError, ValueError) as exc:
        # xarray raises ValueError("no files to open") when the glob is empty.
        logger.warning(
            "initfields.inp.*.nc not found or not readable in '%s' (%s); "
            "falling back to init.*.nc profile for initial 3D fields.",
            inpath,
            exc,
        )

        init_ds = xr.open_mfdataset(
            (inpath / "init.*.nc").as_posix(),
            join="outer",
        )

        # Map openBC / DALES variables to candidate profile names in init.*.nc
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
                for cand in profile_var_candidates[var]:
                    if cand in init_ds:
                        return np.asarray(init_ds[cand].values, dtype=float)
                if var == "w":
                    # For vertical velocity, if no profile is found, assume zero
                    return np.zeros_like(zh, dtype=float)
                raise KeyError(
                    f"None of {profile_var_candidates[var]} found in init.*.nc for '{var}'"
                )
            # tracers: expect the name itself to be present
            if var not in init_ds:
                raise KeyError(f"Tracer '{var}' not found in init.*.nc")
            return np.asarray(init_ds[var].values, dtype=float)

        # Construct uniform 3D fields on the DALES grid (xt/xm, yt/ym, zt/zm)
        coords = {
            "xt": ("xt", grid.xt),
            "xm": ("xm", grid.xm),
            "yt": ("yt", grid.yt),
            "ym": ("ym", grid.ym),
            "zt": ("zt", grid.zt),
            "zm": ("zm", grid.zm),
        }
        data_vars = {}

        base_vars = ["u", "v", "w", "thl", "qt", "e12"]
        tracers = list(input_json.tracernames)

        def _make_uniform_field(
            profile: np.ndarray, dims: tuple[str, ...]
        ) -> np.ndarray:
            """Broadcast 1D vertical profile over the given spatial dims.

            dims is a tuple like ("yt", "xm", "zt") or ("yt", "xt", "zm").
            The vertical dimension is assumed to be either "zt" or "zm".
            """

            shape = [len(coords[d][1]) for d in dims]
            data = np.zeros(shape, dtype=float)

            if "zm" in dims:
                vertical_dim = "zm"
            else:
                vertical_dim = "zt"

            kdim = dims.index(vertical_dim)
            for k in range(len(profile)):
                idx = [slice(None)] * len(shape)
                idx[kdim] = k
                data[tuple(idx)] = profile[k]

            return data

        for var in base_vars + tracers:
            prof_zt = _get_profile(var)

            if var == "w":
                # For vertical velocity, interpolate from zh/zt -> zm
                prof_target = np.interp(grid.zm, zh, prof_zt)
                dims_no_time = ("zm", "yt", "xt")
                arr = _make_uniform_field(prof_target, dims_no_time)
                data_vars["w0"] = ("time",) + dims_no_time, arr[None, ...]
            else:
                # Interpolate profile from zh to the current grid zt before broadcasting
                prof_target = np.interp(grid.zt, zh, prof_zt)
                # Use the natural stagger for core fields
                if var == "u":
                    dims_no_time = ("zt", "yt", "xm")
                    arr = _make_uniform_field(prof_target, dims_no_time)
                    data_vars["u0"] = ("time",) + dims_no_time, arr[None, ...]
                elif var == "v":
                    dims_no_time = ("zt", "ym", "xt")
                    arr = _make_uniform_field(prof_target, dims_no_time)
                    data_vars["v0"] = ("time",) + dims_no_time, arr[None, ...]
                else:
                    # Scalars (thl, qt, e12, tracers): defined on (yt, xt, zt)
                    dims_no_time = ("zt", "yt", "xt")
                    arr = _make_uniform_field(prof_target, dims_no_time)
                    data_vars[f"{var}0"] = ("time",) + dims_no_time, arr[None, ...]

        initfields_fine = xr.Dataset(data_vars=data_vars, coords=coords)
        initfields_fine = initfields_fine.assign_coords(
            {
                "time": (
                    "time",
                    [
                        np.datetime64(input_json.time0),
                    ],
                )
            }
        )

    # Add global attributes
    initfields_fine = initfields_fine.assign_attrs(
        {
            "history": f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "author": input_json.author,
            "time0": input_json.time0,
        }
    )

    return initfields_fine
