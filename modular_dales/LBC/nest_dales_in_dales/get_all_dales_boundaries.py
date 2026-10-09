import glob
import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.LBC.nest_dales_in_dales.load_any_boundary_var import (
    get_boundary_dict,
    load_any_boundary_var,
)
from modular_dales.LBC.nest_dales_in_dales.timestep0 import boundaries_timestep0
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap

if TYPE_CHECKING:
    from modular_dales.LBC.nesting_idx import NestingIndices

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def _as_scalar_float(value) -> float:
    """Convert scalar-like inputs (numpy/xarray scalars) to plain float."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar value, got {arr!r}")
    return float(arr.reshape(-1)[0])


def _coord_values_for_expand(ds: xr.Dataset, coord_name: str) -> np.ndarray:
    """Get coordinate values in a shape suitable for expand_dims."""

    coord = ds[coord_name]
    if coord_name in coord.dims:
        return coord.values

    return np.asarray([_as_scalar_float(coord.values)])


def _promote_boundary_stagger_dims(ds: xr.Dataset, boundary: str) -> xr.Dataset:
    """Promote singleton stagger coords to real dims before multi-file merge.

    DALES cross-section files can store the selected section coordinate as a
    scalar coordinate (e.g. xt/xm, yt/ym, zt/zm) instead of a dimension on each
    variable. We add the expected singleton stagger dimension per variable so
    combine-by-coords can align files across both time and section location.
    """

    boundary_dim_map = {
        "west": {"default": "xt", "u": "xm"},
        "east": {"default": "xt", "u": "xm"},
        "south": {"default": "yt", "v": "ym"},
        "north": {"default": "yt", "v": "ym"},
        "top": {"default": "zt", "w": "zm"},
    }

    if boundary not in boundary_dim_map:
        return ds

    ds_out = ds
    dim_map = boundary_dim_map[boundary]

    for var_name, da in ds.data_vars.items():
        target_dim = dim_map.get(var_name, dim_map["default"])
        if target_dim in da.dims:
            continue

        if target_dim not in ds.coords:
            continue

        ds_out[var_name] = da.expand_dims(
            {target_dim: _coord_values_for_expand(ds, target_dim)}
        )

    return ds_out


def _prepare_boundary_variable(
    ds: xr.Dataset, boundary: str, variable: str
) -> xr.Dataset:
    """Combine one staggered field without aligning it along other fields' axes."""
    selected = ds[[variable]]
    return _promote_boundary_stagger_dims(selected, boundary).reset_coords(drop=True)


def _file_matches_boundary_selection(file_path: str, sel_index: dict) -> bool:
    """Return True when a cross-section file contains the requested section coords."""

    try:
        with xr.open_dataset(file_path) as ds_file:
            for dim_name, target_value in sel_index.items():
                if dim_name not in ds_file.coords:
                    return False

                coord_vals = np.asarray(ds_file[dim_name].values).reshape(-1)
                if coord_vals.size == 0:
                    return False

                if not np.any(
                    np.isclose(coord_vals, float(target_value), rtol=0.0, atol=1e-8)
                ):
                    return False
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot inspect boundary file '{file_path}' for coordinates {sel_index}"
        ) from exc

    return True


def _select_boundary_files(
    boundary_files: list[str], sel_index: dict, boundary: str
) -> list[str]:
    """Keep only files that match requested section coordinates for this boundary."""

    selected_files = [
        file_path
        for file_path in boundary_files
        if _file_matches_boundary_selection(file_path, sel_index)
    ]

    if not selected_files:
        raise ValueError(
            f"No cross-section files match the {boundary} boundary coordinates {sel_index}; "
            f"inspected {len(boundary_files)} candidate files. Regenerate and rerun the parent "
            "case with the matching NestingTopology cross-sections."
        )

    return selected_files


def _warn_if_suspicious_time_axis(
    boundaries: xr.Dataset, input_json: OpenBoundaryConfig
) -> None:
    """Warn when the concatenated boundary time axis looks inconsistent.

    This is intentionally non-invasive: it only emits warnings and does not
    reorder or drop entries.
    """
    if "time" not in boundaries.coords:
        logger.warning("Open-boundary dataset has no 'time' coordinate.")
        return

    times = boundaries["time"].values
    if times.size == 0:
        logger.warning("Open-boundary dataset has an empty 'time' coordinate.")
        return
    # At this stage times are datetimes. Convert to integer seconds for robust
    # monotonicity/duplicate checks.
    t_sec = times.astype("datetime64[s]").astype(np.int64)

    if t_sec.size > 1:
        dt = np.diff(t_sec)
        if np.any(dt < 0):
            logger.warning(
                "Open-boundary time axis is not monotonically increasing; "
                "this can cause unrealistic temporal interpolation of boundary forcing."
                f"{times}"
            )
        if np.any(dt == 0):
            logger.warning(
                "Open-boundary time axis contains duplicate timestamps; "
                "this can cause ambiguous boundary forcing at repeated times."
                f"{times}"
            )

    start = input_json.start
    time0 = input_json.time0
    if start is None or time0 is None:
        return

    first_time = np.datetime64(times[0], "s")
    start_time = np.datetime64(start, "s")
    time0_time = np.datetime64(time0, "s")

    expected_first = time0_time if start_time == time0_time else start_time
    if first_time != expected_first:
        logger.warning(
            "First boundary timestamp (%s) does not match expected initial timestamp (%s). "
            "This may indicate a misconfigured start/time0 relationship or source selection.",
            str(first_time),
            str(expected_first),
        )


@logwrap
def get_all_dales_boundaries(
    input_json: OpenBoundaryConfig,
    grid: GridDalesOpenBC,
    indices: "NestingIndices",
    chunks=None,
):
    """
    This function gets DALES boundaries from a host DALES simulation.

    :param input_json: Configuration dict for LBC
    :param grid: Grid object describing the output grid for open boundaries, which is different from the normal DALES grid.
    :type grid: GridDalesOpenBC
    :param indices: Indices describing where in the supergrid the edges of the current grid lie.
    :type indices: nesting_idx
    """
    ds_boundaries_timestep0 = boundaries_timestep0(
        input_json, grid, indices, chunks=chunks
    )
    # Get later time steps from corresponding coarse simulation output
    boundary_dict = get_boundary_dict(input_json.outpath_coarse, grid, indices)
    all_ls = []
    crosssection_chunks = chunks
    if crosssection_chunks is None:
        tchunk = input_json.tchunk
        crosssection_chunks = (
            {"time": int(tchunk)} if tchunk is not None else {"time": 1}
        )
    for boundary, (boundaryfile, sel_index) in boundary_dict.items():
        boundary_files = sorted(glob.glob(boundaryfile.as_posix()))
        boundary_files = _select_boundary_files(boundary_files, sel_index, boundary)
        for var in ["u", "v", "w", "thl", "qt", "e12", *input_json.tracernames]:
            var_postfix = "0" if var == "e12" else ""
            with xr.open_mfdataset(
                boundary_files,
                combine="by_coords",
                chunks=crosssection_chunks,
                join="outer",
                data_vars="all",
                preprocess=partial(
                    _prepare_boundary_variable,
                    boundary=boundary,
                    variable=f"{var}{var_postfix}",
                ),
            ) as ds:
                all_ls.append(
                    load_any_boundary_var(
                        ds.sel(
                            {
                                dim: value
                                for dim, value in sel_index.items()
                                if dim in ds.dims
                            }
                        ),
                        var,
                        boundary=boundary,
                        grid=grid,
                        indices=indices,
                        var_postfix=var_postfix,
                    )
                )
    boundaries = xr.concat([ds_boundaries_timestep0, xr.merge(all_ls)], dim="time")
    _warn_if_suspicious_time_axis(boundaries, input_json)
    return boundaries
