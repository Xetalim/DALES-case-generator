import logging
from pathlib import Path

import pandas as pd
from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.LBC.nesting_idx import NestingIndices
from modular_dales.logging_wrapper import logwrap

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@logwrap
def load_any_boundary_var(
    ds,
    var: str,
    boundary: str,
    grid: GridDalesOpenBC,
    indices: "NestingIndices",
    isel: bool | None = False,
    expand_dims=False,
    expand_dims_time0=None,
    var_postfix="",
):
    """
    This function is a bit of a very generalised function to read in DALES staggered grid boundary variables.
    Should be made more readable or separated into 2 functions later.

    :param ds: The input xarray dataset containing variables on the DALES staggered grid.
    :param var: Which variable to get from the input ds
    :type var: str
    :param boundary: Which boundary to process , either north,south,west,east,top
    :type boundary: str
    :param grid: Grid object describing the output grid for open boundaries, which is different from the normal DALES grid.
    :type grid: GridDalesOpenBC
    :param indices: Indices describing where in the supergrid the edges of the current grid lie.
    :type indices: nesting_idx
    :param isel: If True, selects variable var at the correct boundary index in the host grid. If dict, select according to the dict.
    :type isel: Union[bool, None]
    :param expand_dims: Bool to enable expanding the time dimension with variable expand_dims_time0.
    :param expand_dims_time0: Variable to expand the time dimension with. Requires expand_dims=True
    :param var_postfix: Postfix for the var to load; somtimes the input variable has an extra long name.
    """
    var_dims_dic = {
        "u": {"x": "xm", "y": "yt", "z": "zt"},
        "v": {"x": "xt", "y": "ym", "z": "zt"},
        "w": {"x": "xt", "y": "yt", "z": "zm"},
        "default": {"x": "xt", "y": "yt", "z": "zt"},
    }
    assign_grid_dic = {
        "xt": grid.xt,
        "xm": grid.xm,
        "yt": grid.yt,
        "ym": grid.ym,
        "zt": grid.zt,
        "zm": grid.zm,
    }

    interp_grid_dic = assign_grid_dic

    boundary_var_interp_dic = {
        "west": {"w": ["y", "z"], "default": ["y", "z"]},
        "east": {"w": ["y", "z"], "default": ["y", "z"]},
        "south": {"w": ["x", "z"], "default": ["x", "z"]},
        "north": {"w": ["x", "z"], "default": ["x", "z"]},
        "top": {"default": ["x", "y"]},
    }
    boundary_var_assign_dic = {
        "west": {"default": ["y", "z"]},
        "east": {"default": ["y", "z"]},
        "south": {"default": ["x", "z"]},
        "north": {"default": ["x", "z"]},
        "top": {"default": ["x", "y"]},
    }

    boundary_var_isel_dim_dic = {
        "west": {"default": "x"},
        "east": {"default": "x"},
        "south": {"default": "y"},
        "north": {"default": "y"},
        "top": {"default": "z"},
    }

    boundary_var_isel_index_dic = {
        "east": {"default": indices.ix_east},
        "west": {"default": indices.ix_west},
        "south": {"default": indices.iy_south},
        "north": {"default": indices.iy_north},
        "top": {"default": grid.kmax - 1, "w": grid.kmax},
    }

    interp_kwargs_dic = {"w": {"fill_value": "extrapolate"}}

    # mapping from variable to (mapping from dimension name to dimension (x->xt))
    # we get var_dims : (x->xt,y->ym...)
    if var in var_dims_dic:
        var_dims = var_dims_dic[var]
    else:
        var_dims = var_dims_dic["default"]

    # which dimension to use to interpolate. for all variables this is y, x, or xy, but w has yz, xz and xy
    # at the end, var_interp_dims is a list of xt or xm or zt and so forth
    var_interp_dims = []
    if var in boundary_var_interp_dic[boundary]:
        for el in boundary_var_interp_dic[boundary][var]:
            var_interp_dims.append(var_dims[el])
    else:
        for el in boundary_var_interp_dic[boundary]["default"]:
            var_interp_dims.append(var_dims[el])

    # which dimension to use to assign coords to. for all variables this is yz, xz, or xy
    # at the end, var_assign_dims is a list of xt or xm or zt and so forth
    var_assign_dims = []
    if var in boundary_var_assign_dic[boundary]:
        for el in boundary_var_assign_dic[boundary][var]:
            var_assign_dims.append(var_dims[el])

    else:
        for el in boundary_var_assign_dic[boundary]["default"]:
            var_assign_dims.append(var_dims[el])

    # which dimension to use to select boundary indices from. for all variables this is yz, xz, or xy
    # at the end, var_assign_dims is a list of xt or xm or zt and so forth
    if var in boundary_var_isel_dim_dic[boundary]:
        var_isel_dim = var_dims[boundary_var_isel_dim_dic[boundary][var]]
    else:
        var_isel_dim = var_dims[boundary_var_isel_dim_dic[boundary]["default"]]

    if var in boundary_var_isel_index_dic[boundary]:
        var_isel_index = boundary_var_isel_index_dic[boundary][var]
    else:
        var_isel_index = boundary_var_isel_index_dic[boundary]["default"]

    # the precise coordinates to which to interpolate. for interpolating we subtract x0 and y0 as dales doesn't use those for output
    # for assigning coords, we do use x0 and y0 so we have a different dict for this.
    interp_coords = {dim: interp_grid_dic[dim] for dim in var_interp_dims}
    assign_coords = {dim: assign_grid_dic[dim] for dim in var_assign_dims}

    if var in interp_kwargs_dic:
        interp_kwargs = interp_kwargs_dic[var]
    else:
        interp_kwargs = None

    if isinstance(isel, bool):
        if isel:
            isel_dict = {var_isel_dim: var_isel_index}
        else:
            isel_dict = {}
    elif isinstance(isel, dict):
        isel_dict = isel
    else:
        raise TypeError(f"Unknown type for isel {isel}")
    interpolated_ds = (
        ds[f"{var}{var_postfix}"]
        .isel(isel_dict, drop=True)
        .interp(interp_coords, kwargs=interp_kwargs)
        .rename(f"{var}{boundary}")
        .assign_coords(assign_coords)
    )
    if expand_dims:
        if expand_dims_time0 is None:
            raise ValueError("Missing time0!")
        if not isel:
            logger.warning("Are you sure you want to do this?")
        if "time" in interpolated_ds.dims:
            pass
        else:
            interpolated_ds = interpolated_ds.expand_dims(
                {"time": [pd.Timestamp(expand_dims_time0)]}, axis=0
            )

    if interpolated_ds.isnull().sum() > 0:
        raise ValueError(f"Found NaN in dataset {boundary} {var}")
    if var == "qt":
        # make sure we don't have negative moisture
        interpolated_ds = interpolated_ds.where(interpolated_ds > 0, 0)
    return interpolated_ds


def get_boundary_dict(path, grid: GridDalesOpenBC, indices: "NestingIndices"):
    return {
        "west": (
            Path(path) / "crossyz.*.*.nc",
            {
                "xm": indices.supergrid.xm[indices.ix_west],
                "xt": indices.supergrid.xt[indices.ix_west],
            },
        ),
        "east": (
            Path(path) / "crossyz.*.*.nc",
            {
                "xm": indices.supergrid.xm[indices.ix_east],
                "xt": indices.supergrid.xt[indices.ix_east],
            },
        ),
        "south": (
            Path(path) / "crossxz.*.*.nc",
            {
                "ym": indices.supergrid.ym[indices.iy_south],
                "yt": indices.supergrid.yt[indices.iy_south],
            },
        ),
        "north": (
            Path(path) / "crossxz.*.*.nc",
            {
                "ym": indices.supergrid.ym[indices.iy_north],
                "yt": indices.supergrid.yt[indices.iy_north],
            },
        ),
        "top": (
            Path(path) / "crossxy.*.*.nc",
            {
                "zt": indices.supergrid.zt[indices.iz_top],
                "zm": indices.supergrid.zm[indices.iz_top],
            },
        ),
    }
    return {
        "west": (
            Path(path) / ".." / "crossyz.nc",
            {"xm": indices.ix_west + 1, "xt": indices.ix_west + 1},
        ),
        "east": (
            Path(path) / ".." / "crossyz.nc",
            {"xm": indices.ix_east + 1, "xt": indices.ix_east + 1},
        ),
        "south": (
            Path(path) / ".." / "crossxz.nc",
            {"ym": indices.iy_south + 1, "yt": indices.iy_south + 1},
        ),
        "north": (
            Path(path) / ".." / "crossxz.nc",
            {"ym": indices.iy_north + 1, "yt": indices.iy_north + 1},
        ),
        "top": (
            Path(path) / ".." / "crossxy.nc",
            {"zt": grid.kmax},
        ),
    }
