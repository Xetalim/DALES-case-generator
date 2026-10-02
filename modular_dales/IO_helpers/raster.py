import logging
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import rasterio.fill
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

if TYPE_CHECKING:
    from modular_dales.Geometry import GridDales

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def raster_to_xarray(tif_path, name):
    """
    Load a raster file and convert it to an xarray DataArray.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF raster file.
    name : str
        Name to assign to the resulting DataArray.

    Returns
    -------
    xr.DataArray
        DataArray containing the raster data with the specified name.
    """
    # with rasterio.open(tif_path) as ds:
    #     arr = ds.read(1).astype(float)
    return xr.open_dataarray(
        tif_path, engine="rasterio", backend_kwargs={"band_as_variable": True}
    ).rename(name)


def ensure_sorted(ds):
    """
    Ensure that the x and y coordinates of a dataset are sorted in ascending order.

    If the x or y coordinates are in descending order, the dataset is sorted along
    that dimension to ensure consistent ordering.

    Args:
        ds: An xarray Dataset with x and y dimensions.

    Returns:
        The dataset with x and y coordinates sorted in ascending order.
    """
    x = ds.x.values
    if x[0] > x[-1]:
        ds = ds.sortby("x")
    y = ds.y.values
    if y[0] > y[-1]:
        ds = ds.sortby("y")
    return ds


def get_reproject(
    grid: "GridDales",
    out_file,
    subset,
    src_transform,
    src_crs,
    profile,
    resampling=Resampling.mode,
    fillnodata=True,
):
    """
    Reproject a raster subset to a target grid and save to file.

    Reprojects raster data from a source CRS to a target CRS defined by the
    GridDales object, filling no-data values if specified, and writes the
    reprojected raster to an output file.

    Args:
        grid (GridDales): Target grid object containing coordinate system (proj4),
            x-coordinates (xt), and y-coordinates (yt).
        out_file (str or Path): Output file path where the reprojected raster
            will be written.
        subset (np.ndarray): Source raster data to be reprojected.
        src_transform (rasterio.transform.Affine): Affine transformation matrix
            of the source raster.
        src_crs (rasterio.crs.CRS): Coordinate reference system of the source
            raster.
        profile (dict): Rasterio profile dictionary containing metadata for the
            source raster (e.g., nodata value, dtype).
        resampling (rasterio.enums.Resampling, optional): Resampling method to use
            during reprojection. Defaults to Resampling.mode (nearest neighbor).
        fillnodata (bool, optional): Whether to fill no-data values in the
            reprojected raster. Defaults to True.
    """
    dst_crs = CRS(grid.crs)

    dx = grid.xt[1] - grid.xt[0]
    dy = grid.yt[1] - grid.yt[0]

    dst_transform = from_origin(grid.xt[0], grid.yt[-1], dx, dy)

    dst_height = len(grid.yt)
    dst_width = len(grid.xt)

    dst = np.empty((1, dst_height, dst_width), dtype=subset.dtype)

    reproject(
        source=subset,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )

    out_transform = dst_transform

    out_profile = profile.copy()
    out_profile.update(
        {
            "count": dst.shape[0],
            "height": dst.shape[1],
            "width": dst.shape[2],
            "transform": out_transform,
        }
    )

    if fillnodata and np.any(np.isclose(dst, profile["nodata"])):
        logger.warning("NANs found in raster. Filling in with rasterio.fill!!")
        dst = rasterio.fill.fillnodata(
            dst, mask=np.logical_not(np.isclose(dst, profile["nodata"]))
        )

    with rasterio.open(out_file, "w", **out_profile) as ds:
        ds.write(dst)


def fix_lambert_offsets(ds):
    """Fix Lambert false_easting/false_northing by shifting x/y coordinates.

    With ``decode_coords='all'`` xarray may change how CF grid mapping
    information is represented, so this helper tries multiple strategies
    to locate the grid-mapping variable:

    1. Look for a ``grid_mapping`` attribute on data variables.
    2. Fallback: search all variables for typical grid-mapping attributes
       (``grid_mapping_name``, ``false_easting``, ``false_northing``).

    If no suitable variable is found or no offsets are present, the input
    dataset is returned unchanged.
    """

    # 1) Try the classic CF pattern: data_var.attrs['grid_mapping']
    gridmap_name = None
    for v in ds.data_vars:
        gm_name = ds[v].attrs.get("grid_mapping")
        if gm_name:
            gridmap_name = gm_name
            break

    # 2) Fallback for decode_coords='all' or slightly non-standard files:
    if gridmap_name is None:
        for v in ds.variables:
            attrs = ds[v].attrs
            if (
                "grid_mapping_name" in attrs
                or "false_easting" in attrs
                or "false_northing" in attrs
            ):
                gridmap_name = v
                break

    if gridmap_name is None:
        logger.warning(
            "fix_lambert_offsets: no grid mapping variable found; skipping offset fix"
        )
        return ds

    gm = ds[gridmap_name]

    fe = float(gm.attrs.get("false_easting", 0.0) or 0.0)
    fn = float(gm.attrs.get("false_northing", 0.0) or 0.0)

    # Nothing to do if there are no offsets
    if fe == 0.0 and fn == 0.0:
        return ds

    # create absolute projected coordinates (try coords first, then variables)
    x = ds.coords["x"] if "x" in ds.coords else ds["x"]
    y = ds.coords["y"] if "y" in ds.coords else ds["y"]

    ds = ds.assign_coords(x=ds["x"] - fe, y=ds["y"] - fn)
    ds["x"].attrs = x.attrs
    ds["y"].attrs = y.attrs
    # remove offsets to avoid double-counting later
    ds[gridmap_name].attrs["false_easting"] = 0.0
    ds[gridmap_name].attrs["false_northing"] = 0.0

    return ds
