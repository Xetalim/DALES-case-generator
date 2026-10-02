import hashlib
import logging
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.warp import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box

from modular_dales.Geometry import GridDales
from modular_dales.IO_helpers import (
    cache_root,
    ensure_sorted,
    get_reproject,
    raster_to_xarray,
)

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def get_cached_AHN(cache_dir, grid: GridDales):
    """
    Download and cache AHN (Actueel Hoogtebestand Nederland) elevation data for a given grid.

    Retrieves AHN raster data from PDOK service, clips it to the specified grid bounds,
    and caches the result locally. Subsequent calls with identical parameters load from cache.

    Args:
        cache_dir (str or Path): Directory path where cached tiles are stored.
        grid (GridDales): Grid object containing projection and coordinate bounds (xt, yt, proj4).

    Returns:
        tuple: A tuple containing:
            - mosaic (ndarray): Merged elevation data as numpy array with shape (bands, height, width).
            - out_transform (Affine): Geospatial transformation matrix for the output raster.
            - dst_crs (str): Destination coordinate reference system (EPSG:28992).
            - profile (dict): Rasterio profile dictionary with metadata (driver, dimensions, dtype, crs, transform, compression).

    Notes:
        - Uses SHA256 hash of grid parameters as deterministic cache key.
        - Automatically handles single or multiple intersecting tiles via merging.
        - Converts nodata values to 0 in the output mosaic.
        - Requires GDAL/rasterio with VSICURL support for remote data access.
    """
    # -------------------------------
    # Parameters
    # -------------------------------
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dst_crs = "EPSG:28992"

    # ---- deterministic cache key ----
    key_str = (
        f"{grid.crs}|"
        f"{grid.xt[0]:.4f}|{grid.xt[-1]:.4f}|"
        f"{grid.yt[0]:.4f}|{grid.yt[-1]:.4f}|"
        "ahn|"
    )
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.tif"

    # ---- load from cache if present ----
    if cache_path.exists():
        logger.debug("Loading %s from cache", key_str)
        with rasterio.open(cache_path) as src:
            data = src.read()
            return data, src.transform, dst_crs, src.profile.copy()
    logger.debug("Downloading and processing %s", key_str)

    # ---- open COG and compute window ----
    url = "/vsicurl/https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/downloads/dsm_05m/kaartbladindex.json"
    wc_grid = gpd.read_file(url)

    # -------------------------------
    # Transform AOI bbox to tile CRS (EPSG:4326)
    # -------------------------------

    transformer = Transformer.from_crs(grid.crs, dst_crs, always_xy=True)
    xmin, ymin = transformer.transform(grid.xt[0], grid.yt[0])
    xmax, ymax = transformer.transform(grid.xt[-1], grid.yt[-1])

    # bbox in WorldCover CRS (EPSG:4326)
    aoi_geom = box(xmin, ymin, xmax, ymax)  # shapely box

    # intersect tiles
    intersecting_tiles = wc_grid[wc_grid.intersects(aoi_geom)]

    # -------------------------------
    # Open tiles individually via rasterio
    # -------------------------------
    tile_urls = [f"/vsicurl/{url}" for url in intersecting_tiles.url]

    datasets = [rasterio.open(url) for url in tile_urls]

    # -------------------------------
    # Merge into a single mosaic in memory
    # -------------------------------
    if len(datasets) == 1:
        window = from_bounds(xmin, ymin, xmax, ymax, transform=datasets[0].transform)
        mosaic = datasets[0].read(window=window)
        out_transform = datasets[0].window_transform(window)
        # direct window read (fast, safe)
    else:
        mosaic, out_transform = merge(
            datasets,
            resampling=rasterio.enums.Resampling.nearest,
            bounds=aoi_geom.bounds,
        )
    mosaic[mosaic == datasets[0].nodata] = 0  # Convert nodata to NaN
    # Convert masked nodata to NaN

    profile = dict(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=mosaic.shape[0],  # number of bands
        dtype=mosaic.dtype,
        crs=dst_crs,
        transform=out_transform,
        compress="deflate",  # optional compression
    )
    with rasterio.open(cache_path, "w", **profile) as dst:
        dst.write(mosaic)

    return mosaic, out_transform, dst_crs, profile.copy()


def get_ahn(grid: GridDales, out_file):
    """
    Retrieve and reproject AHN (Actueel Hoogtebestand Nederland) data for a given grid.
    Fetches cached AHN data for the specified grid and reprojects it to the output file
    using mode resampling without filling nodata values.
    Args:
        grid (GridDales): The grid object defining the spatial extent and properties.
        out_file: The output file path where the reprojected AHN data will be written.
    Returns:
        None
    """

    mosaic, out_transform, mosaic_crs, mosaic_profile = get_cached_AHN(
        cache_root("ahn"), grid
    )
    logger.warning("Do we really want to use mode?")
    get_reproject(
        grid,
        out_file,
        mosaic,
        src_transform=out_transform,
        src_crs=mosaic_crs,
        profile=mosaic_profile,
        resampling=Resampling.mode,
        fillnodata=False,
    )


def get_process_ahn(grid, zeroes_buffer=5, subtract_ahn_mode=True):
    """
    Process AHN (Actueel Hoogtebestand Nederland) data and prepare it for use.
    Downloads AHN data for the specified grid, normalizes the height values,
    optionally subtracts the mode value, applies zero buffers at the edges,
    and returns a normalized xarray Dataset.
    Parameters
    ----------
    grid : Grid-like object
        The geographic grid defining the area for which to download AHN data.
    zeroes_buffer : int, optional
        Number of pixels from each edge to set to zero (default: 5).
    subtract_ahn_mode : bool, optional
        If True, subtract the mode (most frequent elevation value) from the dataset
        and clip negative values to zero (default: True).
    Returns
    -------
    xarray.Dataset
        Processed elevation data with variable 'bc_height', normalized to start at
        zero, with buffer zones set to zero and optionally mode-adjusted.
    Notes
    -----
    - Downloads AHN data to a temporary directory
    - Normalizes data by subtracting the minimum value
    - Sets border regions to zero based on zeroes_buffer parameter
    - Ensures data is floor-divided to integer heights
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)
        ahn_file = path / "ahn.tif"
        get_ahn(grid, ahn_file)

        ds = raster_to_xarray(ahn_file, "bc_height")
        ds = ensure_sorted(ds)
        ds[:, :] = ds[:, :] - ds.min()
        if subtract_ahn_mode:
            a = np.round(np.ravel(ds.values)).astype(int)
            counts = np.bincount(a)
            mode = np.argmax(counts)
            ds[:, :] = ds[:, :] - mode
            ds[:, :] = np.where(ds[:, :] < 0, 0, ds[:, :])
            ds[:, :] = np.floor(ds[:, :])
        ds[:zeroes_buffer, :] = 0
        ds[-zeroes_buffer:, :] = 0
        ds[:, :zeroes_buffer] = 0
        ds[:, -zeroes_buffer:] = 0
        ds[:, :] = ds[:, :] - ds.min()
    return ds
