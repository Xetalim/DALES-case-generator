import hashlib
import logging
import os
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from modular_dales.Geometry.GridDales import GridDales
from modular_dales.IO_helpers import get_reproject
from modular_dales.IO_helpers.external_data_cache import cache_root

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def cached_cog_subset_from_url(
    cog_url,
    grid,
    cache_dir,
    pad=10,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- deterministic cache key ----
    key_str = (
        f"{cog_url}|"
        f"{grid.crs}|"
        f"{grid.xt[0]:.4f}|{grid.xt[-1]:.4f}|"
        f"{grid.yt[0]:.4f}|{grid.yt[-1]:.4f}|"
        f"{pad}"
    )
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.tif"

    # ---- load from cache if present ----
    if cache_path.exists():
        logger.debug("Loading %s from cache" % key_str)
        with rasterio.open(cache_path) as src:
            data = src.read()
            return data, src.transform, src.crs, src.profile.copy()
    logger.debug("Downloading and processing %s" % key_str)

    # ---- open COG and compute window ----
    with rasterio.open(cog_url) as ds:
        transformer = Transformer.from_crs(grid.crs, ds.crs, always_xy=True)

        xmin, ymin = transformer.transform(grid.xt[0], grid.yt[0])
        xmax, ymax = transformer.transform(grid.xt[-1], grid.yt[-1])
        logger.debug(
            "Original grid bounds: xt[0]=%f, xt[-1]=%f, yt[0]=%f, yt[-1]=%f",
            grid.xt[0],
            grid.xt[-1],
            grid.yt[0],
            grid.yt[-1],
        )
        logger.debug("Transformed grid bounds to COG CRS: %s", (xmin, ymin, xmax, ymax))
        logger.debug("Transform is %s", ds.transform)
        window = (
            from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
            .round_offsets()
            .round_lengths()
        )

        window = Window(
            col_off=max(window.col_off - pad, 0),
            row_off=max(window.row_off - pad, 0),
            width=min(window.width + 2 * pad, ds.width),
            height=min(window.height + 2 * pad, ds.height),
        )

        data = ds.read(window=window)
        transform = rasterio.windows.transform(window, ds.transform)

        profile = ds.profile.copy()
        profile.update(
            width=window.width,
            height=window.height,
            transform=transform,
            count=data.shape[0],
        )

        with rasterio.open(cache_path, "w", **profile) as dst:
            dst.write(data)

        return data, transform, ds.crs, ds.profile.copy()


def get_cog(grid: GridDales, out_file):

    cog_url = "https://lcz-generator.rub.de/cogs/lcz_filter_v3_cog.tif"

    subset, src_transform, src_crs, profile = cached_cog_subset_from_url(
        cog_url, grid, cache_root("cog"), pad=10
    )
    get_reproject(
        grid,
        out_file,
        subset,
        src_transform,
        src_crs,
        profile,
        resampling=Resampling.mode,
        fillnodata=True,
    )


def get_cached_esa(cache_dir, grid: GridDales):
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    # -------------------------------
    # Parameters
    # -------------------------------
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    year = 2021
    version = {2020: "v100", 2021: "v200"}[year]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dst_crs = "EPSG:4326"

    # ---- deterministic cache key ----
    key_str = (
        f"{grid.crs}|"
        f"{grid.xt[0]:.4f}|{grid.xt[-1]:.4f}|"
        f"{grid.yt[0]:.4f}|{grid.yt[-1]:.4f}|"
    )
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.tif"

    # ---- load from cache if present ----
    if cache_path.exists():
        logger.debug("Loading %s from cache" % key_str)
        with rasterio.open(cache_path) as src:
            data = src.read()
            return data, src.transform, dst_crs, src.profile.copy()
    logger.debug("Downloading and processing %s" % key_str)

    # ---- open COG and compute window ----
    url = f"/vsicurl/{s3_url_prefix}/esa_worldcover_grid.geojson"
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
    tile_urls = [
        f"/vsicurl/{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        for tile in intersecting_tiles.ll_tile
    ]

    datasets = [rasterio.open(url) for url in tile_urls]

    # -------------------------------
    # Merge into a single mosaic in memory
    # -------------------------------
    mosaic, out_transform = merge(
        datasets, resampling=rasterio.enums.Resampling.nearest
    )
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


def get_esa(grid: GridDales, out_file):
    mosaic, out_transform, mosaic_crs, mosaic_profile = get_cached_esa(
        cache_root("esa_worldcover"), grid
    )
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


def query_zip_virtual_mosaic(zip_file_path, grid, pad=0):
    """
    Query a virtual mosaic of all TIFFs in a ZIP given a grid, returns data for the requested window.
    Reads only the overlapping pixels (lazy, efficient).

    Parameters
    ----------
    zip_file_path : str
        Path to the ZIP file containing TIFFs.
    grid : object
        Object with attributes xt, yt, proj4 (grid coordinates & CRS).
    pad : int
        Number of extra pixels to pad around the window.

    Returns
    -------
    data : np.ndarray
        Array of shape (num_bands, height, width) covering the grid window.
    """
    # List TIFFs inside ZIP
    # https://zenodo.org/records/11517903
    with ZipFile(zip_file_path, "r") as zf:
        tiff_files = [f for f in zf.namelist() if f.lower().endswith(".tif")]
    if not tiff_files:
        raise ValueError("No TIFFs found in ZIP")

    # Build vsizip paths
    vsizip_paths = [f"/vsizip/{zip_file_path}/{f}" for f in tiff_files]

    # Open all datasets as WarpedVRT (lazy)
    datasets = [WarpedVRT(rasterio.open(p)) for p in vsizip_paths]

    # Transform grid coordinates to CRS of the first dataset
    transformer = Transformer.from_crs(grid.crs, datasets[0].crs, always_xy=True)
    xmin, ymin = transformer.transform(grid.xt[0], grid.yt[0])
    xmax, ymax = transformer.transform(grid.xt[-1], grid.yt[-1])

    # Merge datasets into virtual mosaic (lazy)
    mosaic, out_transform = merge(
        datasets, bounds=(xmin, ymin, xmax, ymax), res=datasets[0].res
    )

    # Compute window in mosaic coordinates
    window = (
        from_bounds(xmin, ymin, xmax, ymax, transform=out_transform)
        .round_offsets()
        .round_lengths()
    )

    # Apply padding
    row_off = max(int(window.row_off - pad), 0)
    col_off = max(int(window.col_off - pad), 0)
    height = min(int(window.height + 2 * pad), mosaic.shape[1] - row_off)
    width = min(int(window.width + 2 * pad), mosaic.shape[2] - col_off)

    # Slice mosaic to the window
    data = mosaic[:, row_off : row_off + height, col_off : col_off + width]

    # Close all datasets
    for ds in datasets:
        ds.close()

    return data
