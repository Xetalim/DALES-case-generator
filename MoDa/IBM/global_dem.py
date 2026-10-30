import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling

from moda.Geometry import GridDales
from moda.IO_helpers import ensure_sorted, get_reproject, raster_to_xarray

logger = logging.getLogger(__name__)

_RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
}


def _resolve_resampling(name: str) -> Resampling:
    key = str(name).strip().lower()
    if key not in _RESAMPLING_MAP:
        valid = ", ".join(sorted(_RESAMPLING_MAP))
        raise ValueError(
            f"Unsupported DEM resampling '{name}'. Expected one of: {valid}"
        )
    return _RESAMPLING_MAP[key]


def get_process_global_dem(
    grid: GridDales,
    dem_path: str | Path,
    resampling_name: str = "bilinear",
) -> xr.DataArray:
    """Reproject a user DEM onto the DALES grid.

    The source raster is assumed to contain elevation above mean sea level in
    its first band. Heights are relative to the minimum elevation. Use
    IBMPostProcessing for mode subtraction and edge buffers.
    """

    dem_path = Path(dem_path).expanduser()
    if not dem_path.exists():
        raise FileNotFoundError(f"Global DEM file not found: {dem_path}")

    resampling = _resolve_resampling(resampling_name)

    with rasterio.open(dem_path) as src:
        subset = src.read()
        profile = src.profile.copy()
        src_transform = src.transform
        src_crs = src.crs

    if src_crs is None:
        raise ValueError(f"Global DEM '{dem_path}' has no CRS metadata")

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "global_dem_reprojected.tif"
        get_reproject(
            grid,
            out_file,
            subset,
            src_transform=src_transform,
            src_crs=src_crs,
            profile=profile,
            resampling=resampling,
            fillnodata=False,
        )

        ds = raster_to_xarray(out_file, "bc_height")
        ds = ensure_sorted(ds)

    ds = ds.astype(float)

    nodata = profile.get("nodata")
    if nodata is not None:
        ds = ds.where(~np.isclose(ds, nodata), 0.0)

    ds = ds - float(ds.min())
    return ds
