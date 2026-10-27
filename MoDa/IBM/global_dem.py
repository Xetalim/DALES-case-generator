import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling

from MoDa.Geometry import GridDales
from MoDa.IO_helpers import ensure_sorted, get_reproject, raster_to_xarray

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
    zeroes_buffer: int = 5,
    subtract_dem_mode: bool = True,
    resampling_name: str = "bilinear",
) -> xr.DataArray:
    """Reproject a user DEM onto the DALES grid and normalize it for IBM.

    The source raster is assumed to contain elevation above mean sea level in
    its first band. After reprojection, the field is normalized to the local
    minimum and optionally mode-subtracted to mimic AHN-based IBM processing.
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

    ds[:, :] = ds[:, :] - float(ds.min())
    if subtract_dem_mode:
        values = np.round(np.ravel(ds.values)).astype(int)
        values = values[np.isfinite(values)]
        if values.size > 0:
            counts = np.bincount(values)
            mode = np.argmax(counts)
            ds[:, :] = ds[:, :] - mode
            ds[:, :] = np.where(ds[:, :] < 0, 0, ds[:, :])
            ds[:, :] = np.floor(ds[:, :])

    if zeroes_buffer > 0:
        ds[:zeroes_buffer, :] = 0
        ds[-zeroes_buffer:, :] = 0
        ds[:, :zeroes_buffer] = 0
        ds[:, -zeroes_buffer:] = 0

    ds[:, :] = ds[:, :] - float(ds.min())
    return ds
