"""Read surface-cover and morphology data from a user-supplied NetCDF file.

Expected variables
------------------
Required cover fractions (all in [0, 1]):
  FRAC_WATER   – open-ocean fraction  → `aqu` / `wat` lu-type
  FRAC_SEA     – sea fraction          → merged with FRAC_WATER into aquatic cover
  FRAC_NATURE  – natural land fraction → distributed via ESA WorldCover ifs_land_cover
  FRAC_TOWN    – urban fraction        → `slb` lu-type; drives SLuRB inslurb file

Optional morphological fields used to fill inslurb.nc (all spatial arrays):
  D_Z0_town    → z0_urb   (urban roughness length,  m)
  D_BLD        → f_bld    (building plan-area fraction, –)
  D_BLD_HEIG   → h_bld    (mean building height, m)
  WALL_O_HOR   → hw_can   = 0.5 * WALL_O_HOR / (1 - D_BLD)  (canyon aspect ratio, –)

The file must have x/y (or lon/lat) coordinates.  It will be bilinearly
interpolated onto the DALES grid using rasterio/get_reproject.

The returned xarray Dataset follows the same convention used by
``get_from_LCZ.build_land_surface_dataset`` so it can be consumed by
``LSM_output_dales.apply_from_netcdf``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.warp import Resampling

from modular_dales.IO_helpers import raster_to_xarray
from modular_dales.IO_helpers.raster import fix_lambert_offsets, get_reproject
from modular_dales.Surface.LSM.LCZ.downloaders import get_esa
from modular_dales.Surface.LSM.LCZ.get_from_LCZ import (
    map_esa_to_ifs,
    map_soil_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable name mapping
# ---------------------------------------------------------------------------
NC_COVER_VARS = {
    "FRAC_WATER": "frac_water",
    "FRAC_SEA": "frac_sea",
    "FRAC_NATURE": "frac_nature",
    "FRAC_TOWN": "frac_town",
}

NC_MORPH_VARS = ["D_Z0_town", "D_BLD", "D_BLD_HEIG", "WALL_O_HOR"]


def _get_src_crs(ds_in: xr.Dataset) -> str | None:
    """Best-effort extraction of CRS string from an xarray Dataset."""
    # 1. grid_mapping variable (CF convention)
    for var in ds_in.data_vars:
        gm = ds_in[var].attrs.get("grid_mapping")
        if gm and gm in ds_in:
            crs_wkt = ds_in[gm].attrs.get("crs_wkt") or ds_in[gm].attrs.get("spatial_ref")
            if crs_wkt:
                return crs_wkt
    # 2. spatial_ref coordinate
    if "spatial_ref" in ds_in.coords:
        crs_wkt = ds_in.coords["spatial_ref"].attrs.get("crs_wkt")
        if crs_wkt:
            return crs_wkt
    # 3. dataset-level attribute
    for attr in ("crs", "proj4", "projection"):
        val = ds_in.attrs.get(attr)
        if val:
            return val
    return None


def _open_nc_normalised(nc_path: Path) -> xr.Dataset:
    """Open the user NetCDF, fix Lambert offsets, and normalise dim names to x/y."""
    ds = xr.open_dataset(nc_path, decode_coords="all")
    ds = fix_lambert_offsets(ds)

    rename_map = {}
    for dim in ds.dims:
        if dim.lower() in ("lon", "longitude"):
            rename_map[dim] = "x"
        elif dim.lower() in ("lat", "latitude"):
            rename_map[dim] = "y"
    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def _reproject_da(
    da: xr.DataArray,
    src_crs: str,
    grid,
    out_tif: Path,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Reproject a 2-D DataArray (dims 'y', 'x') onto ``grid`` via rasterio.

    Returns a (jtot, itot) float32 array.
    """
    vals = np.asarray(da.values, dtype=np.float32)
    if vals.ndim == 2:
        vals = vals[np.newaxis, :, :]  # (1, y, x)

    x_vals = da.coords["x"].values.astype(float)
    y_vals = da.coords["y"].values.astype(float)

    # from_bounds expects (west, south, east, north, width, height)
    src_transform = from_bounds(
        x_vals.min(), y_vals.min(), x_vals.max(), y_vals.max(),
        vals.shape[2], vals.shape[1],
    )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": float("nan"),
        "count": 1,
        "height": vals.shape[1],
        "width": vals.shape[2],
        "transform": src_transform,
        "crs": src_crs,
    }

    get_reproject(grid, out_tif, vals, src_transform, src_crs, profile,
                  resampling=resampling, fillnodata=True)

    with rasterio.open(out_tif) as src:
        result = src.read(1).astype(np.float32)

    return result


def _derive_slurb_fields(raw: dict) -> dict:
    """Convert raw NetCDF morph fields to SLuRB-named arrays."""
    out = {}
    if "D_Z0_town" in raw:
        out["z0_urb"] = raw["D_Z0_town"]
    if "D_BLD" in raw:
        out["f_bld"] = raw["D_BLD"]
    if "D_BLD_HEIG" in raw:
        out["h_bld"] = raw["D_BLD_HEIG"]
    if "WALL_O_HOR" in raw and "D_BLD" in raw:
        denom = np.maximum(1.0 - raw["D_BLD"], 1e-6)
        out["hw_can"] = 0.5 * raw["WALL_O_HOR"] / denom
    elif "WALL_O_HOR" in raw:
        logger.warning(
            "from_netcdf: WALL_O_HOR present but D_BLD missing; "
            "hw_can = WALL_O_HOR / 2 (no building-fraction correction)"
        )
        out["hw_can"] = raw["WALL_O_HOR"] / 2.0
    return out


def load_from_netcdf(
    nc_path,
    grid,
    esa_cache_dir: Path | None = None,
) -> xr.Dataset:
    """
    Load a surface-cover + morphology NetCDF and return an xarray Dataset
    projected onto the DALES grid.

    The returned dataset contains:

    Cover fields (all in [0, 1], shape (jtot, itot)):
      ``frac_water``, ``frac_sea``, ``frac_town``, ``frac_nature``

    ESA-derived fields for land-use sub-type distribution:
      ``ifs_land_cover``, ``index_soil``

    Morphology / SLuRB fields (where source variables were present):
      ``z0_urb``, ``f_bld``, ``h_bld``, ``hw_can``

    Parameters
    ----------
    nc_path:
        Path to the input NetCDF file.
    grid:
        DALES grid to project onto.
    esa_cache_dir:
        Directory used *only* to cache the reprojected ESA tile between runs.
        When *None* a fresh temporary directory is used.
    """
    nc_path = Path(nc_path)
    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    logger.info("from_netcdf: loading %s", nc_path)
    ds_in = _open_nc_normalised(nc_path)

    src_crs = _get_src_crs(ds_in)
    if src_crs is None:
        logger.warning(
            "from_netcdf: no CRS found in '%s'; assuming same projection as DALES grid",
            nc_path,
        )
        src_crs = grid.crs

    data_vars: dict = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- cover fractions -------------------------------------------------
        for nc_name, internal_name in NC_COVER_VARS.items():
            if nc_name not in ds_in:
                logger.warning("from_netcdf: '%s' not found in file", nc_name)
                continue
            da = ds_in[nc_name].squeeze()
            if "x" not in da.dims or "y" not in da.dims:
                logger.warning(
                    "from_netcdf: '%s' has no x/y dims after normalisation; skipping",
                    nc_name,
                )
                continue
            out_tif = tmp / f"{internal_name}.tif"
            arr = _reproject_da(da, src_crs, grid, out_tif, Resampling.bilinear)
            data_vars[internal_name] = xr.DataArray(
                arr, dims=("y", "x"),
                coords={"y": grid.yt, "x": grid.xt},
                name=internal_name,
            )
            logger.debug("from_netcdf: reprojected cover field '%s'", nc_name)

        required = ["frac_water", "frac_sea", "frac_nature", "frac_town"]
        missing = [r for r in required if r not in data_vars]
        if missing:
            raise ValueError(
                f"from_netcdf: required cover variable(s) not found or missing "
                f"x/y dims in input file: {missing}"
            )

        # --- morphological fields --------------------------------------------
        raw_morph: dict = {}
        for nc_name in NC_MORPH_VARS:
            if nc_name not in ds_in:
                continue
            da = ds_in[nc_name].squeeze()
            if "x" not in da.dims or "y" not in da.dims:
                continue
            out_tif = tmp / f"{nc_name}.tif"
            raw_morph[nc_name] = _reproject_da(
                da, src_crs, grid, out_tif, Resampling.bilinear
            )
            logger.debug("from_netcdf: reprojected morphology field '%s'", nc_name)

        slurb_fields = _derive_slurb_fields(raw_morph)
        for name, arr in slurb_fields.items():
            data_vars[name] = xr.DataArray(
                arr, dims=("y", "x"),
                coords={"y": grid.yt, "x": grid.xt},
                name=name,
            )

    # --- ESA WorldCover for natural land-cover sub-type distribution ---------
    if esa_cache_dir is not None:
        esa_cache_dir = Path(esa_cache_dir)
        esa_cache_dir.mkdir(parents=True, exist_ok=True)
        esa_tif = esa_cache_dir / "esa_from_netcdf.tif"
    else:
        _esa_tmp = tempfile.mkdtemp()
        esa_tif = Path(_esa_tmp) / "esa_from_netcdf.tif"

    logger.info("from_netcdf: fetching ESA WorldCover for nature land-use types")
    get_esa(grid, esa_tif)
    esa_da = raster_to_xarray(esa_tif, "esa_worldcover")

    ifs_da = map_esa_to_ifs(esa_da)
    soil_da = map_soil_type(esa_da)

    data_vars["ifs_land_cover"] = ifs_da.assign_coords({"y": grid.yt, "x": grid.xt})
    data_vars["index_soil"] = soil_da.assign_coords({"y": grid.yt, "x": grid.xt})

    ds_out = xr.Dataset(data_vars)
    logger.info("from_netcdf: dataset ready with %d variables", len(ds_out.data_vars))
    return ds_out
