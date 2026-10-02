"""
Adapter module for nesting DALES inside HARMONIE using KNMI operational GRIB output.

This module reads pre-converted NetCDF files originating from KNMI's operational
HARMONIE output (archives N55ML and/or N20/N55), remaps variable names and fills
missing fields, and produces xarray Datasets in the exact format expected by
``prep_harmonie.harmoniePrepper``.

Expected input: NetCDF files produced by converting HARMONIE GRIB output
(e.g. via ``grib_to_netcdf`` or ``cdo``). The module does NOT read GRIB directly.

Supported archives
------------------
- **N55ML (P5)**: 3D fields on 90 hybrid levels, rotated_ll grid (676×564).
  Contains: ``u``, ``v``, ``t``, ``q`` on hybrid levels; ``pres``, ``z``, ``t``
  at surface.  **Primary source for 3D model-level data.**
- **N55 (P3)**: Surface/diagnostic fields on rotated_ll grid. Supplements N55ML
  with surface variables (``t(2m)``, ``r(2m)``, ``pres``, ``shf``, ``lhe``, etc.).
- **N20 (P1)**: Surface/diagnostic fields on regular_ll grid (390×390).
  Can be used as surface supplement when rotated_ll is not required.

Missing variables handled
-------------------------
- ``wa`` (vertical velocity): set to zero by default, or optionally derived
  from horizontal wind divergence via mass continuity.
- ``clw`` (cloud liquid water): set to zero (qt = q).
- ``huss`` (2m specific humidity): derived from 2m dew point temperature
  (``td``), or from 2m relative humidity (``r_2``) and temperature.
- ``synturb`` variables (``tke``, ``tauu``, ``tauv``): not available in
  KNMI GRIB output — synturb mode is not supported.

CDO conversion notes
--------------------
After ``cdo -f nc4 copy`` conversion, CDO renames duplicate GRIB shortNames
by appending ``_2``, ``_3`` suffixes when the same shortName appears on
different vertical coordinate types.  For the N55ML file this means:

- ``t`` = temperature on 90 hybrid levels,  ``t_2`` = surface temperature
- ``pres`` = surface pressure (at height=0)

For the N55 (surface) file:

- ``pres`` = MSL pressure (at alt=0),  ``pres_2`` = surface pressure
- ``t_2`` = temperature at 6 height levels (0–300 m) — 2m value selected
- ``r_2`` = 2m relative humidity,  ``td`` = 2m dew point temperature

Grid dimensions are named ``rlon``/``rlat`` for rotated latitude-longitude.

Pre-conversion documentation
----------------------------
Convert GRIB archives to NetCDF before use. Example with ``grib_to_netcdf``::

    grib_to_netcdf -o HA43_N55ML_*.nc HA43_N55ML_*_GB

Or with CDO::

    cdo -f nc copy HA43_N55ML_*_GB HA43_N55ML_*.nc

The resulting NetCDF files must preserve:
- Dimensions: ``time``, ``lev`` (or ``hybrid``/``level``), ``rlat``/``rlon``
  (CDO default) or ``y``/``x`` (other tools)
- CRS metadata (grid_mapping variable with projection parameters)
- Variable short names as assigned by the GRIB encoding
"""

import datetime as dt
import glob
import logging
import os
from pathlib import Path

import dask
import numpy as np
import rioxarray  # noqa: F401 — activates .rio accessor on xr objects
import xarray as xr
from modular_dales.IO_helpers.raster import fix_lambert_offsets
from modular_dales.LBC.nest_dales_in_HARMONIE import hybrid_levels, prep_harmonie
from modular_dales.LBC.nest_dales_in_HARMONIE.Transform import Transform
from modular_dales.logging_wrapper import logwrap
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

# Physical constants (same as prep_harmonie)
Rd = 287.04
Rv = 461.5
cp = 1004.0


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------


def _saturation_vapour_pressure(T):
    """Tetens formula for saturation vapour pressure over liquid water.

    Parameters
    ----------
    T : array-like
        Temperature in Kelvin.

    Returns
    -------
    e_sat : same type as T
        Saturation vapour pressure in Pa.
    """
    T_C = T - 273.15
    return 610.78 * np.exp(17.27 * T_C / (T_C + 237.3))


def q_from_rh(rh, T, p):
    """Convert relative humidity to specific humidity.

    Parameters
    ----------
    rh : array-like
        Relative humidity (0-1 fraction).
    T : array-like
        Temperature in Kelvin.
    p : array-like
        Pressure in Pa.

    Returns
    -------
    q : same type as inputs
        Specific humidity in kg/kg.
    """
    e_sat = _saturation_vapour_pressure(T)
    e = rh * e_sat
    q = (Rd / Rv) * e / (p - (1 - Rd / Rv) * e)
    return q


def q_from_td(td, p):
    """Convert dew point temperature to specific humidity.

    The saturation vapour pressure evaluated at the dew point equals
    the actual vapour pressure.

    Parameters
    ----------
    td : array-like
        Dew point temperature in Kelvin.
    p : array-like
        Pressure in Pa.

    Returns
    -------
    q : same type as inputs
        Specific humidity in kg/kg.
    """
    e = _saturation_vapour_pressure(td)
    q = (Rd / Rv) * e / (p - (1 - Rd / Rv) * e)
    return q


def w_from_continuity(u, v, p_sfc, ahalf, bhalfs):
    """Derive vertical velocity from horizontal wind divergence.

    Uses the incompressible continuity equation on hybrid levels:
    ``dw/dz = -(du/dx + dv/dy)`` integrated from the surface (w=0) upward.

    **Important**: ``u`` and ``v`` must have x/y coordinates in meters
    (i.e. call this *after* reprojection to a metric CRS) so that
    ``differentiate("x")`` produces correct (m/s)/m = 1/s units.

    Parameters
    ----------
    u, v : xr.DataArray
        Wind components with dims (time, lev, y, x) and metric x/y.
    p_sfc : xr.DataArray
        Surface pressure with dims (time, y, x).
    ahalf, bhalfs : array-like
        Hybrid half-level coefficients (n_lev + 1 values).

    Returns
    -------
    w : xr.DataArray
        Estimated vertical velocity with same dims as u.
    """
    # Horizontal divergence  [1/s]
    # x and y MUST be in meters at this point.
    dudx = u.differentiate("x")
    dvdy = v.differentiate("y")
    div = dudx + dvdy

    # Layer thickness from hydrostatic relation dp = -rho g dz
    A = xr.DataArray(ahalf, dims=["lev_h"])
    B = xr.DataArray(bhalfs, dims=["lev_h"])
    ph = A + B * p_sfc  # half-level pressure
    dp = ph.diff("lev_h").rename({"lev_h": "lev"})  # pressure thickness
    dp = dp.assign_coords(lev=u.lev)

    grav = 9.81
    T_approx = 270.0
    # Full-level pressure (average of bounding half-levels)
    p_full = 0.5 * (
        ph.isel(lev_h=slice(None, -1)).values + ph.isel(lev_h=slice(1, None)).values
    )
    p_full = xr.DataArray(p_full, dims=dp.dims, coords=dp.coords)
    rho = p_full / (Rd * T_approx)  # approximate density
    dz = dp.astype(float) / (rho * grav)  # layer thickness [m]

    # KNMI hybrid levels run from 1 (top) to 90 (bottom).
    # Ensure we integrate from the surface (highest lev index) upward.
    ascending = float(u.lev[0]) < float(u.lev[-1])  # True if 1…90
    if ascending:
        # Reverse so surface is first, integrate, then reverse back
        dw = (-div * dz).isel(lev=slice(None, None, -1))
        w = dw.cumsum("lev").isel(lev=slice(None, None, -1))
    else:
        w = (-div * dz).cumsum("lev")

    return w.transpose(*u.dims)


# ---------------------------------------------------------------------------
# Coordinate reprojection: latlon/rotated-latlon → Lambert conformal (meters)
# ---------------------------------------------------------------------------


def _detect_source_crs(ds):
    """Detect the CRS of the source dataset.

    Returns a pyproj CRS object.  Tries rioxarray first, then falls back
    to inspecting the grid_mapping variable attributes.
    """
    # rioxarray path
    try:
        src_crs = ds.rio.crs
        if src_crs is not None:
            return CRS(src_crs)
    except (AttributeError, KeyError):
        pass

    # Manual: look for a grid_mapping variable
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if "grid_mapping_name" in attrs:
            return CRS.from_cf(attrs)
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            gm_name = ds[var].attrs["grid_mapping"]
            if gm_name in ds:
                return CRS.from_cf(ds[gm_name].attrs)

    # Check for CDO-style rotated grid mapping variable by name
    for name in ("rotated_latitude_longitude", "rotated_pole"):
        if name in ds:
            gm_attrs = ds[name].attrs
            if "grid_mapping_name" in gm_attrs:
                logger.info("Found rotated grid mapping variable '%s'", name)
                return CRS.from_cf(gm_attrs)

    # If x/y look like degrees, assume WGS84 lat/lon.
    # Only use this fallback if we are sure this is NOT a rotated grid —
    # rotated grids also use degree-like coordinates but with a shifted pole.
    x_vals = ds["x"].values
    if np.all(np.abs(x_vals) <= 360):
        logger.warning(
            "x coordinates look like degrees and no grid_mapping found; "
            "assuming EPSG:4326.  If this is a rotated grid, results will "
            "be WRONG — ensure the NetCDF contains grid_mapping metadata."
        )
        return CRS("EPSG:4326")

    raise ValueError(
        "Cannot determine source CRS from dataset. "
        "Ensure the NetCDF contains CRS metadata (grid_mapping)."
    )


def _build_target_grid(grid, buffer, resolution):
    """Build 1-D x/y coordinate arrays for the target Lambert grid.

    Parameters
    ----------
    grid : GridDalesOpenBC
        DALES grid object (x0, y0, xsize, ysize in meters).
    buffer : float
        Extra margin in meters around the domain.
    resolution : float
        Target grid spacing in meters.

    Returns
    -------
    x_target, y_target : np.ndarray
        1-D arrays of x and y coordinates in the target projection.
    """
    x_min = grid.x0 - buffer
    x_max = grid.x0 + grid.xsize + buffer
    y_min = grid.y0 - buffer
    y_max = grid.y0 + grid.ysize + buffer
    x_target = np.arange(x_min, x_max + resolution, resolution)
    y_target = np.arange(y_min, y_max + resolution, resolution)
    return x_target, y_target


@logwrap
def _reproject_dataset(ds, src_crs, dst_crs, x_target, y_target):
    """Reproject an xarray Dataset from *src_crs* to *dst_crs* onto a
    regular target grid using **rioxarray** (backed by GDAL/rasterio).

    Handles rotated-latlon, regular latlon, Lambert, etc. — any CRS pair
    that PROJ supports.  Dask-backed arrays are supported natively.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with dims (..., y, x).
    src_crs, dst_crs : pyproj.CRS
        Source and destination CRS.
    x_target, y_target : 1-D np.ndarray
        Target grid *centre* coordinates in dst_crs (meters).

    Returns
    -------
    ds_out : xr.Dataset
        Dataset on (y_target, x_target) grid with x/y in meters.
    """
    if src_crs == dst_crs:
        logger.info("Source and target CRS are identical; skipping reprojection")
        return ds

    ny_dst, nx_dst = len(y_target), len(x_target)
    res_x = float(x_target[1] - x_target[0])
    res_y = float(y_target[1] - y_target[0])

    # Affine: pixel-edge origin (half-pixel before first centre coordinate)
    dst_transform = Affine(
        res_x,
        0,
        float(x_target[0]) - res_x / 2,
        0,
        res_y,
        float(y_target[0]) - res_y / 2,
    )

    logger.info(
        "Reprojecting from %s to %s: (%d×%d) → (%d×%d) via rioxarray",
        src_crs.to_epsg() or src_crs.to_proj4()[:40],
        dst_crs.to_epsg() or dst_crs.to_proj4()[:40],
        ds.sizes.get("x", 0),
        ds.sizes.get("y", 0),
        nx_dst,
        ny_dst,
    )

    out_vars = {}
    for name, da in ds.data_vars.items():
        if "x" not in da.dims or "y" not in da.dims:
            out_vars[name] = da
            continue

        # rioxarray supports at most 3-D (band, y, x).
        # Reshape all non-spatial dims into a single "band" dim so we need
        # only ONE warp call per variable (avoids per-timestep overhead).
        non_spatial = [d for d in da.dims if d not in ("x", "y")]
        if non_spatial:
            # Remember original shape/coords for the non-spatial dims
            ns_sizes = [da.sizes[d] for d in non_spatial]
            ns_coords = {d: da[d] for d in non_spatial if d in da.coords}
            n_bands = int(np.prod(ns_sizes))

            # Flatten to (band, y, x) — materialises the array once
            flat = da.values.reshape(n_bands, da.sizes["y"], da.sizes["x"])
            da_flat = xr.DataArray(
                flat,
                dims=("band", "y", "x"),
                coords={"y": da["y"], "x": da["x"]},
            )
            da_flat = da_flat.rio.write_crs(src_crs).rio.set_spatial_dims(
                x_dim="x", y_dim="y"
            )
            da_reproj = da_flat.rio.reproject(
                dst_crs,
                shape=(ny_dst, nx_dst),
                transform=dst_transform,
                resampling=Resampling.bilinear,
                nodata=np.nan,
            )
            da_reproj = da_reproj.assign_coords(x=x_target, y=y_target)
            if "spatial_ref" in da_reproj.coords:
                da_reproj = da_reproj.drop_vars("spatial_ref")

            # Reshape back to original non-spatial dims + new (y, x)
            new_shape = tuple(ns_sizes) + (ny_dst, nx_dst)
            data_reshaped = da_reproj.values.reshape(new_shape)
            new_coords = {**ns_coords, "y": y_target, "x": x_target}
            da_reproj = xr.DataArray(
                data_reshaped,
                dims=non_spatial + ["y", "x"],
                coords=new_coords,
            )
        else:
            da_crs = da.rio.write_crs(src_crs).rio.set_spatial_dims(
                x_dim="x", y_dim="y"
            )
            da_reproj = da_crs.rio.reproject(
                dst_crs,
                shape=(ny_dst, nx_dst),
                transform=dst_transform,
                resampling=Resampling.bilinear,
                nodata=np.nan,
            )
            da_reproj = da_reproj.assign_coords(x=x_target, y=y_target)
            if "spatial_ref" in da_reproj.coords:
                da_reproj = da_reproj.drop_vars("spatial_ref")

        out_vars[name] = da_reproj

    coords = {
        c: ds.coords[c]
        for c in ds.coords
        if "x" not in ds.coords[c].dims and "y" not in ds.coords[c].dims
    }
    coords["x"] = x_target
    coords["y"] = y_target

    return xr.Dataset(out_vars, coords=coords)


# ---------------------------------------------------------------------------
# KNMI variable name mapping
# ---------------------------------------------------------------------------

# Default mapping from GRIB shortNames (as they appear in converted NetCDF)
# to the internal variable names expected by prep_harmonie's create_xarray_dataset.
#
# The prep_harmonie pipeline expects these names after the var-dict mapping:
#   3D model-level:  u, v, wz, t, q, clwc   (keyed ua, va, wa, ta, hus, clw)
#   Surface:         msl, 2t, 2sh            (keyed ps, tas, huss)
#
# After GRIB→NetCDF conversion the variable names depend on the tool used.
# cfgrib typically keeps shortNames; cdo may use paramId-based names.
# This default mapping assumes shortNames are preserved.

KNMI_ML_VAR_MAP = {
    "u": "u",  # u-wind on hybrid levels
    "v": "v",  # v-wind on hybrid levels
    "t": "t",  # temperature on hybrid levels
    "q": "q",  # specific humidity on hybrid levels
}

# Surface variable mapping is now handled explicitly in load_data().
# CDO naming conventions for KNMI output:
#   ML file:  'pres' (surface pressure at height=0), 't_2' (surface T), 'z' (sfc geopot)
#   SFC file: 'pres' (MSL at alt=0), 't_2' (T at 6 heights), 'r_2' (RH at 2m),
#             'td' (dew point at 2m)


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class KNMIPrepper(prep_harmonie.harmoniePrepper):
    """Adapter that loads KNMI operational HARMONIE output and feeds
    the standard ``harmoniePrepper`` pipeline.

    Subclasses ``harmoniePrepper`` and overrides ``load_data()`` to handle
    KNMI-specific file format and variable naming, while reusing the full
    ``prep_harmonie()`` processing chain.

    Additional ``input_json`` keys (beyond those of ``harmoniePrepper``):

    - ``KNMI_ml_glob``  — glob pattern for pre-converted N55ML NetCDF (3D fields)
    - ``KNMI_sfc_glob`` — glob pattern for pre-converted N55/N20 NetCDF (surface)
    - ``w_from_continuity`` — bool, default False.  Derive w from continuity.
    - ``knmi_ml_var_map`` — dict, optional override for 3D variable name mapping

    The input is expected to be an ``OpenBoundaryConfig`` instance.
    """

    @logwrap
    def load_data(self):
        """Load KNMI GRIB or NetCDF and remap to harmoniePrepper format.

        Handles CDO naming conventions:
        - Dimensions ``rlon``/``rlat`` for rotated grids
        - ``_2`` suffix for duplicate shortNames on different vertical coords
        - Multiple vertical coord types (hybrid, height, height_2, plev, …)
        """
        input_json = self.input_json
        grid = self.grid

        # ── Open model-level (N55ML) data ──────────────────────────────
        # Only chunk in time; keep lev, y, x as single chunks so spatial
        # operations (interpolation, reprojection) work on contiguous arrays.
        tchunk = input_json.tchunk
        is_ml_grib = _is_grib_source(input_json.KNMI_ml_glob, input_json.use_grib)
        if is_ml_grib:
            logger.info(
                "Opening KNMI model-level GRIB data from %s", input_json.KNMI_ml_glob
            )
            ds_ml = _open_knmi_grib_dataset(
                input_json.KNMI_ml_glob,
                chunks={"time": tchunk, "lev": -1, "y": -1, "x": -1},
            )
        else:
            logger.info(
                "Opening KNMI model-level NetCDF data from %s", input_json.KNMI_ml_glob
            )
            ds_ml = _open_knmi_dataset(
                input_json.KNMI_ml_glob,
                chunks={"time": tchunk, "lev": -1, "y": -1, "x": -1},
            )

        # ── Open surface data (N55 / N20) ──────────────────────────────
        is_sfc_grib = _is_grib_source(input_json.KNMI_sfc_glob, input_json.use_grib)
        if is_sfc_grib:
            logger.info(
                "Opening KNMI surface GRIB data from %s", input_json.KNMI_sfc_glob
            )
            ds_sfc = _open_knmi_grib_dataset(
                input_json.KNMI_sfc_glob,
                chunks={"time": tchunk, "y": -1, "x": -1},
            )
        else:
            logger.info(
                "Opening KNMI surface NetCDF data from %s", input_json.KNMI_sfc_glob
            )
            ds_sfc = _open_knmi_dataset(
                input_json.KNMI_sfc_glob,
                chunks={"time": tchunk, "y": -1, "x": -1},
            )

        # ── Extract time coordinate ───────────────────────────────────
        time = ds_ml["time"]

        # Interpolate surface time to model-level time if they differ
        if not ds_sfc.time.equals(ds_ml.time):
            ds_sfc = ds_sfc.interp(
                time=time,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )

        # ── Separate 3D and surface variables in the ML file ──────────
        # CDO puts 3D fields on dim 'lev' (hybrid); surface fields get
        # a singleton 'height' (or similar) dimension.
        ml_map = input_json.knmi_ml_var_map or KNMI_ML_VAR_MAP
        ds_ml = ds_ml.rename({k: v for k, v in ml_map.items() if k in ds_ml})

        ml_3d_vars = [v for v in ds_ml.data_vars if "lev" in ds_ml[v].dims]
        ds_3d = ds_ml[ml_3d_vars]

        # ── Extract surface pressure from the ML file ─────────────────
        # CDO names it 'pres' at height=0 in the N55ML output.
        sfc_ps = None
        if "pres" in ds_ml and "lev" not in ds_ml["pres"].dims:
            sfc_ps = _squeeze_vertical(ds_ml["pres"])
            logger.info("Extracted surface pressure from ML file ('pres')")

        # Fallback: surface pressure from SFC file
        # In the N55 SFC output, 'pres' at alt=0 is MSL pressure.
        # 'pres_2' at height=0 would be the actual surface pressure.
        if sfc_ps is None:
            for pname in ("pres_2", "pres"):
                if pname in ds_sfc and "lev" not in ds_sfc[pname].dims:
                    sfc_ps = _squeeze_vertical(ds_sfc[pname])
                    logger.info(
                        "Extracted surface pressure from SFC file ('%s')", pname
                    )
                    break
        if sfc_ps is None:
            raise KeyError(
                "Cannot find surface pressure in ML or SFC file. "
                "Expected 'pres' (ML) or 'pres'/'pres_2' (SFC)."
            )

        # ── Validate hybrid levels ────────────────────────────────────
        n_lev = ds_3d.sizes.get("lev", 0)
        expected_lev = len(hybrid_levels.ahalf) - 1  # 91 half-levels → 90 full
        if n_lev != expected_lev:
            logger.warning(
                "N55ML has %d hybrid levels, expected %d. "
                "Hybrid coefficient arrays in hybrid_levels.py may not match.",
                n_lev,
                expected_lev,
            )

        # ── Fill missing 3D fields ────────────────────────────────────
        # Vertical velocity — placeholder zeros here; if w_from_continuity
        # is requested it is computed AFTER reprojection when x/y are in
        # meters so that differentiate("x"/"y") gives correct units.
        ds_3d = ds_3d.assign({"wz": xr.zeros_like(ds_3d["u"])})

        # Cloud liquid water content (not in KNMI output)
        ds_3d = ds_3d.assign({"clwc": xr.zeros_like(ds_3d["q"])})

        # ── Extract 2m temperature ────────────────────────────────────
        # SFC file: CDO 't_2' on multi-level 'height_2' dim (0-300 m).
        # ML  file: CDO 't_2' at height=0 (skin temperature).
        t2m = None
        if "t_2" in ds_sfc:
            t2m = _select_height_level(ds_sfc["t_2"], target_height=2.0)
            logger.info("Extracted 2m temperature from SFC file ('t_2')")
        elif "t_2" in ds_ml and "lev" not in ds_ml["t_2"].dims:
            t2m = _squeeze_vertical(ds_ml["t_2"])
            logger.info("Using ML surface temperature ('t_2') as 2m T fallback")
        else:
            raise KeyError(
                "Cannot find 2m temperature. Expected 't_2' in SFC or ML file."
            )

        # ── Derive 2m specific humidity ───────────────────────────────
        # Preferred: dew point temperature → q  (most direct).
        # Fallback:  RH × saturation VP → q.
        if "td" in ds_sfc:
            td = _select_height_level(ds_sfc["td"], target_height=2.0)
            huss = q_from_td(td, sfc_ps)
            logger.info("Derived 2m q from dew point temperature ('td')")
        elif "r_2" in ds_sfc:
            rh = _select_height_level(ds_sfc["r_2"], target_height=2.0)
            huss = q_from_rh(rh / 100.0, t2m, sfc_ps)
            logger.info("Derived 2m q from relative humidity ('r_2')")
        elif "r" in ds_sfc:
            rh = _select_height_level(ds_sfc["r"], target_height=2.0)
            huss = q_from_rh(rh / 100.0, t2m, sfc_ps)
            logger.info("Derived 2m q from relative humidity ('r')")
        else:
            raise KeyError(
                "Cannot derive 2m specific humidity: need 'td', 'r_2', "
                "or 'r' in SFC file."
            )

        # ── Build clean datasets with only the needed variables ────────
        ds_3d_clean = ds_3d[["u", "v", "wz", "t", "q", "clwc"]]
        sfc_ds = xr.Dataset({"ps": sfc_ps, "2t": t2m, "2sh": huss})

        # ── Reproject to target CRS with x/y in meters ────────────────
        src_crs = _detect_source_crs(ds_ml)
        if grid.crs is not None:
            dst_crs = CRS(grid.crs)
        else:
            # Default: EPSG:28992 (Amersfoort / RD New) — the Dutch national
            # coordinate system.  Suitable when the DALES domain is in NL.
            dst_crs = CRS("EPSG:28992")
            logger.info("Grid has no CRS set; defaulting to EPSG:28992 (RD New)")

        # Determine target grid resolution from source grid spacing.
        # Transform two adjacent grid points near the domain centre to the
        # target CRS and measure the actual metric distance.  This is correct
        # for rotated grids where lat/lon degree spacing ≠ metric spacing.
        fwd = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        mid_ix = len(ds_ml["x"]) // 2
        mid_iy = len(ds_ml["y"]) // 2
        x0_src, y0_src = float(ds_ml["x"][mid_ix]), float(ds_ml["y"][mid_iy])
        x1_src = float(ds_ml["x"][mid_ix + 1])
        x0_dst, y0_dst = fwd.transform(x0_src, y0_src)
        x1_dst, y1_dst = fwd.transform(x1_src, y0_src)
        resolution = np.hypot(x1_dst - x0_dst, y1_dst - y0_dst)
        logger.info(
            "Source grid metric spacing ≈ %.0f m (from adjacent points at grid centre)",
            resolution,
        )

        # boundary.py selects data within ±16·dx of each domain edge and
        # then uses xr.interp() which needs ≥2 points.  Cap the target
        # resolution so there are enough grid points in those windows.
        # Factor 8 gives ≈4 points per boundary window.
        grid_dx = min(grid.xsize / grid.itot, grid.ysize / grid.jtot)
        max_res = 8 * grid_dx
        if resolution > max_res:
            logger.info(
                "Capping target resolution from %.0f m to %.0f m "
                "(need ≥2 points per boundary window of ±%d·dx)",
                resolution,
                max_res,
                16,
            )
            resolution = max_res

        if input_json.target_resolution is not None:
            resolution = input_json.target_resolution

        x_sw, y_sw = grid.x0, grid.y0

        if input_json.filter is not None:
            buffer = 4 * input_json.filter["sigma"]
        else:
            buffer = resolution * 4

        x_target, y_target = _build_target_grid(grid, buffer, resolution)

        # Time selection (do before reprojection — cheaper on source grid)
        time_sel = time.sortby("time").sel(time=slice(input_json.start, input_json.end))
        ds_3d_clean = ds_3d_clean.sel(time=time_sel)
        sfc_ds = sfc_ds.sel(time=time_sel)

        # ── Crop source grid to the region of interest ─────────────────
        # Transform target bounding box corners back to source CRS,
        # then select only the source pixels that cover that area (+margin).
        inv = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        corners_x_dst = [
            float(x_target[0]),
            float(x_target[-1]),
            float(x_target[0]),
            float(x_target[-1]),
        ]
        corners_y_dst = [
            float(y_target[0]),
            float(y_target[-1]),
            float(y_target[-1]),
            float(y_target[0]),
        ]
        corners_x_src, corners_y_src = inv.transform(corners_x_dst, corners_y_dst)

        # Add generous margin in source CRS units (5 source grid cells)
        src_dx = float(np.abs(ds_3d_clean["x"].diff("x").median()))
        src_dy = float(np.abs(ds_3d_clean["y"].diff("y").median()))
        margin_x = 5 * src_dx
        margin_y = 5 * src_dy

        x_lo = min(corners_x_src) - margin_x
        x_hi = max(corners_x_src) + margin_x
        y_lo = min(corners_y_src) - margin_y
        y_hi = max(corners_y_src) + margin_y

        ds_3d_clean = ds_3d_clean.sel(x=slice(x_lo, x_hi), y=slice(y_lo, y_hi))
        sfc_ds = sfc_ds.sel(x=slice(x_lo, x_hi), y=slice(y_lo, y_hi))
        logger.info(
            "Cropped source grid to %d×%d (from 676×564) before reprojection",
            ds_3d_clean.sizes["x"],
            ds_3d_clean.sizes["y"],
        )

        # Eagerly load the small cropped subset into memory.
        # This avoids 9 separate dask graph evaluations (one per .values
        # call inside _reproject_dataset) each reading from compressed NetCDF.
        ds_3d_clean = ds_3d_clean.load()
        sfc_ds = sfc_ds.load()

        # Reproject 3D model-level fields
        logger.info("Reprojecting model-level fields to Lambert grid")
        ds_3d_clean = _reproject_dataset(
            ds_3d_clean, src_crs, dst_crs, x_target, y_target
        )

        # Reproject surface fields (same source CRS & grid)
        logger.info("Reprojecting surface fields to Lambert grid")
        sfc_ds = _reproject_dataset(sfc_ds, src_crs, dst_crs, x_target, y_target)

        # ── Compute w from continuity (now that x/y are in metres) ──
        if input_json.w_from_continuity:
            logger.info("Deriving vertical velocity from continuity equation")
            wz = w_from_continuity(
                ds_3d_clean["u"],
                ds_3d_clean["v"],
                sfc_ds["ps"],
                hybrid_levels.ahalf,
                hybrid_levels.bhalfs,
            )
            ds_3d_clean = ds_3d_clean.assign({"wz": wz})

        # ── Merge into a single dataset ───────────────────────────────
        data = xr.merge([ds_3d_clean, sfc_ds], compat="override", join="outer")

        ds_ml.close()
        ds_sfc.close()

        # ── Build transform for the target Lambert CRS ────────────────
        transform = Transform({"proj4": dst_crs.to_proj4()})

        # ── Store results (same interface as harmoniePrepper.load_data) ──
        self.data = data
        self.transform = transform
        # Ensure lev is a single chunk — prep_harmonie's apply_ufunc requires it.
        if "lev" in self.data.dims:
            self.data = self.data.chunk({"lev": -1})
        (self.data,) = dask.optimize(self.data)
        self.transform = prep_harmonie.update_transform(self.transform, x_sw, y_sw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_grib_source(glob_pattern, use_grib_flag=None) -> bool:
    """Determine whether the source path or glob points to GRIB files."""
    if use_grib_flag is not None:
        return bool(use_grib_flag)
    if not glob_pattern:
        return False
    if isinstance(glob_pattern, (list, tuple)) and glob_pattern:
        pattern_str = str(glob_pattern[0])
    else:
        pattern_str = str(glob_pattern)

    if pattern_str.endswith((".nc", ".nc4", ".netcdf", ".nc*", ".nc4*")):
        return False

    if "_GB" in pattern_str or pattern_str.endswith(
        (".grib", ".grb", ".grib2", ".grb2")
    ):
        return True

    matched = glob.glob(pattern_str)
    if matched:
        files = [f for f in matched if os.path.isfile(f)]
        if files:
            if files[0].endswith((".nc", ".nc4", ".netcdf")):
                return False
            return True
    return False


def _get_grib_metadata(field, key, default=None):
    """Safely extract metadata item from an earthkit GRIB Field."""
    for k in (f"metadata.{key}", key):
        try:
            val = field.get(k)
            if val is not None:
                return val
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    try:
        val = field.metadata(key)
        if val is not None:
            return val
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return default


@logwrap
def _open_knmi_grib_dataset(glob_pattern, chunks=None):
    """Open KNMI GRIB dataset using earthkit-data, remapping duplicate parameters.

    Disambiguates parameter names across vertical level types and timeRangeIndicators
    using the CDO-like `_2`, `_3` suffix convention, then converts to xarray.
    """
    try:
        import earthkit.data as ekd
    except ImportError as exc:
        raise ImportError(
            "earthkit-data is required to open GRIB files. Please install earthkit-data."
        ) from exc

    if isinstance(glob_pattern, (str, Path)):
        matched = sorted(glob.glob(str(glob_pattern)))
        if matched:
            files = [f for f in matched if os.path.isfile(f)]
            source_input = files if files else str(glob_pattern)
        else:
            source_input = str(glob_pattern)
    else:
        source_input = glob_pattern

    ds_src = ekd.from_source("file", source_input)
    fl = ds_src.to_fieldlist()
    if len(fl) == 0:
        raise FileNotFoundError(f"No GRIB fields found matching {glob_pattern}")

    new_fields = []
    assignments = {}
    used_names = set()

    for field in fl:
        parameter = field.parameter.variable()
        level_type = field.vertical.level_type()

        tri = _get_grib_metadata(field, "timeRangeIndicator")
        if tri is None:
            tri = 0

        key = (parameter, level_type, tri)

        if key in assignments:
            new_parameter = assignments[key]
        else:
            # First occurrence of this parameter/level/TRI combination
            if parameter not in used_names:
                new_parameter = parameter
            else:
                n = 2
                new_parameter = f"{parameter}_{n}"

                while new_parameter in used_names:
                    n += 1
                    new_parameter = f"{parameter}_{n}"

            assignments[key] = new_parameter
            used_names.add(new_parameter)

        if new_parameter == parameter:
            new_fields.append(field)
        else:
            new_fields.append(field.set({"parameter.variable": new_parameter}))

    new_fl = ekd.FieldList.from_fields(new_fields)
    dataset = new_fl.to_xarray(allow_holes=True)

    # 1. Resolve time coordinate
    first_field = fl[0]
    base_time = None
    try:
        dt_info = first_field.datetime()
        if isinstance(dt_info, dict):
            base_time = dt_info.get("base_time") or dt_info.get(
                "forecast_reference_time"
            )
        elif isinstance(dt_info, dt.datetime):
            base_time = dt_info
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

    if base_time is None:
        val = _get_grib_metadata(first_field, "forecast_reference_time")
        if isinstance(val, dt.datetime):
            base_time = val
        elif isinstance(val, str):
            try:
                base_time = dt.datetime.fromisoformat(val)
            except ValueError:
                pass

    if base_time is None:
        data_date = _get_grib_metadata(first_field, "dataDate")
        data_time = _get_grib_metadata(first_field, "dataTime")
        if data_date is not None and data_time is not None:
            try:
                data_date_str = str(data_date)
                data_time_int = int(data_time)
                hour = data_time_int // 100
                minute = data_time_int % 100
                base_time = dt.datetime.strptime(data_date_str, "%Y%m%d").replace(
                    hour=hour, minute=minute
                )
            except (ValueError, TypeError):
                pass

    if "step" in dataset.dims and base_time is not None:
        time_values = np.datetime64(base_time) + dataset["step"].values
        dataset = dataset.assign_coords(time=("step", time_values)).swap_dims(
            {"step": "time"}
        )
        dataset = dataset.drop_vars("step", errors="ignore")
    elif "step" in dataset.dims and "time" not in dataset.dims:
        dataset = dataset.rename({"step": "time"})

    # 2. Normalise hybrid level dimension: for 91-level datasets (0..90), select 1..90 and rename to lev
    if "level" in dataset.dims:
        if 0 in dataset["level"].values and len(dataset["level"]) == 91:
            dataset = dataset.sel(level=slice(1, 90))
            dataset = dataset.rename({"level": "lev"})

    # Drop level_type if unused by data variables
    if "level_type" in dataset.dims and not any(
        "level_type" in da.dims for da in dataset.data_vars.values()
    ):
        dataset = dataset.drop_dims("level_type", errors="ignore")

    # 3. Detect and set rotated CRS and 1D x/y coordinates
    src_crs = None
    try:
        grid_type = _get_grib_metadata(first_field, "gridType")
        if grid_type == "rotated_ll":
            sp_lat = _get_grib_metadata(first_field, "latitudeOfSouthernPoleInDegrees")
            if sp_lat is None:
                sp_lat = _get_grib_metadata(first_field, "latitudeOfSouthernPole")
            sp_lon = _get_grib_metadata(first_field, "longitudeOfSouthernPoleInDegrees")
            if sp_lon is None:
                sp_lon = _get_grib_metadata(first_field, "longitudeOfSouthernPole")
            angle = _get_grib_metadata(first_field, "angleOfRotationInDegrees", 0.0)
            if angle is None:
                angle = _get_grib_metadata(first_field, "angleOfRotation", 0.0)

            if sp_lat is not None and sp_lon is not None:
                np_lat = -float(sp_lat)
                np_lon = (float(sp_lon) + 180.0) % 360.0 - 180.0
                src_crs = CRS.from_cf(
                    {
                        "grid_mapping_name": "rotated_latitude_longitude",
                        "grid_north_pole_latitude": np_lat,
                        "grid_north_pole_longitude": np_lon,
                        "north_pole_grid_longitude": (
                            float(angle) if angle is not None else 0.0
                        ),
                    }
                )
            else:
                # Default KNMI HARMONIE rotated pole
                src_crs = CRS.from_cf(
                    {
                        "grid_mapping_name": "rotated_latitude_longitude",
                        "grid_north_pole_latitude": 38.0,
                        "grid_north_pole_longitude": 180.0,
                        "north_pole_grid_longitude": 0.0,
                    }
                )
    except Exception as exc:
        logger.debug("Failed to extract rotated grid metadata from GRIB: %s", exc)

    if src_crs is None:
        src_crs = CRS("EPSG:4326")

    if "latitude" in dataset.coords and "longitude" in dataset.coords:
        try:
            fwd = Transformer.from_crs(CRS("EPSG:4326"), src_crs, always_xy=True)
            rot_x_2d, rot_y_2d = fwd.transform(
                dataset["longitude"].values, dataset["latitude"].values
            )
            rot_x = np.mean(rot_x_2d, axis=0)
            rot_y = np.mean(rot_y_2d, axis=1)
            dataset = dataset.assign_coords(x=("x", rot_x), y=("y", rot_y))
        except Exception as exc:
            logger.warning("Could not transform lat/lon to 1D x/y: %s", exc)

    # Ensure coordinates are sorted
    if "x" in dataset.dims and len(dataset["x"]) > 1:
        dataset = dataset.sortby("x")
    if "y" in dataset.dims and len(dataset["y"]) > 1:
        dataset = dataset.sortby("y")
    if "time" in dataset.dims and len(dataset["time"]) > 1:
        dataset = dataset.sortby("time")
    if "lev" in dataset.dims and len(dataset["lev"]) > 1:
        dataset = dataset.sortby("lev")

    dataset = dataset.rio.write_crs(src_crs)
    cf_attrs = src_crs.to_cf()
    gm_var = cf_attrs.get("grid_mapping_name", "rotated_latitude_longitude")
    dataset[gm_var] = xr.DataArray(0, attrs=cf_attrs)
    for var in dataset.data_vars:
        dataset[var].attrs["grid_mapping"] = gm_var

    try:
        dataset = fix_lambert_offsets(dataset)
    except (AttributeError, KeyError, ValueError):
        logger.debug("fix_lambert_offsets not applicable, skipping")

    if chunks is not None:
        chunk_dict = {k: v for k, v in chunks.items() if k in dataset.dims}
        if chunk_dict:
            dataset = dataset.chunk(chunk_dict)

    return dataset


@logwrap
def _open_knmi_dataset(glob_pattern, chunks):
    """Open a pre-converted KNMI NetCDF dataset, normalising dimensions.

    Handles dimension naming differences between conversion tools:
    ``level``/``hybrid`` → ``lev``; ``rlon``/``longitude``/``lon`` → ``x``;
    ``rlat``/``latitude``/``lat`` → ``y``.
    """
    ds = xr.open_mfdataset(
        glob_pattern,
        decode_coords="all",
        engine="netcdf4",
        chunks=chunks,
    )

    # Try to apply Lambert/rotated offset fix (may be a no-op)
    try:
        ds = fix_lambert_offsets(ds)
    except (AttributeError, KeyError, ValueError):
        logger.debug("fix_lambert_offsets not applicable, skipping")

    # Normalise dimension names
    rename_map = {}
    for dim in ds.dims:
        dl = dim.lower()
        if dl in ("level", "hybrid"):
            rename_map[dim] = "lev"
        elif dl in ("rlat", "latitude", "lat") and "y" not in ds.dims:
            rename_map[dim] = "y"
        elif dl in ("rlon", "longitude", "lon") and "x" not in ds.dims:
            rename_map[dim] = "x"
    if rename_map:
        logger.debug("Renaming dimensions: %s", rename_map)
        ds = ds.rename(rename_map)

    return ds


def _squeeze_vertical(da):
    """Remove any singleton vertical dimensions from a DataArray.

    Keeps ``time``, ``x``, ``y`` dimensions.  All other singleton dims
    (``height``, ``alt``, ``height_2``, …) are squeezed out.
    """
    keep = {"time", "x", "y"}
    for dim in list(da.dims):
        if dim not in keep and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
    return da


def _select_height_level(da, target_height=2.0):
    """Select the level closest to *target_height* from a multi-level variable.

    Handles CDO's various height dimension names (``height``, ``height_2``,
    ``height_3``, …).  If the extra vertical dim is a singleton, it is simply
    squeezed out.
    """
    keep = {"time", "x", "y", "lev"}
    height_dims = [d for d in da.dims if d not in keep]
    if not height_dims:
        return da

    for hdim in height_dims:
        if da.sizes[hdim] == 1:
            da = da.squeeze(hdim, drop=True)
        else:
            levels = np.asarray(da[hdim].values, dtype=float)
            idx = int(np.argmin(np.abs(levels - target_height)))
            chosen = float(levels[idx])
            logger.info(
                "Selected level %.1f m from dim '%s' (target: %.1f m)",
                chosen,
                hdim,
                target_height,
            )
            da = da.isel({hdim: idx}, drop=True)
    return da


def _get_time(ds):
    """Extract time coordinate from dataset."""
    return ds["time"]
