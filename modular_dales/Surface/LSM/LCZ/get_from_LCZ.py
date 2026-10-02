import logging
import pathlib
import tempfile
from dataclasses import dataclass

import numpy as np
import rasterio
import xarray as xr

from modular_dales.IO_helpers import ensure_sorted, raster_to_xarray
from modular_dales.Surface.LSM.LCZ.downloaders import get_cog, get_esa

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)

LCZ_fields = [
    "svf",
    "aspect_ratio",
    "building_surface_fraction",
    "impervious_surface_fraction",
    "pervious_surface_fraction",
    "height_roughness_element",
    "terrain_roughness_class",
    "roughness_length",
    "albedo",
]
LCZ_field_to_slurb = {
    "aspect_ratio": "hw_can",
    "building_surface_fraction": "f_bld",
    "height_roughness_element": "h_bld",
    "roughness_length": "z0_urb",
}
# 10:"tree_cover", 2,3,4,5,17,18
# 20:"shrubland",15,16
# 30:"grassland", 1,6
# 40:"cropland",0,9
# 50:""built-up",20,21
# 60:"bare/sparse-vegetation",7,10,21
# 70:"snow_and_ice",8,11
# 80:"permanent_water_bodies",13,14,22
# 90:"herbaceous wetland",12
# 95:"mangroves", 12
# 100:"moss_lichen", 8
# IFS class -> representative ESA WorldCover code
IFS_FROM_ESA = {
    10: 2,  # tree_cover
    20: 15,  # shrubland
    30: 1,  # grassland
    40: 0,  # cropland
    50: 20,  # built-up
    60: 7,  # bare / sparse
    70: 8,  # snow and ice
    80: 22,  # permanent water
    90: 12,  # herbaceous wetland
    95: 12,  # mangroves
    100: 8,  # moss / lichen
}

# soil type from LCZ
SOIL_FROM_LCZ = {
    10: 4,  # tree_cover
    20: 4,  # shrubland
    30: 3,  # grassland
    40: 3,  # cropland
    60: 1,  # bare / sparse vegetation
    70: 1,  # snow and ice
    90: 6,  # herbaceous wetland
    95: 6,  # mangroves
    100: 6,  # moss / lichen
}


ESA_TO_IFS = {v: k for k, v in IFS_FROM_ESA.items()}

roughness_table = {1: 0.0002, 2: 0.005, 3: 0.03, 4: 0.10, 5: 0.25, 6: 0.5, 7: 1, 8: 2}


@dataclass
class LCZ:
    svf: float
    aspect_ratio: float
    building_surface_fraction: float
    impervious_surface_fraction: float
    pervious_surface_fraction: float
    height_roughness_element: float
    terrain_roughness_class: int
    roughness_length: float
    albedo: float

    def __init__(
        self,
        svf,
        aspect_ratio,
        building_surface_fraction,
        impervious_surface_fraction,
        pervious_surface_fraction,
        height_roughness_element,
        terrain_roughness_class,
        albedo,
    ):
        self.svf = svf
        self.aspect_ratio = aspect_ratio
        self.building_surface_fraction = building_surface_fraction / 100
        self.impervious_surface_fraction = impervious_surface_fraction / 100
        self.pervious_surface_fraction = pervious_surface_fraction / 100
        if height_roughness_element < 1:
            logger.warning(
                "Roughness element should be at least 1m for MOST stability!!! Setting to 1m"
            )
            height_roughness_element = 1
        self.height_roughness_element = height_roughness_element
        self.terrain_roughness_class = terrain_roughness_class
        self.roughness_length = roughness_table[terrain_roughness_class]
        self.albedo = albedo


lcz_dict = {
    1: LCZ(0.3, 2, 45, 45, 10, 25, 8, 0.15),
    2: LCZ(0.45, 1.375, 50, 30, 20, 17.5, 6, 0.15),
    3: LCZ(0.4, 1.125, 50, 20, 30, 6, 6, 0.15),
    4: LCZ(0.6, 1, 30, 35, 35, 25, 7, 0.185),
    5: LCZ(0.65, 0.525, 30, 40, 30, 17.5, 5, 0.185),
    6: LCZ(0.75, 0.525, 25, 25, 50, 6.5, 5, 0.185),
    7: LCZ(0.35, 1.5, 75, 10, 15, 3, 4, 0.25),
    8: LCZ(0.7, 0.2, 40, 50, 10, 6.5, 5, 0.2),
    9: LCZ(0.8, 0.175, 10, 15, 75, 6.5, 5, 0.185),
    10: LCZ(0.75, 0.35, 25, 30, 45, 10, 5, 0.16),
    11: LCZ(0.4, 1, 5, 5, 90, 16.5, 8, 0.15),  # A
    12: LCZ(0.65, 0.5, 5, 5, 90, 9, 5, 0.2),  # B
    13: LCZ(0.8, 0.625, 5, 5, 90, 2, 4, 0.225),  # C
    14: LCZ(0.9, 0.1, 5, 5, 90, 1, 3, 0.2),  # D
    15: LCZ(0.9, 0.1, 5, 90, 5, 0.25, 1, 0.225),  # E
    16: LCZ(0.9, 0.1, 5, 5, 90, 0.1, 1, 0.275),  # F
    17: LCZ(0.9, 0.1, 5, 5, 90, 0.1, 1, 0.06),  # G
}


# LCZ classes A-G (11-17) mapped to natural IFS classes when requested.
# A=dense trees, B=scattered trees, C=bush/scrub, D=low plants,
# E=bare rock or paved, F=bare soil or sand, G=water.
LCZ_URBAN_NATURAL_TO_IFS = {
    11: 17,  # A -> mixed forest/wood
    12: 18,  # B -> interrupted forest
    13: 16,  # C -> deciduous shrubs
    14: 1,  # D -> short grass
    15: 10,  # E -> semidesert (bare/paved proxy)
    16: 7,  # F -> desert (bare soil/sand proxy)
    17: 22,  # G -> water
}


def apply_urban_natural_lcz_overrides(
    lcz_da,
    ifs_da,
    urban_natural_lcz_to_10=False,
    urban_natural_lcz_to_natural_lsm=False,
):
    """Apply optional overrides for urban pixels with natural LCZ classes (11-17)."""
    lcz_values = np.asarray(lcz_da.values, dtype=float).copy()
    ifs_values = np.asarray(ifs_da.values, dtype=float).copy()

    is_urban = ifs_values == 20
    urban_natural_lcz = (
        is_urban & np.isfinite(lcz_values) & (lcz_values >= 11) & (lcz_values <= 17)
    )

    if urban_natural_lcz_to_10:
        lcz_values[urban_natural_lcz] = 10

    if urban_natural_lcz_to_natural_lsm:
        for lcz_class, ifs_class in LCZ_URBAN_NATURAL_TO_IFS.items():
            mask = urban_natural_lcz & (lcz_values == lcz_class)
            ifs_values[mask] = ifs_class

    lcz_out = xr.DataArray(
        lcz_values,
        dims=lcz_da.dims,
        coords=lcz_da.coords,
        name=lcz_da.name,
        attrs=lcz_da.attrs,
    )
    ifs_out = xr.DataArray(
        ifs_values,
        dims=ifs_da.dims,
        coords=ifs_da.coords,
        name=ifs_da.name,
        attrs=ifs_da.attrs,
    )
    return lcz_out, ifs_out


def map_lcz_raster(
    lcz, lcz_profile, dst_tif, lcz_dict, field, nodata_in=0, nodata_out=np.nan
):
    # build lookup table (index 0 unused)
    lut = np.full(18, nodata_out, dtype=float)
    for k, v in lcz_dict.items():
        lut[k] = getattr(v, field)

    # mapped = np.zeros_like(lcz)
    mapped = np.where(lcz == nodata_in, nodata_out, lut[lcz])

    lcz_profile.update(
        dtype="float32",
        nodata=nodata_out,
    )

    with rasterio.open(dst_tif, "w", **lcz_profile) as dst:
        # dst.write(mapped.astype("float32"))
        dst.write(mapped[0, :, :].astype("float32"), 1)  # band=1


def map_esa_to_ifs(esa_da):
    """
    Map ESA WorldCover raster to IFS land-cover codes.
    """
    ifs = np.full_like(esa_da.values, np.nan, dtype=float)

    for esa_code, ifs_code in IFS_FROM_ESA.items():
        ifs[esa_da.values == esa_code] = ifs_code

    return xr.DataArray(
        ifs,
        dims=esa_da.dims,
        coords=esa_da.coords,
        name="ifs_land_cover",
    )


def map_lcz_to_fields(lcz_da, ifs_da, lcz_dict):
    """
    Create LCZ-derived fields as xarray DataArrays.
    Values are mapped from LCZ wherever LCZ is valid.
    """
    out = {}

    for field in LCZ_fields:
        lut = np.full(18, np.nan, dtype=float)
        for k, v in lcz_dict.items():
            lut[k] = getattr(v, field)

        mapped = np.full_like(lcz_da.values, np.nan, dtype=float)
        valid = lcz_da.values > 0
        for num in np.arange(start=1, stop=18):
            mapped[valid & (lcz_da.values == num)] = lut[num]

        out[field] = xr.DataArray(
            mapped,
            dims=lcz_da.dims,
            coords=lcz_da.coords,
            name=field,
        )

    return out


def map_soil_type(esa_da):
    """
    Create soil type map:
    - built-up -> derived from LCZ
    - others -> from IFS land-cover
    - water -> NaN
    """
    soil = np.full_like(esa_da.values, np.nan, dtype=float)

    for lcz_type, soil_type in SOIL_FROM_LCZ.items():
        soil[esa_da.values == lcz_type] = soil_type
    if np.any(np.isnan(soil)):
        logger.warning(
            "Soil is set to NaN for built-up type as there is no sensible data. Setting to index 2 anyway."
        )
        soil[np.isnan(soil)] = 2

    return xr.DataArray(
        soil,
        dims=esa_da.dims,
        coords=esa_da.coords,
        name="soil_index",
    )


def build_land_surface_dataset(
    esa_tif,
    lcz_tif,
    lcz_dict,
    urban_natural_lcz_to_10=False,
    urban_natural_lcz_to_natural_lsm=False,
):
    # --- ESA WorldCover ---
    esa = raster_to_xarray(esa_tif, "esa_worldcover")

    # --- IFS land cover ---
    ifs = map_esa_to_ifs(esa)

    # --- LCZ (only meaningful for built-up) ---
    lcz = raster_to_xarray(lcz_tif, "lcz")

    lcz, ifs = apply_urban_natural_lcz_overrides(
        lcz,
        ifs,
        urban_natural_lcz_to_10=urban_natural_lcz_to_10,
        urban_natural_lcz_to_natural_lsm=urban_natural_lcz_to_natural_lsm,
    )

    lcz_fields = map_lcz_to_fields(lcz, ifs, lcz_dict)

    soil = map_soil_type(esa)

    # --- Build dataset ---
    ds = xr.Dataset(
        {
            "ifs_land_cover": ifs,
            "lcz": lcz,
            "index_soil": soil,
            **lcz_fields,
        }
    )

    return ds


def do_everything(
    grid,
    urban_natural_lcz_to_10=False,
    urban_natural_lcz_to_natural_lsm=False,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = pathlib.Path(tmpdirname)
        lcz_file = path / "lcz.tif"
        esa_file = path / "esa.tif"
        get_cog(grid, lcz_file)
        get_esa(grid, esa_file)

        ds = build_land_surface_dataset(
            esa_file,
            lcz_file,
            lcz_dict=lcz_dict,
            urban_natural_lcz_to_10=urban_natural_lcz_to_10,
            urban_natural_lcz_to_natural_lsm=urban_natural_lcz_to_natural_lsm,
        )

        ds = ensure_sorted(ds)

    return ds
