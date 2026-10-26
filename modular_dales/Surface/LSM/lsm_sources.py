"""Map LCZ and fractional NetCDF sources onto typed land-use state."""

import pathlib
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.ls2d_atmosphere import FromLS2D
from modular_dales.module_registry import register_module
from modular_dales.Surface.LSM.lsm_data import (
    FloatArray,
    LandUseName,
    LandUseState,
    SoilFields,
)
from modular_dales.Surface.LSM.lu_modifications import (
    LandUseModification,
    LandUseModifications,
)
from modular_dales.Surface.LSM.modular_temps_moisture import LSMProfileSpec


def distribute_fraction(
    land_use: dict[LandUseName, LandUseState], fraction: FloatArray, ifs_id: int
) -> None:
    """Share an IFS fraction equally between tiles with the same mapping."""
    matching = [
        state for state in land_use.values() if state.definition.ifs_id == ifs_id
    ]
    for state in matching:
        state.lu_frac[:] = fraction / len(matching)


def apply_lcz(
    land_use: dict[LandUseName, LandUseState], soil: SoilFields, dataset: xr.Dataset
) -> None:
    """Populate soil classes and land-use fractions from an LCZ dataset."""
    soil.index_soil[:] = dataset["index_soil"].values
    cover = dataset["ifs_land_cover"].values
    ifs_ids = {state.definition.ifs_id for state in land_use.values()}
    for ifs_id in ifs_ids:
        if ifs_id is not None:
            distribute_fraction(
                land_use, np.asarray(cover == ifs_id, dtype=np.float64), ifs_id
            )


def apply_from_netcdf(
    land_use: dict[LandUseName, LandUseState], soil: SoilFields, ds: xr.Dataset
) -> None:
    """Populate soil classes and clipped fractional land cover from NetCDF."""
    soil.index_soil[:] = ds["index_soil"].values
    water = np.asarray(ds["frac_water"].values, dtype=np.float64).clip(0, 1)
    sea = np.asarray(ds["frac_sea"].values, dtype=np.float64).clip(0, 1)
    town = np.asarray(ds["frac_town"].values, dtype=np.float64).clip(0, 1)
    nature = np.asarray(ds["frac_nature"].values, dtype=np.float64).clip(0, 1)
    ifs_cover = np.asarray(ds["ifs_land_cover"].values, dtype=np.float64)
    distribute_fraction(land_use, (water + sea).clip(0, 1), ifs_id=22)
    distribute_fraction(land_use, town, ifs_id=20)
    natural_states = [
        state
        for state in land_use.values()
        if state.definition.ifs_id not in (None, 20, 22)
    ]
    if natural_states:
        weights = np.stack(
            [ifs_cover == state.definition.ifs_id for state in natural_states]
        ).astype(np.float64)
        total = weights.sum(axis=0)
        weights[:, total == 0] = 1.0 / len(natural_states)
        total[total == 0] = 1.0
        for state, weight in zip(natural_states, weights, strict=True):
            state.lu_frac[:] = nature * weight / total


@register_module
@dataclass
class FromLCZ:
    """LCZ classification source; supports the IFS and DEPAC land-use sets."""

    SUPPORTED_LAND_USE_SETS: ClassVar[frozenset[str]] = frozenset({"ifs", "depac"})

    urban_natural_lcz_to_10: bool = field(default=False, metadata={"serialize": True})
    """If true, pixels with urban IFS cover and LCZ 11-17 are coerced to LCZ 10."""

    urban_natural_lcz_to_natural_lsm: bool = field(
        default=False,
        metadata={"serialize": True},
    )
    """If true, pixels with urban IFS cover and LCZ 11-17 are remapped to natural IFS classes."""


@register_module
@dataclass
class FromTop10:
    """Apply Top10NL land-use classes to LSM fractions.

    This can be combined with ``FromLCZ`` as an overlay. When both are used,
    LCZ-derived values are built first and Top10 fractions are then applied.
    Supports all land-use sets, using direct Top10 ids where available and IFS
    mappings for catalogues that define IFS classes.
    """

    SUPPORTED_LAND_USE_SETS: ClassVar[frozenset[str]] = frozenset(
        {"basic", "build", "crop", "depac", "ifs"}
    )

    spatial_data_path: pathlib.Path = field(
        default=pathlib.Path(
            "DALES_input_generator/dales_openBC_setup/scripts/land_surface/spatial_data"
        ),
        metadata={"serialize": True},
    )
    top10_filename: str = field(
        default="top10nl_landuse_010m.nc", metadata={"serialize": True}
    )
    fill_north_sea: bool = field(default=False, metadata={"serialize": True})
    urban_natural_lcz_to_10: bool = field(
        default=False,
        metadata={"serialize": True},
    )
    """If true, urban Top10 pixels with LCZ 11-17 are coerced to LCZ 10."""

    urban_natural_lcz_to_natural_lsm: bool = field(
        default=False,
        metadata={"serialize": True},
    )
    """If true, urban Top10 pixels with LCZ 11-17 are remapped to natural IFS classes."""


@register_module
@dataclass
class FromBofek:
    """Apply BOFEK soil classes to ``index_soil`` for any land-use set."""

    SUPPORTED_LAND_USE_SETS: ClassVar[frozenset[str]] = frozenset(
        {"basic", "build", "crop", "depac", "ifs"}
    )

    spatial_data_path: pathlib.Path = field(
        default=pathlib.Path(
            "DALES_input_generator/dales_openBC_setup/scripts/land_surface/spatial_data"
        ),
        metadata={"serialize": True},
    )
    bofek_filename: str = field(
        default="BOFEK2012_010m.nc", metadata={"serialize": True}
    )
    bofek_profile_csv: str = field(
        default="BOFEK2012_profielen_versie2_1.csv", metadata={"serialize": True}
    )


@register_module
@dataclass
class FromNetCDF:
    """Use a user-supplied NetCDF file as land-surface data source.

    The file must contain:
      FRAC_WATER, FRAC_SEA  – water/ocean fractions
      FRAC_NATURE           – natural land fraction (sub-typed via ESA WorldCover)
      FRAC_TOWN             – urban fraction (drives SLuRB)
      D_Z0_town, D_BLD, D_BLD_HEIG, WALL_O_HOR  – optional SLuRB morphology
    This source supports the IFS and DEPAC land-use sets.
    """

    SUPPORTED_LAND_USE_SETS: ClassVar[frozenset[str]] = frozenset({"ifs", "depac"})

    path: str
    """Absolute or relative path to the NetCDF file."""

    esa_cache_dir: str | None = field(default=None, metadata={"serialize": True})
    """Optional directory for caching the ESA WorldCover reprojected tile."""


LSMSourceSpec = (
    FromLCZ
    | FromTop10
    | FromBofek
    | FromNetCDF
    | FromLS2D
    | LSMProfileSpec
    | LandUseModification
    | LandUseModifications
)


@register_module
@dataclass
class _PrioritizedSource:
    """Internal serializable record of a source added with an explicit priority.

    Sources are applied from lowest to highest priority, so higher-priority
    sources can override values supplied by earlier sources.
    """

    source: LSMSourceSpec = field(metadata={"serialize": True})
    priority: int = field(metadata={"serialize": True})

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject unsupported providers and non-integer priorities at runtime."""
        if not isinstance(self.source, LSMSourceSpec):
            raise TypeError(f"Unsupported LSM source type: {type(self.source)}")
        if not isinstance(self.priority, int):
            raise TypeError("Source priority must be an integer")
