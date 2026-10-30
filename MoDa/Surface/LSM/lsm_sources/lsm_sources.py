"""Define prioritized LSM sources and map datasets onto typed land-use state."""

import pathlib
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from moda.Atmosphere.ls2d_atmosphere import FromLS2D as FromLS2D
from moda.modular.sources import Source
from moda.module_registry import register_module
from moda.Surface.LSM.lsm_data import (
    FloatArray,
    LandUseName,
    LandUseState,
    SoilFields,
)
from moda.Surface.LSM.lsm_sources.lu_modifications import (
    LandUseModification as LandUseModification,
    LandUseModifications as LandUseModifications,
)
from moda.Surface.LSM.lsm_sources.modular_temps_moisture import (
    LSMProfileSpec as LSMProfileSpec,
)
from moda.Surface.LSM.lsm_sources.slurb_modifications import (
    SlurbModification as SlurbModification,
    SlurbModifications as SlurbModifications,
)
from .context import LSMSourceContext
from . import slurb_modifications
from .top10_bofek_ags import apply_bofek_to_lsm_writer, apply_top10_to_lsm_writer


def _validate_lcz_options(source: "FromLCZ | FromTop10") -> None:
    if source.urban_natural_lcz_to_10 and source.urban_natural_lcz_to_natural_lsm:
        raise ValueError(
            f"{type(source).__name__} options urban_natural_lcz_to_10 and "
            "urban_natural_lcz_to_natural_lsm are mutually exclusive"
        )


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

    def apply_source(self, context: LSMSourceContext) -> None:
        from moda.Surface.LSM.LCZ import get_from_lcz

        _validate_lcz_options(self)
        grid = context.module.grid
        if grid.crs is None:
            raise ValueError("A valid projection is required for real-world LSM maps")
        dataset = get_from_lcz.do_everything(
            grid,
            urban_natural_lcz_to_10=self.urban_natural_lcz_to_10,
            urban_natural_lcz_to_natural_lsm=self.urban_natural_lcz_to_natural_lsm,
        )
        writer = context.writer
        apply_lcz(writer.land_use, writer.soil, dataset)
        writer.LCZ_ds = dataset
        writer.init_lutypes_ifs()
        writer.recalculate_remaining_cover()
        generator = context.module._init_slurb_generator()
        if generator is not None:
            urban_cover = next(
                (
                    writer.land_use_fields[name].cover
                    for name, state in writer.land_use.items()
                    if state.definition.ifs_id == 20
                ),
                None,
            )
            slurb_modifications.apply_slurb_parameters_lcz(
                dataset, generator, urban_cover
            )


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

    def apply_source(self, context: LSMSourceContext) -> None:
        _validate_lcz_options(self)
        path = pathlib.Path(self.spatial_data_path) / self.top10_filename
        if not path.exists():
            raise FileNotFoundError(
                f"Top10 file not found: {path}. "
                "Set FromTop10.spatial_data_path/top10_filename accordingly."
            )
        apply_top10_to_lsm_writer(
            context.writer,
            top10_path=path,
            fill_north_sea=self.fill_north_sea,
            urban_natural_lcz_to_10=self.urban_natural_lcz_to_10,
            urban_natural_lcz_to_natural_lsm=self.urban_natural_lcz_to_natural_lsm,
        )


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

    def apply_source(self, context: LSMSourceContext) -> None:
        map_path = pathlib.Path(self.spatial_data_path) / self.bofek_filename
        csv_path = pathlib.Path(self.spatial_data_path) / self.bofek_profile_csv
        if not map_path.exists():
            raise FileNotFoundError(
                f"BOFEK map file not found: {map_path}. "
                "Set FromBofek.spatial_data_path/bofek_filename accordingly."
            )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"BOFEK profile table not found: {csv_path}. "
                "Set FromBofek.spatial_data_path/bofek_profile_csv accordingly."
            )
        apply_bofek_to_lsm_writer(
            context.writer, bofek_nc_path=map_path, bofek_csv_path=csv_path
        )


@register_module
@dataclass
class FromNetCDF:
    """Use a user-supplied NetCDF file as land-surface data source.

    The file must contain:
        * ``FRAC_WATER``, ``FRAC_SEA``: water and ocean fractions.
        * ``FRAC_NATURE``: natural land fraction, subtyped using ESA WorldCover.
        * ``FRAC_TOWN``: urban fraction used by SLUrb.
        * ``D_Z0_town``, ``D_BLD``, ``D_BLD_HEIG``, ``WALL_O_HOR``: optional SLUrb
            morphology.

    This source supports the IFS and DEPAC land-use sets.
    """

    SUPPORTED_LAND_USE_SETS: ClassVar[frozenset[str]] = frozenset({"ifs", "depac"})

    path: str
    """Absolute or relative path to the NetCDF file."""

    esa_cache_dir: str | None = field(default=None, metadata={"serialize": True})
    """Optional directory for caching the ESA WorldCover reprojected tile."""

    def apply_source(self, context: LSMSourceContext) -> None:
        from moda.Surface.LSM.LCZ.from_netcdf import load_from_netcdf

        cache = (
            pathlib.Path(self.esa_cache_dir) if self.esa_cache_dir is not None else None
        )
        dataset = load_from_netcdf(self.path, context.module.grid, esa_cache_dir=cache)
        writer = context.writer
        apply_from_netcdf(writer.land_use, writer.soil, dataset)
        writer.init_lutypes_ifs()
        writer.recalculate_remaining_cover()
        generator = context.module._init_slurb_generator()
        if generator is not None:
            slurb_modifications.apply_from_netcdf(dataset, generator)


LSMSourceSpec = Source[LSMSourceContext]


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
        if not isinstance(self.source, Source):
            raise TypeError(f"Unsupported LSM source type: {type(self.source)}")
        if not isinstance(self.priority, int):
            raise TypeError("Source priority must be an integer")
