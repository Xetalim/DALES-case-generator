import logging
import pathlib
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from MoDa.Atmosphere.ls2d_atmosphere import FromLS2D
from MoDa.modular.namelist import nml_field
from MoDa.modular.time_dependent_scalars import SOIL_DIM, value_at
from MoDa.module_registry import register_module
from MoDa.Surface.LSM import lsm_netcdf, lsm_slurb, lsm_sources, plot_lsm
from MoDa.Surface.LSM.LCZ import get_from_lcz
from MoDa.Surface.LSM.lsm_output import LSM_output_dales, LsmModifier
from MoDa.Surface.LSM.lsm_sources import (
    FromBofek,
    FromLCZ,
    FromNetCDF,
    FromTop10,
    LSMSourceSpec,
    _PrioritizedSource,
)
from MoDa.Surface.LSM.lu_modifications import (
    LandUseModification,
    LandUseModifications,
)
from MoDa.Surface.LSM.modular_temps_moisture import (
    LSMProfileSpec,
    SoilTemperatureMoistureFromHarmonie,
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
)
from MoDa.Surface.LSM.top10_bofek_ags import (
    apply_ags_parameters_to_lsm_writer,
    apply_bofek_to_lsm_writer,
    apply_top10_to_lsm_writer,
)
from MoDa.Surface.LSM.translation_tables.landuse_types import LandUseSet
from MoDa.vars import t_soil, theta_soil, thls, type_soil, z0h, z0m

from .base import BaseLSMModule
from .SLuRB.slurb import SLURBModule, slbCreatorClass

logger = logging.getLogger(__name__)


def _soil_profile_on_levels(
    profile: xr.DataArray, nz_soil: int, target_depths: np.ndarray | None
) -> xr.DataArray:
    """Soil profile on ``nz_soil`` levels: by depth when known, else by relative level index."""
    n_source = profile.sizes[SOIL_DIM]
    if n_source == nz_soil:
        return profile
    if n_source < 2:
        return xr.DataArray(np.full(nz_soil, float(profile[0])), dims=(SOIL_DIM,))
    if target_depths is not None and SOIL_DIM in profile.coords:
        profile = profile.sortby(SOIL_DIM)
        depths = profile[SOIL_DIM].values
        return profile.interp({SOIL_DIM: np.clip(target_depths, depths[0], depths[-1])})
    relative = profile.assign_coords({SOIL_DIM: np.linspace(0.0, 1.0, n_source)})
    return relative.interp({SOIL_DIM: np.linspace(0.0, 1.0, nz_soil)})


@register_module
@dataclass
class LSMModule(BaseLSMModule):
    """Land Surface Model simulation module.

    When added to a simulation, automatically enables LSM by setting isurf=11
    in the namelist (unless overridden by a surface module like ConstantFluxesModule).

    Args:
        sim: Parent simulation instance
        isurf: Surface scheme selector (11 for LSM)
        ps: Surface pressure (Pa)
        z0mav: Momentum roughness length (m)
        z0hav: Heat roughness length (m)
        albedoav: Albedo (dimensionless)
        iinterp_t: Interpolation method for temperature (1-4)
        iinterp_theta: Interpolation method for potential temperature (1-4)
        dz_soil: List of soil layer thicknesses (m)
    """

    lheterogeneous: bool = nml_field(
        default=True,
        section="NAMLSM",
        key="lheterogeneous",
        required=True,
        metadata={"serialize": True},
    )
    land_use_set: LandUseSet = field(kw_only=True, metadata={"serialize": True})
    sources: list[_PrioritizedSource] = field(
        default_factory=list,
        metadata={"serialize": True},
    )
    ags_parameters: bool = field(
        default=False,
        init=True,
        metadata={"serialize": True},
    )
    tskin_laqu: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    lsm_writer: LSM_output_dales | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    slurb_module: SLURBModule | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )

    module_name: str = field(
        default="LSMModule", init=False, metadata={"serialize": False}
    )

    def add(
        self,
        obj: LSMSourceSpec | list[LSMSourceSpec],
        *,
        priority: int,
    ) -> None:
        """Add sources at an explicit priority; higher priorities apply later."""
        providers = obj if isinstance(obj, list) else [obj]
        entries = [_PrioritizedSource(source, priority) for source in providers]
        self.sources.extend(entries)

    def prepare_calculation(self) -> None:
        """Build land use, apply overlays and soil state, then finalize tile counts."""
        self.slurb_module = (
            self.retrieve_module(SLURBModule)
            if self.module_exists(SLURBModule)
            else None
        )
        writer = self._create_lsm_writer()
        self.lsm_writer = writer
        writer.standard_fill_geometry_modification()
        self._apply_sources(writer)
        if self.ags_parameters:
            apply_ags_parameters_to_lsm_writer(writer)
        writer.trim_landuse()
        self.nlu = writer.nlu + 1

    def _create_lsm_writer(self) -> LSM_output_dales:
        """Create the writer using the explicitly selected land-use catalogue."""
        self._validate_sources_land_use_set()

        return LSM_output_dales(
            self.grid,
            self.land_use_set,
            soil_levels=self.kmax_soil,
            include_slurb=self.slurb_module is not None,
        )

    def _validate_sources_land_use_set(self) -> None:
        """Require an explicit catalogue and reject incompatible land-use sources."""
        if self.land_use_set is None:
            raise ValueError("Choose a land_use_set before preparing LSMModule")
        self.land_use_set = LandUseSet(self.land_use_set)
        for entry in self.sources:
            if not isinstance(entry, _PrioritizedSource):
                raise TypeError(
                    "Use LSMModule.add(source, priority=...) to add sources"
                )
            entry.validate()
            supported = getattr(entry.source, "SUPPORTED_LAND_USE_SETS", None)
            if supported is not None and self.land_use_set.value not in supported:
                supported_names = ", ".join(sorted(supported))
                raise ValueError(
                    f"{type(entry.source).__name__} does not support "
                    f"land-use set {self.land_use_set.value!r}; supported sets: "
                    f"{supported_names}"
                )

    def _apply_sources(self, writer: LSM_output_dales) -> None:
        """Apply configured sources from lowest to highest priority."""
        self._validate_sources_land_use_set()
        for entry in sorted(self.sources, key=lambda source: source.priority):
            source = entry.source
            if isinstance(source, FromNetCDF):
                self._apply_netcdf_source(writer, source)
            elif isinstance(source, FromLCZ):
                self._apply_lcz_source(writer, source)
            elif isinstance(source, FromTop10):
                self._apply_top10_overlay(writer, source)
            elif isinstance(source, FromBofek):
                self._apply_bofek_overlay(writer, source)
            elif isinstance(source, FromLS2D):
                self._apply_ls2d_override(writer)
            elif isinstance(source, (LandUseModification, LandUseModifications)):
                self._apply_land_use_modifications(writer, source)
            elif isinstance(source, LSMProfileSpec):
                self._apply_thermal_moisture_source(writer, source)
            else:
                raise TypeError(f"Unsupported LSM source type: {type(source)}")

    def _init_slurb_generator(self) -> slbCreatorClass | None:
        """Initialize source-driven SLURB morphology only when SLURB is present."""
        if self.slurb_module is None:
            return None
        self.slurb_module.init_generator()
        return self.slurb_module.slb_generator

    @staticmethod
    def _validate_lcz_options(source: FromLCZ | FromTop10) -> None:
        """Reject conflicting urban-to-natural remapping options for either source."""
        if source.urban_natural_lcz_to_10 and source.urban_natural_lcz_to_natural_lsm:
            raise ValueError(
                f"{type(source).__name__} options urban_natural_lcz_to_10 and "
                "urban_natural_lcz_to_natural_lsm are mutually exclusive"
            )

    def _apply_netcdf_source(
        self, writer: LSM_output_dales, source: FromNetCDF
    ) -> None:
        """Load external fractions and optional morphology onto the simulation grid."""
        from MoDa.Surface.LSM.LCZ.from_netcdf import load_from_netcdf

        esa_cache = (
            pathlib.Path(source.esa_cache_dir)
            if source.esa_cache_dir is not None
            else None
        )
        dataset = load_from_netcdf(source.path, self.grid, esa_cache_dir=esa_cache)
        lsm_sources.apply_from_netcdf(writer.land_use, writer.soil, dataset)
        writer.init_lutypes_ifs()
        writer.recalculate_remaining_cover()
        generator = self._init_slurb_generator()
        if generator is not None:
            lsm_slurb.apply_from_netcdf(dataset, generator)

    def _apply_lcz_source(self, writer: LSM_output_dales, source: FromLCZ) -> None:
        """Build LCZ fractions before applying optional LCZ-derived SLURB fields."""
        self._validate_lcz_options(source)
        if self.grid.crs is None:
            raise ValueError("A valid projection is required for real-world LSM maps")
        dataset = get_from_lcz.do_everything(
            self.grid,
            urban_natural_lcz_to_10=source.urban_natural_lcz_to_10,
            urban_natural_lcz_to_natural_lsm=source.urban_natural_lcz_to_natural_lsm,
        )
        lsm_sources.apply_lcz(writer.land_use, writer.soil, dataset)
        writer.LCZ_ds = dataset
        writer.init_lutypes_ifs()
        writer.recalculate_remaining_cover()
        generator = self._init_slurb_generator()
        if generator is not None:
            urban_cover = next(
                (
                    writer.land_use_fields[name].cover
                    for name, state in writer.land_use.items()
                    if state.definition.ifs_id == 20
                ),
                None,
            )
            lsm_slurb.apply_slurb_parameters_lcz(dataset, generator, urban_cover)

    def _apply_land_use_modifications(
        self,
        writer: LSM_output_dales,
        source: LandUseModification | LandUseModifications,
    ) -> None:
        """Apply geometric fractions, initialize tile parameters and fill bare soil."""
        modifier = LsmModifier(writer, self.grid)
        modifications = (
            source.modifications
            if isinstance(source, LandUseModifications)
            else [source]
        )
        for modification in modifications:
            modifier.apply_modification(modification)
        writer.init_lutypes_ifs()
        writer.recalculate_remaining_cover()

    def _apply_top10_overlay(self, writer: LSM_output_dales, source: FromTop10) -> None:
        """Overlay optional Top10NL fractions after validating options and input."""
        self._validate_lcz_options(source)
        top10_path = pathlib.Path(source.spatial_data_path) / source.top10_filename
        if not top10_path.exists():
            raise FileNotFoundError(
                f"Top10 file not found: {top10_path}. "
                "Set FromTop10.spatial_data_path/top10_filename accordingly."
            )
        apply_top10_to_lsm_writer(
            writer,
            top10_path=top10_path,
            fill_north_sea=source.fill_north_sea,
            urban_natural_lcz_to_10=source.urban_natural_lcz_to_10,
            urban_natural_lcz_to_natural_lsm=source.urban_natural_lcz_to_natural_lsm,
        )

    def _apply_bofek_overlay(self, writer: LSM_output_dales, source: FromBofek) -> None:
        """Map optional BOFEK soil classes after validating both required files."""
        bofek_nc_path = pathlib.Path(source.spatial_data_path) / source.bofek_filename
        bofek_csv_path = (
            pathlib.Path(source.spatial_data_path) / source.bofek_profile_csv
        )
        if not bofek_nc_path.exists():
            raise FileNotFoundError(
                f"BOFEK map file not found: {bofek_nc_path}. "
                "Set FromBofek.spatial_data_path/bofek_filename accordingly."
            )
        if not bofek_csv_path.exists():
            raise FileNotFoundError(
                f"BOFEK profile table not found: {bofek_csv_path}. "
                "Set FromBofek.spatial_data_path/bofek_profile_csv accordingly."
            )
        apply_bofek_to_lsm_writer(
            writer, bofek_nc_path=bofek_nc_path, bofek_csv_path=bofek_csv_path
        )

    def _set_laqu_skin_temperature(
        self, writer: LSM_output_dales, aquatic_skin_temperature: float | None = None
    ) -> None:
        """Apply the aquatic skin-temperature override when configured."""
        laqu_skin_temp = (
            self.tskin_laqu
            if aquatic_skin_temperature is None
            else aquatic_skin_temperature
        )
        if laqu_skin_temp is None:
            return
        writer.set_skin_temperature_laqu(laqu_skin_temp)

    def _apply_ls2d_override(self, writer: LSM_output_dales) -> None:
        """Apply provided soil state, roughness and skin temperature (opt-in via ``FromLS2D``).

        Requests ``t_soil``/``theta_soil`` (on ``zs``), ``type_soil``,
        ``z0m``/``z0h`` and the ``thls`` series from ``sim.forcings()``, as
        provided by e.g. :class:`LS2DAtmosphereModule`. Surface pressure is
        taken from the provided ``psurf`` through the ``ps`` field itself.
        """

        forcings = self.sim.forcings()
        soil = writer.soil
        nz_soil, jtot, itot = soil.t_soil.shape
        dz_soil = (
            np.asarray(self.dz_soil, dtype=float)
            if self.dz_soil is not None and len(self.dz_soil) == nz_soil
            else None
        )
        target_depths = None if dz_soil is None else np.cumsum(dz_soil) - 0.5 * dz_soil

        for name, target in (
            (t_soil.name, soil.t_soil),
            (theta_soil.name, soil.theta_soil),
        ):
            provided = forcings.initial.get(name)
            if provided is not None:
                profile = _soil_profile_on_levels(provided, nz_soil, target_depths)
                target[:, :, :] = profile.values[:, np.newaxis, np.newaxis]

        provided_type = forcings.initial.get(type_soil.name)
        if provided_type is not None:
            values = np.asarray(provided_type.values)
            if values.shape == (jtot, itot):
                soil.index_soil[:, :, :] = values.astype(int)[np.newaxis, :, :]
            else:
                soil.index_soil[:, :, :] = int(values.ravel()[0])

        for name, attr in ((z0m.name, "z0mav"), (z0h.name, "z0hav")):
            if name in forcings.initial:
                setattr(self, attr, float(forcings.initial[name]))

        skin = forcings.series.get(thls.name)
        if skin is not None:
            writer.set_skin_temperature(float(value_at(skin, 0.0)), lu_type="all")
            self._set_laqu_skin_temperature(writer)

        logger.info(
            "LSMModule: applied provided soil profiles, soil index, roughness and skin temperature where available"
        )

    def _apply_thermal_moisture_source(
        self, writer: LSM_output_dales, source: LSMProfileSpec
    ) -> None:
        """Apply only the fields supplied by this thermal/moisture source."""
        if isinstance(source, UniformSoilMoisture):
            writer.soil.set_uniform_moisture(source.soil_moisture)
        elif isinstance(source, VaryingSoilMoisture):
            writer.soil.set_moisture(source.soil_moisture)
        elif isinstance(source, UniformSoilTemperature):
            writer.soil.set_uniform_temperature(source.soil_temperature)
        elif isinstance(source, VaryingSoilTemperature):
            writer.soil.set_temperature(source.soil_temperature)
        elif isinstance(source, UniformSkinTemperature):
            writer.set_skin_temperature(source.skin_temperature, lu_type="all")
            self._set_laqu_skin_temperature(writer, source.aquatic_skin_temperature)
        elif isinstance(source, VaryingSkinTemperature):
            writer.set_skin_temperature_array(source.skin_temperature, lu_type="all")
            self._set_laqu_skin_temperature(writer, source.aquatic_skin_temperature)
        elif isinstance(source, SoilTemperatureMoistureFromHarmonie):
            self._apply_harmonie_source(writer, source)
        else:
            raise TypeError(f"Unsupported thermal/moisture source type: {type(source)}")

    def _apply_harmonie_source(
        self, writer: LSM_output_dales, source: SoilTemperatureMoistureFromHarmonie
    ) -> None:
        """Load HARMONIE soil state and optionally apply its skin temperature."""
        source.get_soil_temp_moisture_arrays(self.grid, self.dz_soil)
        writer.soil.set_moisture(
            source.get_soil_moisture_array(self.grid, self.dz_soil)
        )
        writer.soil.set_temperature(
            source.get_soil_temperature_array(self.grid, self.dz_soil)
        )
        if source.use_as_tskin:
            writer.set_skin_temperature_array(source.get_tskin_array(), lu_type="all")
            self._set_laqu_skin_temperature(writer, source.aquatic_skin_temperature)

    def write_files(self) -> None:
        """Write LSM input files and generate plots."""
        if self.lsm_writer is None:
            raise ValueError("Prepare the LSM before writing files")
        self.apply_namelist_from_fields()

        lsm_netcdf.save_netcdf(self.lsm_writer, self.output_path / "input", self.exp_id)
        plot_lsm.plot_lsm_cover(
            pathlib.Path(self.output_path) / "input" / f"lsm.inp_{self.exp_id:03d}.nc",
            pathlib.Path(self.output_path) / "profiles",
        )
