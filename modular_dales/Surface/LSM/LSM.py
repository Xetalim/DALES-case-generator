import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Self, cast

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.ls2d_atmosphere import FromLS2D
from modular_dales.modular.namelist import nml_field
from modular_dales.modular.time_dependent_scalars import SOIL_DIM, value_at
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.Surface.LSM import lsm_netcdf, lsm_slurb, lsm_sources, plot_lsm
from modular_dales.Surface.LSM.LandUseModifications import (
    LandUseModification,
    LandUseModifications,
)
from modular_dales.Surface.LSM.LCZ import get_from_LCZ
from modular_dales.Surface.LSM.LSM_output_dales import LSM_output_dales, LsmModifier
from modular_dales.Surface.LSM.modular_temps_moisture import (
    SkinTempSpec,
    SoilMoistureSpec,
    SoilTemperatureMoistureFromHarmonie,
    SoilTempSpec,
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
)
from modular_dales.Surface.LSM.top10_bofek_ags import (
    apply_ags_parameters_to_lsm_writer,
    apply_bofek_to_lsm_writer,
    apply_top10_to_lsm_writer,
)
from modular_dales.Surface.LSM.translation_tables.landuse_types import LandUseSet
from modular_dales.vars import t_soil, theta_soil, thls, type_soil, z0h, z0m

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
class FromLCZ:
    """LCZ (Local Climate Zone) classification approach for LSM."""

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
    """

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
    """Apply BOFEK soil classes to ``index_soil`` in LSM input."""

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
class AGSParameters:
    """Enable AGS parameter fields in generated LSM NetCDF."""

    grass_planttype: int | None = field(default=None, metadata={"serialize": True})


@register_module
@dataclass
class FromNetCDF:
    """Use a user-supplied NetCDF file as land-surface data source.

    The file must contain:
      FRAC_WATER, FRAC_SEA  – water/ocean fractions
      FRAC_NATURE           – natural land fraction (sub-typed via ESA WorldCover)
      FRAC_TOWN             – urban fraction (drives SLuRB)
      D_Z0_town, D_BLD, D_BLD_HEIG, WALL_O_HOR  – optional SLuRB morphology
    """

    path: str
    """Absolute or relative path to the NetCDF file."""

    esa_cache_dir: str | None = field(default=None, metadata={"serialize": True})
    """Optional directory for caching the ESA WorldCover reprojected tile."""


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
    land_use_set: LandUseSet | None = field(default=None, metadata={"serialize": True})
    land_use_modifications: LandUseModifications = field(
        default_factory=LandUseModifications,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    from_lcz: FromLCZ | None = field(
        default=None,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    from_top10: FromTop10 | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    from_bofek: FromBofek | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    ags_parameters: AGSParameters | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    from_netcdf: Optional["FromNetCDF"] = field(
        default=None,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    from_ls2d: FromLS2D | None = field(
        default=None,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    skin_temperature: SkinTempSpec | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    tskin_laqu: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    soil_temperature: SoilTempSpec | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    soil_moisture: SoilMoistureSpec | None = field(
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

    def __add__(self, obj) -> Self:
        """Add configurations to LSM module.

        Args:
            obj: LandUseModification, LandUseModifications, FromLCZ,
                FromTop10, FromBofek, AGSParameters or temperature/moisture helpers

        Returns:
            self for chaining
        """
        if isinstance(obj, LandUseModification):
            self.land_use_modifications += obj
        elif isinstance(obj, LandUseModifications):
            for mod in obj.modifications:
                self.land_use_modifications += mod
        elif isinstance(obj, FromLCZ):
            self.from_lcz = obj
        elif isinstance(obj, FromTop10):
            self.from_top10 = obj
        elif isinstance(obj, FromBofek):
            self.from_bofek = obj
        elif isinstance(obj, AGSParameters):
            self.ags_parameters = obj
        elif isinstance(obj, FromNetCDF):
            self.from_netcdf = obj
        elif isinstance(obj, FromLS2D):
            self.from_ls2d = obj
        elif isinstance(obj, (UniformSkinTemperature, VaryingSkinTemperature)):
            self.skin_temperature = obj
        elif isinstance(obj, (UniformSoilTemperature, VaryingSoilTemperature)):
            self.soil_temperature = obj
        elif isinstance(
            obj,
            (
                UniformSoilMoisture,
                VaryingSoilMoisture,
            ),
        ):
            self.soil_moisture = obj
        elif isinstance(obj, SoilTemperatureMoistureFromHarmonie):
            self.soil_moisture = obj
            self.soil_temperature = obj
            if obj.use_as_tskin:
                self.skin_temperature = obj
        else:
            raise TypeError(
                "Expected LandUseModification/FromLCZ/FromNetCDF/FromLS2D/"
                "Expected LandUseModification/FromLCZ/FromTop10/FromBofek/"
                "AGSParameters/FromLS2D/"
                "SkinTemperatures/SoilTemperatures/SoilMoistures, got "
                f"{type(obj)}"
            )
        return self

    def __iadd__(self, modification) -> Self:
        """In-place addition of modification."""
        return self.__add__(modification)

    def prepare_calculation(self):
        return self.prepare_calculations()

    def prepare_calculations(self) -> None:
        """Build land use, apply overlays and soil state, then finalize tile counts."""
        self.exists_soil_temp_moisture_skin_temp()
        self.slurb_module = (
            cast(SLURBModule, self.retrieve_module(SLURBModule))
            if self.module_exists(SLURBModule)
            else None
        )
        writer = self._create_lsm_writer()
        self.lsm_writer = writer
        writer.standard_fill_geometry_modification()
        self._apply_land_use_source(writer)
        self._apply_top10_overlay(writer)
        self._apply_bofek_overlay(writer)
        self._apply_thermal_moisture(writer)
        self._apply_ls2d_override(writer)
        if self.ags_parameters is not None:
            apply_ags_parameters_to_lsm_writer(
                writer, grass_planttype=self.ags_parameters.grass_planttype
            )
        writer.trim_landuse()
        self.nlu = writer.nlu + 1

    def _create_lsm_writer(self) -> LSM_output_dales:
        """Select the explicit or source-default catalogue, independently of SLURB."""
        land_use_set = self.land_use_set
        if land_use_set is None:
            land_use_set = (
                LandUseSet.IFS
                if self.from_lcz is not None or self.from_netcdf is not None
                else LandUseSet.DEPAC
            )

        return LSM_output_dales(
            self.grid,
            land_use_set,
            soil_levels=cast(int, self.kmax_soil),
            include_slurb=self.slurb_module is not None,
        )

    def _apply_land_use_source(self, writer: LSM_output_dales) -> None:
        """Apply one source in priority order: NetCDF, LCZ, explicit geometry."""
        if self.from_netcdf is not None:
            self._apply_netcdf_source(writer, self.from_netcdf)
        elif self.from_lcz is not None:
            self._apply_lcz_source(writer, self.from_lcz)
        else:
            self._apply_land_use_modifications(writer)

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
        from modular_dales.Surface.LSM.LCZ.from_netcdf import load_from_netcdf

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
        dataset = get_from_LCZ.do_everything(
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

    def _apply_land_use_modifications(self, writer: LSM_output_dales) -> None:
        """Apply geometric fractions, initialize tile parameters and fill bare soil."""
        modifier = LsmModifier(writer, self.grid)
        for modification in self.land_use_modifications.modifications:
            modifier.apply_modification(modification)
        writer.init_lutypes_ifs()
        writer.recalculate_remaining_cover()

    def _apply_top10_overlay(self, writer: LSM_output_dales) -> None:
        """Overlay optional Top10NL fractions after validating options and input."""
        source = self.from_top10
        if source is None:
            return
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

    def _apply_bofek_overlay(self, writer: LSM_output_dales) -> None:
        """Map optional BOFEK soil classes after validating both required files."""
        source = self.from_bofek
        if source is None:
            return
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

    def exists_soil_temp_moisture_skin_temp(self):
        # check if soil temp exists
        if self.soil_temperature is None:
            raise ValueError(
                "Soil temperature profiles must be specified for LSMModule. "
                "Add a SoilTemperatures object to LSMModule: lsm += SoilTemperatures()"
            )
        # check if soil moisture exists
        if self.soil_moisture is None:
            raise ValueError(
                "Soil moisture profiles must be specified for LSMModule. "
                "Add a SoilMoistures object to LSMModule: lsm += SoilMoistures()"
            )
        # check if skin temp exists
        if self.skin_temperature is None:
            raise ValueError(
                "Skin temperature profiles must be specified for LSMModule. "
                "Add a SkinTemperatures object to LSMModule: lsm += SkinTemperatures()"
            )

    def _resolve_laqu_skin_temperature_override(self) -> float | None:
        candidate = getattr(self, "tskin_laqu", None)
        if self.skin_temperature is not None:
            module_candidate = getattr(
                self.skin_temperature, "aquatic_skin_temperature", None
            )
            if module_candidate is not None:
                candidate = module_candidate
        if candidate is None:
            return None
        return float(candidate)

    def _set_laqu_skin_temperature(self, writer: LSM_output_dales) -> None:
        """Apply the aquatic skin-temperature override when configured."""
        laqu_skin_temp = self._resolve_laqu_skin_temperature_override()
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

        if self.from_ls2d is None:
            return

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

    def _apply_thermal_moisture(self, writer: LSM_output_dales) -> None:
        """Set configured moisture and temperatures, then the aquatic override."""
        if isinstance(self.soil_moisture, (UniformSoilMoisture)):
            writer.soil.set_uniform_moisture(self.soil_moisture.soil_moisture)
        elif isinstance(self.soil_moisture, (VaryingSoilMoisture)):
            writer.soil.set_moisture(self.soil_moisture.soil_moisture)
        elif isinstance(self.soil_moisture, (SoilTemperatureMoistureFromHarmonie)):
            writer.soil.set_moisture(
                self.soil_moisture.get_soil_moisture_array(
                    self.grid, cast(list[float], self.dz_soil)
                )
            )
        else:
            raise TypeError(
                "Invalid soil moisture configuration. Must be UniformSoilMoisture, VaryingSoilMoisture, or SoilTemperatureMoistureFromHarmonie."
            )

        if isinstance(
            self.soil_temperature,
            (UniformSoilTemperature),
        ):
            writer.soil.set_uniform_temperature(self.soil_temperature.soil_temperature)
        elif isinstance(
            self.soil_temperature,
            (VaryingSoilTemperature),
        ):
            writer.soil.set_temperature(self.soil_temperature.soil_temperature)
        elif isinstance(self.soil_temperature, (SoilTemperatureMoistureFromHarmonie)):
            writer.soil.set_temperature(
                self.soil_temperature.get_soil_temperature_array(
                    self.grid, cast(list[float], self.dz_soil)
                )
            )
        else:
            raise TypeError(
                "Invalid soil temperature configuration. Must be UniformSoilTemperature, VaryingSoilTemperature, or SoilTemperatureMoistureFromHarmonie."
            )

        if isinstance(self.skin_temperature, (UniformSkinTemperature)):
            writer.set_skin_temperature(
                self.skin_temperature.skin_temperature, lu_type="all"
            )
        elif isinstance(self.skin_temperature, (VaryingSkinTemperature)):
            writer.set_skin_temperature_array(
                self.skin_temperature.skin_temperature, lu_type="all"
            )
        elif isinstance(self.skin_temperature, (SoilTemperatureMoistureFromHarmonie)):
            writer.set_skin_temperature_array(
                self.skin_temperature.get_tskin_array(),
                lu_type="all",
            )
        else:
            raise TypeError(
                "Invalid skin temperature configuration. Must be UniformSkinTemperature or VaryingSkinTemperature."
            )

        self._set_laqu_skin_temperature(writer)

    def write_files(self):
        """Write LSM input files and generate plots."""

        # Re-apply LS2D-derived overrides at write time. This guarantees
        # values such as ps/z0 are available even when LSM prepare happened
        # before LS2D prepare due to module ordering.
        if self.lsm_writer is None:
            raise ValueError("Prepare the LSM before writing files")
        self._apply_ls2d_override(self.lsm_writer)
        self.apply_namelist_from_fields()

        lsm_netcdf.save_netcdf(self.lsm_writer, self.output_path / "input", self.exp_id)
        plot_lsm.plot_lsm_cover(
            pathlib.Path(self.output_path) / "input" / f"lsm.inp_{self.exp_id:03d}.nc",
            pathlib.Path(self.output_path) / "profiles",
        )
