import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.ls2d_atmosphere import FromLS2D
from modular_dales.Geometry.geometry_modification import (
    GeometricModification,
)
from modular_dales.modular.time_dependent_scalars import SOIL_DIM, value_at
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.Surface.LSM import plot_lsm
from modular_dales.Surface.LSM.LSM_output_dales import LSM_output_dales, LsmModifier
from modular_dales.Surface.LSM.modular_temps_moisture import (
    SoilTemperatureMoistureFromHarmonie,
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
from modular_dales.Surface.LSM.translation_tables import landuse_types
from modular_dales.vars import t_soil, theta_soil, thls, type_soil, z0h, z0m

from .base import BaseLSMModule
from .SLuRB.slurb import SLURBModule

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
class LandUseModification(GeometricModification):
    """Single land use modification."""

    type: str = "grs"
    """Land use type (grs, urb, fbd, etc.)"""
    frac: float = 1.0
    """Optional fractional land-use coverage for selected cells."""
    mode: str = field(default="replace")
    """How to apply fraction on selected cells: replace or add."""


@register_module
@dataclass
class LandUseModifications:
    """Collection of land use modifications."""

    modifications: list[LandUseModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def __add__(self, modification: LandUseModification) -> "LandUseModifications":
        """Add a modification."""
        self.modifications.append(modification)
        return self

    def __iadd__(self, modification: LandUseModification) -> "LandUseModifications":
        """In-place addition."""
        return self.__add__(modification)


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

    lheterogeneous: bool = field(
        default=True,
        metadata={
            "nml": "NAMLSM",
            "key": "lheterogeneous",
            "required": True,
            "serialize": True,
        },
    )
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
    skin_temperature: UniformSkinTemperature | VaryingSkinTemperature | SoilTemperatureMoistureFromHarmonie | None = field(
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
    soil_temperature: UniformSoilTemperature | VaryingSoilTemperature | SoilTemperatureMoistureFromHarmonie | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    soil_moisture: UniformSoilMoisture | VaryingSoilMoisture | SoilTemperatureMoistureFromHarmonie | None = field(
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

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "LSMModule"

    def __add__(self, obj) -> "LSMModule":
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

    def __iadd__(self, modification) -> "LSMModule":
        """In-place addition of modification."""
        return self.__add__(modification)

    def prepare_calculation(self):
        return self.prepare_calculations()

    def prepare_calculations(self):
        """Set up LSM with land use types and soil configuration."""

        self.exists_soil_temp_moisture_skin_temp()

        if self.module_exists(SLURBModule):
            self.slurb_module = self.retrieve_module(SLURBModule)

        # Determine land use types
        if self.from_lcz is not None or self.from_netcdf is not None:
            lu_types = landuse_types.lu_types_ifs
        else:
            lu_types = landuse_types.lu_types_depac.copy()
            if self.slurb_module:
                lu_types["slb"] = landuse_types.slb

        # Initialize LSM writer
        self.lsm_writer = LSM_output_dales(
            self.grid,
            lu_types,
            soil_levels=self.kmax_soil,
        )

        # Fill standard geometry
        self.lsm_writer.standard_fill_geometry_modification()

        # Configure data source path for land-use + optional SLURB morphology.
        if self.from_netcdf is not None:
            from pathlib import Path

            from modular_dales.Surface.LSM.LCZ.from_netcdf import load_from_netcdf

            esa_cache = (
                Path(self.from_netcdf.esa_cache_dir)
                if self.from_netcdf.esa_cache_dir is not None
                else None
            )
            nc_ds = load_from_netcdf(
                self.from_netcdf.path, self.grid, esa_cache_dir=esa_cache
            )

            if self.slurb_module is not None:
                self.slurb_module.init_generator()
            slb_gen = (
                self.slurb_module.slb_generator
                if self.slurb_module is not None
                else None
            )
            self.lsm_writer.apply_from_netcdf(nc_ds, slb_gen)
        elif self.from_lcz is not None:
            if (
                self.from_lcz.urban_natural_lcz_to_10
                and self.from_lcz.urban_natural_lcz_to_natural_lsm
            ):
                raise ValueError(
                    "FromLCZ options urban_natural_lcz_to_10 and "
                    "urban_natural_lcz_to_natural_lsm are mutually exclusive"
                )

            self.lsm_writer.from_lcz(
                urban_natural_lcz_to_10=self.from_lcz.urban_natural_lcz_to_10,
                urban_natural_lcz_to_natural_lsm=self.from_lcz.urban_natural_lcz_to_natural_lsm,
            )

            # Apply LCZ-derived SLURB parameters if SLURB is enabled.
            if self.slurb_module is not None:
                self.slurb_module.init_generator()
                self.lsm_writer.apply_slurb_parameters_lcz(
                    self.slurb_module.slb_generator
                )
        else:
            # Standard non-LCZ path: apply explicit land-use modifications only.
            modifier = LsmModifier(self.lsm_writer, self.grid)
            for modification in self.land_use_modifications.modifications:
                modifier.apply_modification(modification)
            self.lsm_writer = modifier.lsm_input
            self.lsm_writer.init_lutypes_ifs()
            self.lsm_writer.recalculate_remaining_cover()

        # Optional overlay from Top10NL; this can be used standalone or on top
        # of LCZ-derived fields.
        if self.from_top10 is not None:
            if (
                self.from_top10.urban_natural_lcz_to_10
                and self.from_top10.urban_natural_lcz_to_natural_lsm
            ):
                raise ValueError(
                    "FromTop10 options urban_natural_lcz_to_10 and "
                    "urban_natural_lcz_to_natural_lsm are mutually exclusive"
                )

            top10_path = (
                pathlib.Path(self.from_top10.spatial_data_path)
                / self.from_top10.top10_filename
            )
            if not top10_path.exists():
                raise FileNotFoundError(
                    f"Top10 file not found: {top10_path}. "
                    "Set FromTop10.spatial_data_path/top10_filename accordingly."
                )
            # Pass optional LCZ-over-Top10 overrides via writer attributes to keep
            # apply_top10_to_lsm_writer backward-compatible.
            self.lsm_writer.top10_urban_natural_lcz_to_10 = (
                self.from_top10.urban_natural_lcz_to_10
            )
            self.lsm_writer.top10_urban_natural_lcz_to_natural_lsm = (
                self.from_top10.urban_natural_lcz_to_natural_lsm
            )
            apply_top10_to_lsm_writer(
                self.lsm_writer,
                top10_path=top10_path,
                fill_north_sea=self.from_top10.fill_north_sea,
            )

        # Optional BOFEK soil index mapping.
        if self.from_bofek is not None:
            bofek_nc_path = (
                pathlib.Path(self.from_bofek.spatial_data_path)
                / self.from_bofek.bofek_filename
            )
            bofek_csv_path = (
                pathlib.Path(self.from_bofek.spatial_data_path)
                / self.from_bofek.bofek_profile_csv
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
                self.lsm_writer,
                bofek_nc_path=bofek_nc_path,
                bofek_csv_path=bofek_csv_path,
            )
        # Apply soil temperature profiles, skin temperature, soil moisture profiles
        self.apply_soil_temp_moisture_skin_temp()

        # If LS2D-derived soil information is available, override the
        # default soil temperature / moisture / index with LS2D data.
        self._override_soil_from_ls2d_if_available()

        # Optional AGS setup fields (additive; does not alter existing fields).
        if self.ags_parameters is not None:
            apply_ags_parameters_to_lsm_writer(
                self.lsm_writer,
                grass_planttype=self.ags_parameters.grass_planttype,
            )

        # Finalize
        self.lsm_writer.trim_landuse()

        self.nlu = self.lsm_writer.nlu + 1

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

    def _apply_laqu_skin_temperature_override(self) -> None:
        if self.lsm_writer is None:
            return
        laqu_skin_temp = self._resolve_laqu_skin_temperature_override()
        if laqu_skin_temp is None:
            return
        self.lsm_writer.set_skin_temperature_laqu(laqu_skin_temp)

    def _override_soil_from_ls2d_if_available(self) -> None:
        """Apply provided soil state, roughness and skin temperature (opt-in via ``FromLS2D``).

        Requests ``t_soil``/``theta_soil`` (on ``zs``), ``type_soil``,
        ``z0m``/``z0h`` and the ``thls`` series from ``sim.forcings()``, as
        provided by e.g. :class:`LS2DAtmosphereModule`. Surface pressure is
        taken from the provided ``psurf`` through the ``ps`` field itself.
        """

        if self.lsm_writer is None or getattr(self, "from_ls2d", None) is None:
            return

        forcings = self.sim.forcings()
        nz_soil, jtot, itot = self.lsm_writer.value_dic["t_soil"].shape
        dz_soil = (
            np.asarray(self.dz_soil, dtype=float)
            if self.dz_soil is not None and len(self.dz_soil) == nz_soil
            else None
        )
        target_depths = None if dz_soil is None else np.cumsum(dz_soil) - 0.5 * dz_soil

        for name in (t_soil.name, theta_soil.name):
            provided = forcings.initial.get(name)
            if provided is not None:
                profile = _soil_profile_on_levels(provided, nz_soil, target_depths)
                self.lsm_writer.value_dic[name][:, :, :] = profile.values[
                    :, np.newaxis, np.newaxis
                ]

        provided_type = forcings.initial.get(type_soil.name)
        if provided_type is not None:
            values = np.asarray(provided_type.values)
            if values.shape == (jtot, itot):
                self.lsm_writer.value_dic["index_soil"][:, :, :] = values.astype(int)[
                    np.newaxis, :, :
                ]
            else:
                self.lsm_writer.value_dic["index_soil"][:, :, :] = int(
                    values.ravel()[0]
                )

        for name, attr in ((z0m.name, "z0mav"), (z0h.name, "z0hav")):
            if name in forcings.initial:
                setattr(self, attr, float(forcings.initial[name]))

        skin = forcings.series.get(thls.name)
        if skin is not None:
            self.lsm_writer.set_skin_temperature(
                float(value_at(skin, 0.0)), lu_type="all"
            )
            self._apply_laqu_skin_temperature_override()

        logger.info(
            "LSMModule: applied provided soil profiles, soil index, roughness and skin temperature where available"
        )

    def apply_soil_temp_moisture_skin_temp(self):
        if self.lsm_writer is None:
            raise ValueError(
                "Can't apply soil temperature/moisture/skin temp without initializing LSM writer first"
            )
        if isinstance(self.soil_moisture, (UniformSoilMoisture)):
            self.lsm_writer.set_uniform_soil_moisture(self.soil_moisture.soil_moisture)
        elif isinstance(self.soil_moisture, (VaryingSoilMoisture)):
            self.lsm_writer.set_soil_moisture_array(self.soil_moisture.soil_moisture)
        elif isinstance(self.soil_moisture, (SoilTemperatureMoistureFromHarmonie)):
            self.lsm_writer.set_soil_moisture_array(
                self.soil_moisture.get_soil_moisture_array(self.grid, self.dz_soil)
            )
        else:
            raise ValueError(
                "Invalid soil moisture configuration. Must be UniformSoilMoisture, VaryingSoilMoisture, or SoilTemperatureMoistureFromHarmonie."
            )

        if isinstance(
            self.soil_temperature,
            (UniformSoilTemperature),
        ):
            self.lsm_writer.set_uniform_soil_temperature(
                self.soil_temperature.soil_temperature
            )
        elif isinstance(
            self.soil_temperature,
            (VaryingSoilTemperature),
        ):
            self.lsm_writer.set_soil_temperature_array(
                self.soil_temperature.soil_temperature
            )
        elif isinstance(self.soil_temperature, (SoilTemperatureMoistureFromHarmonie)):
            self.lsm_writer.set_soil_temperature_array(
                self.soil_temperature.get_soil_temperature_array(
                    self.grid, self.dz_soil
                )
            )
        else:
            raise ValueError(
                "Invalid soil temperature configuration. Must be UniformSoilTemperature, VaryingSoilTemperature, or SoilTemperatureMoistureFromHarmonie."
            )

        if isinstance(self.skin_temperature, (UniformSkinTemperature)):
            self.lsm_writer.set_skin_temperature(
                self.skin_temperature.skin_temperature, lu_type="all"
            )
        elif isinstance(self.skin_temperature, (VaryingSkinTemperature)):
            self.lsm_writer.set_skin_temperature_array(
                self.skin_temperature.skin_temperature, lu_type="all"
            )
        elif isinstance(self.skin_temperature, (SoilTemperatureMoistureFromHarmonie)):
            self.lsm_writer.set_skin_temperature_array(
                self.skin_temperature.get_tskin_array(),
                lu_type="all",
            )
        else:
            raise ValueError(
                "Invalid skin temperature configuration. Must be UniformSkinTemperature or VaryingSkinTemperature."
            )

        self._apply_laqu_skin_temperature_override()

    def write_files(self):
        """Write LSM input files and generate plots."""

        # Re-apply LS2D-derived overrides at write time. This guarantees
        # values such as ps/z0 are available even when LSM prepare happened
        # before LS2D prepare due to module ordering.
        self._override_soil_from_ls2d_if_available()
        self.apply_namelist_from_fields()

        self.lsm_writer.save_netcdf(self.output_path / "input", self.exp_id)
        plot_lsm.plot_lsm_cover(
            pathlib.Path(self.output_path) / "input" / f"lsm.inp_{self.exp_id:03d}.nc",
            pathlib.Path(self.output_path) / "profiles",
        )
