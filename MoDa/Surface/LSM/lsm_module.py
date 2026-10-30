import logging
import pathlib
from dataclasses import dataclass, field, fields

from moda.Core.namelist import get_namelist_spec, nml_field
from moda.module_registry import register_module
from moda.Surface.LSM import lsm_output, plot_lsm
from moda.Surface.LSM.lsm_output import LSM_output_dales

from .base import BaseLSMModule
from .lsm_sources.lsm_sources import (
    LSMSourceSpec,
    _PrioritizedSource,
)
from .lsm_sources.context import LSMSourceContext
from .lsm_sources.slurb_modifications import slbCreatorClass
from .lsm_sources.top10_bofek_ags import (
    apply_ags_parameters_to_lsm_writer,
)
from .translation_tables.landuse_types import LandUseSet

logger = logging.getLogger(__name__)


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
        include_slurb: Generate SLURB morphology and output (default False).
            NAMSLURB settings are configured directly on this module.
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
    include_slurb: bool = field(default=False, metadata={"serialize": True})
    slb_generator: slbCreatorClass | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    urban_fraction: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Urban tile fraction used by the SLURB parameterization.",
        metadata={"serialize": True},
    )
    urban_roughness_length: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Bulk roughness length for urban canopy momentum exchange in m.",
        metadata={"serialize": True},
    )
    building_plan_area_fraction: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Plan area fraction covered by buildings.",
        metadata={"serialize": True},
    )
    building_frontal_area_fraction: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Frontal area density controlling urban drag.",
        metadata={"serialize": True},
    )
    building_height: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Representative building height in m.",
        metadata={"serialize": True},
    )
    window_fraction: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Fraction of facade area occupied by windows.",
        metadata={"serialize": True},
    )
    street_canyon_aspect_ratio: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Street canyon height-to-width ratio.",
        metadata={"serialize": True},
    )
    building_type: int | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Index selecting building material/thermal parameter set.",
        metadata={"serialize": True},
    )
    pavement_type: int | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Index selecting pavement material parameter set.",
        metadata={"serialize": True},
    )
    anisotropic_street_canyons: bool | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Enable anisotropic canyon treatment with separate facade directions.",
        metadata={"serialize": True},
    )
    street_canyon_orientation: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Azimuth orientation of street canyon axis in degrees.",
        metadata={"serialize": True},
    )
    deep_soil_temperature: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Deep soil temperature in K; defaults to 283.15 when SLURB is enabled.",
        metadata={"serialize": True},
    )
    building_indoor_temperature: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="Indoor building temperature in K; defaults to 293.15 when SLURB is enabled.",
        metadata={"serialize": True},
    )
    shf_external: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="External anthropogenic sensible heat flux source in W m-2.",
        metadata={"serialize": True},
    )
    qsws_external: float | None = nml_field(
        default=None,
        section="NAMSLURB",
        description="External anthropogenic latent heat flux source in W m-2.",
        metadata={"serialize": True},
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
        writer = self._create_lsm_writer()
        self.lsm_writer = writer
        self.slb_generator = slbCreatorClass(self.grid) if self.include_slurb else None
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
            include_slurb=self.include_slurb,
        )

    def _validate_sources_land_use_set(self) -> None:
        """Require an explicit catalogue and reject incompatible land-use sources."""
        self._validate_slurb_settings()
        if self.land_use_set is None:
            raise ValueError("Choose a land_use_set before preparing LSMModule")
        self.land_use_set = LandUseSet(self.land_use_set)
        for entry in self.sources:
            if not isinstance(entry, _PrioritizedSource):
                raise TypeError(
                    "Use LSMModule.add(source, priority=...) to add sources"
                )
            entry.validate()
            if (
                getattr(entry.source, "requires_slurb", False)
                and not self.include_slurb
            ):
                raise ValueError(
                    "SLURB modification sources require include_slurb=True"
                )
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
        context = LSMSourceContext(self, writer)
        for entry in sorted(self.sources, key=lambda source: source.priority):
            entry.source.apply_source(context)

    def _init_slurb_generator(self) -> slbCreatorClass | None:
        """Initialize source-driven morphology only when SLURB is enabled."""
        if not self.include_slurb:
            return None
        if self.slb_generator is None:
            self.slb_generator = slbCreatorClass(self.grid)
        return self.slb_generator

    def _validate_slurb_settings(self) -> None:
        """Reject SLURB-only settings when its output is disabled."""
        if not isinstance(self.include_slurb, bool):
            raise TypeError("include_slurb must be a boolean")
        if self.include_slurb:
            return
        for definition in fields(self):
            spec = get_namelist_spec(definition)
            if (
                spec is not None
                and spec.section == "NAMSLURB"
                and getattr(self, definition.name) is not None
            ):
                raise ValueError(f"{definition.name} requires include_slurb=True")

    def apply_namelist_from_fields(self) -> None:
        """Emit SLURB defaults only when the urban parameterization is enabled."""
        self._validate_slurb_settings()
        if self.include_slurb:
            if self.deep_soil_temperature is None:
                raise ValueError(
                    "deep_soil_temperature must be specified when include_slurb is True"
                )
            if self.building_indoor_temperature is None:
                raise ValueError(
                    "building_indoor_temperature must be specified when include_slurb is True"
                )
        super().apply_namelist_from_fields()

    def write_files(self) -> None:
        """Write LSM input files and generate plots."""
        if self.lsm_writer is None:
            raise ValueError("Prepare the LSM before writing files")
        if self.include_slurb and self.slb_generator is None:
            raise ValueError("Prepare SLURB before writing files")
        self.apply_namelist_from_fields()

        lsm_output.save_netcdf(self.lsm_writer, self.output_path / "input", self.exp_id)
        if self.include_slurb and self.slb_generator is not None:
            self.slb_generator.output_nc(
                self.output_path / "input" / f"inslurb.{self.exp_id:03d}.nc"
            )
        plot_lsm.plot_lsm_cover(
            pathlib.Path(self.output_path) / "input" / f"lsm.inp_{self.exp_id:03d}.nc",
            pathlib.Path(self.output_path) / "profiles",
        )
