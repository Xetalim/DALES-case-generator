"""Additional non-output settings modules for DALES namelist configuration."""

from dataclasses import dataclass, field

from moda.Configuration.output_modules import prepare_point_indices
from moda.Core.namelist import nml_field
from moda.modular.simulation_module import simulation_module
from moda.module_registry import register_module


@register_module
@dataclass
class SprayingModule(simulation_module):
    """Settings module for sea-spray and water-spray source terms.

    Supports both index-based and coordinate-based source placement.
    If x_spray/y_spray/z_spray are all provided, coordinates take precedence and
    are mapped to nearest grid centers. If no coordinates are provided, the
    configured i_glob_spray/j_glob_spray/k_glob_spray indices are used.
    """

    x_idx: int | None = nml_field(
        default=None,
        section="namspraying",
        key="i_glob_spray",
        description="Global i-index location for spray source.",
    )
    y_idx: int | None = nml_field(
        default=None,
        section="namspraying",
        key="j_glob_spray",
        description="Global j-index location for spray source.",
    )
    z_idx: int | None = nml_field(
        default=None,
        section="namspraying",
        key="k_glob_spray",
        description="Vertical level index for spray source.",
    )
    x: float | None = field(default=None, metadata={"serialize": False})
    y: float | None = field(default=None, metadata={"serialize": False})
    z: float | None = field(default=None, metadata={"serialize": False})
    lwater_spraying: bool = nml_field(
        default=False,
        section="namspraying",
        key="lwater_spraying",
        description="Enable prescribed water spraying source.",
    )
    lsalt_spraying: bool = nml_field(
        default=False,
        section="namspraying",
        key="lsalt_spraying",
        description="Enable prescribed salt spraying source.",
    )
    water_spray_rate: float = nml_field(
        default=0.0,
        section="namspraying",
        key="water_spray_rate",
        description="Injected water spray mass rate.",
    )
    salt_spray_rate: float = nml_field(
        default=0.0,
        section="namspraying",
        key="salt_spray_rate",
        description="Injected salt spray mass rate.",
    )
    salinity: float = nml_field(
        default=0.0,
        section="namspraying",
        key="salinity",
        description="Salt mass fraction used for water spray.",
    )
    tracer: str = nml_field(
        default="salt",
        section="namspraying",
        key="tracer",
        description="Tracer name that receives spray source.",
    )
    lsalt_sponge: bool = nml_field(
        default=False,
        section="namspraying",
        key="lsalt_sponge",
        description="Enable salt sponge damping around spray source.",
    )
    lcoupled: bool = nml_field(
        default=False,
        section="namspraying",
        key="lcoupled",
        description="Enable coupling with external spray forcing.",
    )
    target_mode: int = nml_field(
        default=0,
        section="namspraying",
        key="target_mode",
        description="Spray forcing mode selector.",
    )

    module_name: str = field(
        default="SprayingModule", init=False, metadata={"serialize": False}
    )

    def prepare_calculation(self) -> None:
        self.x_idx, self.y_idx, self.z_idx = prepare_point_indices(
            "SprayingModule",
            self.grid,
            allow_idx_list=False,
            x_idx=self.x_idx,
            y_idx=self.y_idx,
            z_idx=self.z_idx,
            x=self.x,
            y=self.y,
            z=self.z,
        )


@register_module
@dataclass
class LateralSpongeModule(simulation_module):
    """Settings module for lateral sponge damping configuration."""

    llateral_sponge: bool = nml_field(
        default=False,
        section="lateral_sponge",
        key="llateral_sponge",
        description="Enable lateral sponge damping near side boundaries.",
    )
    nudgedepth: float = nml_field(
        default=0.0,
        section="lateral_sponge",
        key="nudgedepth",
        description="Depth/width scale of lateral sponge region in m.",
    )

    module_name: str = field(
        default="LateralSpongeModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class GeneralPhysicsModule(simulation_module):
    """Settings module combining thermodynamics and subgrid controls."""

    lmoist: bool | None = nml_field(
        default=None,
        section="thermodynamics",
        key="lmoist",
        description="Enable moist thermodynamics.",
    )
    chi_half: bool | None = nml_field(
        default=None,
        section="thermodynamics",
        key="chi_half",
        description="Use half-level chi/exner treatment where applicable.",
    )
    # lconstexner: bool | None = nml_field(
    #     default=None,
    #     section="thermodynamics",
    #     key="lconstexner",
    #     description="Use initial pressure profile in exner function.",
    # )
    # lbaseexner: bool | None = nml_field(
    #     default=None,
    #     section="thermodynamics",
    #     key="lbaseexner",
    #     description="Use base-state pressure profile in exner function.",
    # )
    lnoclouds: bool | None = nml_field(
        default=None,
        section="thermodynamics",
        key="lnoclouds",
        description="Disable cloud liquid calculations in thermodynamics.",
    )
    lqlnr: bool | None = nml_field(
        default=None,
        section="thermodynamics",
        key="lqlnr",
        description="Toggle non-rain liquid handling option.",
    )

    ldelta: bool | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="ldelta",
        description="Use delta-based subgrid length scale.",
    )
    lmason: bool | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="lmason",
        description="Enable Mason wall-damping formulation.",
    )
    cf: float | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="cf",
        description="Subgrid closure constant cf.",
    )
    cn: float | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="cn",
        description="Subgrid closure constant cn.",
    )
    Rigc: float | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="Rigc",
        description="Critical Richardson number for subgrid closure.",
    )
    Prandtl: float | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="Prandtl",
        description="Subgrid turbulent Prandtl number.",
    )
    lsmagorinsky: bool | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="lsmagorinsky",
        description="Enable Smagorinsky-type closure.",
    )
    cs: float | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="cs",
        description="Smagorinsky coefficient.",
    )
    nmason: int | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="nmason",
        description="Selector for Mason damping variant.",
    )
    sgs_surface_fix: bool | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="sgs_surface_fix",
        description="Enable near-surface SGS correction.",
    )
    ch1: float | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="ch1",
        description="Empirical closure coefficient ch1.",
    )
    lanisotrop: bool | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="lanisotrop",
        description="Enable anisotropic diffusion treatment.",
    )
    lD80R: bool | None = nml_field(
        default=None,
        section="NAMSUBGRID",
        key="lD80R",
        description="Enable D80R subgrid option.",
    )

    module_name: str = field(
        default="GeneralPhysicsModule", init=False, metadata={"serialize": False}
    )
