"""Additional non-output settings modules for DALES namelist configuration."""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from modular_dales.Configuration.output_modules import _HorizontalPointOutputMixin
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.modular.dales_simulation import dales_simulation
from modular_dales.modular.simulation_module import simulation_module


@register_module
@dataclass
class SprayingModule(_HorizontalPointOutputMixin, simulation_module):
    """Settings module for sea-spray and water-spray source terms.

    Supports both index-based and coordinate-based source placement.
    If x_spray/y_spray/z_spray are all provided, coordinates take precedence and
    are mapped to nearest grid centers. If no coordinates are provided, the
    configured i_glob_spray/j_glob_spray/k_glob_spray indices are used.
    """

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    x_idx: Optional[Union[int, list[int]]] = field(
        default=None,
        metadata={
            "nml": "namspraying",
            "key": "i_glob_spray",
            "doc": "Global i-index location for spray source.",
        },
    )
    y_idx: Optional[Union[int, list[int]]] = field(
        default=None,
        metadata={
            "nml": "namspraying",
            "key": "j_glob_spray",
            "doc": "Global j-index location for spray source.",
        },
    )
    z_idx: Optional[Union[int, list[int]]] = field(
        default=None,
        metadata={
            "nml": "namspraying",
            "key": "k_glob_spray",
            "doc": "Vertical level index for spray source.",
        },
    )
    x: Optional[Union[float, list[float]]] = field(
        default=None, metadata={"serialize": False}
    )
    y: Optional[Union[float, list[float]]] = field(
        default=None, metadata={"serialize": False}
    )
    z: Optional[Union[float, list[float]]] = field(
        default=None, metadata={"serialize": False}
    )
    lwater_spraying: bool = field(
        default=False,
        metadata={
            "nml": "namspraying",
            "key": "lwater_spraying",
            "doc": "Enable prescribed water spraying source.",
        },
    )
    lsalt_spraying: bool = field(
        default=False,
        metadata={
            "nml": "namspraying",
            "key": "lsalt_spraying",
            "doc": "Enable prescribed salt spraying source.",
        },
    )
    water_spray_rate: float = field(
        default=0.0,
        metadata={
            "nml": "namspraying",
            "key": "water_spray_rate",
            "doc": "Injected water spray mass rate.",
        },
    )
    salt_spray_rate: float = field(
        default=0.0,
        metadata={
            "nml": "namspraying",
            "key": "salt_spray_rate",
            "doc": "Injected salt spray mass rate.",
        },
    )
    salinity: float = field(
        default=0.0,
        metadata={
            "nml": "namspraying",
            "key": "salinity",
            "doc": "Salt mass fraction used for water spray.",
        },
    )
    tracer: int = field(
        default=1,
        metadata={
            "nml": "namspraying",
            "key": "tracer",
            "doc": "Tracer index that receives spray source.",
        },
    )
    lsalt_sponge: bool = field(
        default=False,
        metadata={
            "nml": "namspraying",
            "key": "lsalt_sponge",
            "doc": "Enable salt sponge damping around spray source.",
        },
    )
    lcoupled: bool = field(
        default=False,
        metadata={
            "nml": "namspraying",
            "key": "lcoupled",
            "doc": "Enable coupling with external spray forcing.",
        },
    )
    target_mode: int = field(
        default=0,
        metadata={
            "nml": "namspraying",
            "key": "target_mode",
            "doc": "Spray forcing mode selector.",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "SprayingModule"

    def do_config(self):
        return None

    def prepare_calculation(self):
        self._prepare_point_indices("SprayingModule")
        return None

    def check_settings(self):
        return None

    def write_files(self):
        return None


@register_module
@dataclass
class LateralSpongeModule(simulation_module):
    """Settings module for lateral sponge damping configuration."""

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    llateral_sponge: bool = field(
        default=False,
        metadata={
            "nml": "lateral_sponge",
            "key": "llateral_sponge",
            "doc": "Enable lateral sponge damping near side boundaries.",
        },
    )
    nudgedepth: float = field(
        default=0.0,
        metadata={
            "nml": "lateral_sponge",
            "key": "nudgedepth",
            "doc": "Depth/width scale of lateral sponge region in m.",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "LateralSpongeModule"

    def do_config(self):
        return None

    def prepare_calculation(self):
        return None

    def check_settings(self):
        return None

    def write_files(self):
        return None


@register_module
@dataclass
class GeneralPhysicsModule(simulation_module):
    """Settings module combining thermodynamics and subgrid controls."""

    sim: Optional["dales_simulation"] = field(default=None, repr=False)

    lmoist: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "thermodynamics",
            "key": "lmoist",
            "doc": "Enable moist thermodynamics.",
        },
    )
    chi_half: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "thermodynamics",
            "key": "chi_half",
            "doc": "Use half-level chi/exner treatment where applicable.",
        },
    )
    lconstexner: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "thermodynamics",
            "key": "lconstexner",
            "doc": "Use initial pressure profile in exner function.",
        },
    )
    lbaseexner: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "thermodynamics",
            "key": "lbaseexner",
            "doc": "Use base-state pressure profile in exner function.",
        },
    )
    lnoclouds: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "thermodynamics",
            "key": "lnoclouds",
            "doc": "Disable cloud liquid calculations in thermodynamics.",
        },
    )
    lqlnr: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "thermodynamics",
            "key": "lqlnr",
            "doc": "Toggle non-rain liquid handling option.",
        },
    )

    ldelta: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "ldelta",
            "doc": "Use delta-based subgrid length scale.",
        },
    )
    lmason: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "lmason",
            "doc": "Enable Mason wall-damping formulation.",
        },
    )
    cf: Optional[float] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "cf",
            "doc": "Subgrid closure constant cf.",
        },
    )
    cn: Optional[float] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "cn",
            "doc": "Subgrid closure constant cn.",
        },
    )
    Rigc: Optional[float] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "Rigc",
            "doc": "Critical Richardson number for subgrid closure.",
        },
    )
    Prandtl: Optional[float] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "Prandtl",
            "doc": "Subgrid turbulent Prandtl number.",
        },
    )
    lsmagorinsky: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "lsmagorinsky",
            "doc": "Enable Smagorinsky-type closure.",
        },
    )
    cs: Optional[float] = field(
        default=None,
        metadata={"nml": "NAMSUBGRID", "key": "cs", "doc": "Smagorinsky coefficient."},
    )
    nmason: Optional[int] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "nmason",
            "doc": "Selector for Mason damping variant.",
        },
    )
    sgs_surface_fix: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "sgs_surface_fix",
            "doc": "Enable near-surface SGS correction.",
        },
    )
    ch1: Optional[float] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "ch1",
            "doc": "Empirical closure coefficient ch1.",
        },
    )
    lanisotrop: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "lanisotrop",
            "doc": "Enable anisotropic diffusion treatment.",
        },
    )
    lD80R: Optional[bool] = field(
        default=None,
        metadata={
            "nml": "NAMSUBGRID",
            "key": "lD80R",
            "doc": "Enable D80R subgrid option.",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "GeneralPhysicsModule"

    def do_config(self):
        return None

    def prepare_calculation(self):
        return None

    def check_settings(self):
        return None

    def write_files(self):
        return None
