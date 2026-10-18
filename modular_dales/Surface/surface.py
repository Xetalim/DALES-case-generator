import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from modular_dales.modular.namelist import nml_field
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

if TYPE_CHECKING:
    from modular_dales.modular.time_dependent_scalars import TimeDependentScalar

logger = logging.getLogger(__name__)


@register_module
class SurfaceModule(simulation_module):
    """Surface simulation module base class."""

    def _initialize_from_sim(self, sim):
        sim.has_surface_module = True
        return super()._initialize_from_sim(sim)


@register_module
@dataclass
class ConstantFluxesModule(SurfaceModule):
    """Constant heat and moisture fluxes without shear stress (isurf=4).

    When added to a simulation, automatically enables isurf=4 in the namelist.
    Requires surface config to contain 'wtsurf', 'wqsurf', and 'ps' parameters.

    Args:
      sim: Parent simulation instance
      isurf: Surface scheme selector (4 for constant fluxes)
      wtsurf: Heat flux (K m/s)
      wqsurf: Moisture flux (kg/kg m/s)
      ps: Surface pressure (Pa)
      z0mav: Momentum roughness length (m)
      z0hav: Heat roughness length (m)
      albedoav: Surface albedo (dimensionless)
    """

    isurf: int = nml_field(default=4, init=False, section="NAMSURFACE", key="isurf")
    wtsurf: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="wtsurf",
        required=True,
        metadata={"serialize": True, "forcing_var": "wtsurf"},
    )
    wqsurf: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="wqsurf",
        required=True,
        metadata={"serialize": True, "forcing_var": "wqsurf"},
    )
    ps: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="ps",
        required=True,
        metadata={"serialize": True, "forcing_var": "psurf"},
    )
    z0mav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0mav",
        required=True,
        metadata={"serialize": True},
    )
    z0hav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0hav",
        required=True,
        metadata={"serialize": True},
    )
    albedoav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="albedoav",
        metadata={"serialize": True},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "ConstantFluxesModule"

    def do_config(self):
        """Configure surface settings and namelist."""
        return

    def prepare_calculation(self):
        return self.prepare_calculations()

    def prepare_calculations(self):
        """No additional preparation needed."""
        return

    def check_settings(self):
        """Validate constant fluxes settings."""
        return

    def write_files(self):
        """No files to write."""
        return


@register_module
@dataclass
class ConstantFluxesWithShearModule(SurfaceModule):
    """Constant heat and moisture fluxes with shear stress (isurf=3).

    When added to a simulation, automatically enables isurf=3 in the namelist.
    Requires surface config to contain 'wtsurf', 'wqsurf', and 'ustin' parameters.
    Args:
      sim: Parent simulation instance
      isurf: Surface scheme selector (3 for constant fluxes with shear)
      wtsurf: Heat flux (K m/s)
      wqsurf: Moisture flux (kg/kg m/s)
      ustin: Friction velocity (m/s)
      ps: Surface pressure (Pa)
      z0mav: Momentum roughness length (m)
      z0hav: Heat roughness length (m)
      albedoav: Surface albedo (dimensionless)
    """

    isurf: int = nml_field(default=3, init=False, section="NAMSURFACE", key="isurf")
    wtsurf: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="wtsurf",
        required=True,
        metadata={"serialize": True, "forcing_var": "wtsurf"},
    )
    wqsurf: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="wqsurf",
        required=True,
        metadata={"serialize": True, "forcing_var": "wqsurf"},
    )
    ustin: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="ustin",
        required=True,
        metadata={"serialize": True},
    )
    ps: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="ps",
        required=True,
        metadata={"serialize": True, "forcing_var": "psurf"},
    )
    z0mav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0mav",
        required=True,
        metadata={"serialize": True},
    )
    z0hav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0hav",
        required=True,
        metadata={"serialize": True},
    )
    albedoav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="albedoav",
        metadata={"serialize": True},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "ConstantFluxesWithShearModule"

    def do_config(self):
        """Configure surface settings and namelist."""
        return

    def prepare_calculation(self):
        return self.prepare_calculations()

    def prepare_calculations(self):
        """No additional preparation needed."""
        return

    def check_settings(self):
        """Validate constant fluxes with shear settings."""
        return

    def write_files(self):
        """No files to write."""
        return


@register_module
@dataclass
class ConstantSurfaceTemperatureModule(SurfaceModule):
    """Constant surface temperature boundary condition (isurf=2).

    When added to a simulation, automatically enables isurf=2 in the namelist.
    Expects surface config to contain 'thls' parameter.

    Args:
      sim: Parent simulation instance
      isurf: Surface scheme selector (2 for constant surface temperature)
      thls: Surface liquid water potential temperature [K]
      z0mav: Aerodynamic roughness length for momentum [m]
      z0hav: Aerodynamic roughness length for heat and moisture [m]
      ps: Surface pressure (Pa)
      albedoav: Surface albedo (dimensionless)
    """

    isurf: int = nml_field(default=2, init=False, section="NAMSURFACE", key="isurf")
    thls: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="thls",
        required=True,
        metadata={"serialize": True, "forcing_var": "thls"},
    )
    z0mav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0mav",
        required=True,
        metadata={"serialize": True},
    )
    z0hav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0hav",
        required=True,
        metadata={"serialize": True},
    )
    ps: Union[float, "TimeDependentScalar"] | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="ps",
        required=True,
        metadata={"serialize": True, "forcing_var": "psurf"},
    )
    albedoav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="albedoav",
        metadata={"serialize": True},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "ConstantSurfaceTemperatureModule"

    def do_config(self):
        """Configure surface settings and namelist."""
        return

    def prepare_calculation(self):
        return self.prepare_calculations()

    def prepare_calculations(self):
        """No additional preparation needed."""
        return

    def check_settings(self):
        """Validate constant surface temperature settings."""
        return

    def write_files(self):
        """No files to write."""
        return
