"""Core simulation API and domain namespaces.

Use ``import moda`` for core modules and ``moda.Surface``,
``moda.Atmosphere``, or the other domain namespaces for specialist tools.
Historical top-level class attributes remain available for compatibility;
``__all__`` advertises the core API and domain namespaces.
"""

from . import IBM, LBC, Atmosphere, Configuration, Emission, Radiation, Surface, vars
from .Geometry import GridDales
from .logging_wrapper import setup_logging
from .modular.dales_simulation import (
    DefaultNamelistModule,
    TimeModule,
    dales_simulation,
)
from .modular.simulation_module import set_nml_section, simulation_module
from .modular.time_dependent_scalars import TimeDependentScalar
from .module_registry import (
    module_registry,
    register_module,
)

setup_logging()

__all__ = [
    "IBM",
    "LBC",
    "Atmosphere",
    "Configuration",
    "DefaultNamelistModule",
    "Emission",
    "GridDales",
    "Radiation",
    "Surface",
    "TimeDependentScalar",
    "TimeModule",
    "dales_simulation",
    "module_registry",
    "register_module",
    "set_nml_section",
    "setup_logging",
    "simulation_module",
    "vars",
]
