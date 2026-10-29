import logging
from dataclasses import dataclass, field

from moda.Core.namelist import nml_field
from moda.modular.simulation_module import simulation_module
from moda.module_registry import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class RunModule(simulation_module):
    """Run configuration module for DALES simulation.

    Handles all run-related namelist parameters in the RUN section.

    Args:
        sim: Parent dales_simulation instance
        iexpnr: Experiment number
        runtime: Total simulation time in seconds
        ladaptive: Allow adaptive timestepping (highly recommended)
        dtmax: Maximum timestep
    """

    iexpnr: int | None = nml_field(default=None, section="RUN", key="iexpnr")
    runtime: float | None = nml_field(default=None, section="RUN", key="runtime")
    ladaptive: bool | None = nml_field(default=None, section="RUN", key="ladaptive")
    dtmax: float | None = nml_field(default=None, section="RUN", key="dtmax")

    module_name: str = field(
        default="RunModule", init=False, metadata={"serialize": False}
    )
