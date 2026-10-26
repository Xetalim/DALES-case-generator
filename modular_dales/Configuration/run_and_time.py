import logging
from dataclasses import dataclass, field
from datetime import date

from modular_dales.modular.namelist import nml_field
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.module_registry import register_module

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


@register_module
@dataclass
class TimeModule(simulation_module):
    """Simulation time configuration module.

    Handles time-related parameters in the RUN and DOMAIN sections.

    Args:
        sim: Parent dales_simulation instance
        xtime: Time within the day in seconds (0-86400)
        xday: Day of year (1-365/366)
        xyear: Year
        runtime: Total runtime in seconds
        trestart: Restart file write interval in seconds
    """

    xtime: float = nml_field(
        default=0.0,
        section="DOMAIN",
        key="xtime",
        required=True,
        description="UTC start time in hours within the day for DOMAIN:xtime.",
    )
    xday: int = nml_field(
        default=265,
        section="DOMAIN",
        key="xday",
        required=True,
        description="Day of year (1-365/366) used by DOMAIN:xday.",
    )
    xyear: int = nml_field(
        default=2025,
        section="DOMAIN",
        key="xyear",
        required=True,
        description="Simulation year used by DOMAIN:xyear.",
    )
    enable_datetime: bool = nml_field(
        default=True,
        section="namdatetime",
        key="l_datetime",
        required=True,
        description="Enable NAMDATETIME-based calendar interpretation.",
    )
    timezone: int = nml_field(
        default=0,
        section="namdatetime",
        key="timezone",
        description="Timezone offset in hours for NAMDATETIME.",
    )
    startyear: int = nml_field(
        default=2025,
        section="namdatetime",
        key="startyear",
        required=True,
        description="Calendar start year for NAMDATETIME.",
    )
    startmonth: int = nml_field(
        default=1,
        section="namdatetime",
        key="startmonth",
        required=True,
        description="Calendar start month for NAMDATETIME.",
    )
    startday: int = nml_field(
        default=1,
        section="namdatetime",
        key="startday",
        required=True,
        description="Calendar start day for NAMDATETIME.",
    )
    inferfromdatetime: bool = field(
        default=False,
        metadata={
            "serialize": True,
            "doc": "If true, infer DOMAIN:xday and DOMAIN:xyear from NAMDATETIME start date.",
        },
    )

    runtime: float | None = nml_field(
        default=None,
        section="RUN",
        key="runtime",
        required=True,
        description="Total runtime in seconds for RUN:runtime.",
    )
    trestart: float | None = nml_field(
        default=None,
        section="RUN",
        key="trestart",
        description="Restart write interval in seconds for RUN:trestart.",
    )

    module_name: str = field(
        default="TimeModule", init=False, metadata={"serialize": False}
    )

    def _apply_datetime_inference(self) -> None:
        """Infer DOMAIN xyear/xday from NAMDATETIME start date when enabled."""
        if not self.inferfromdatetime:
            return

        start_dt = date(self.startyear, self.startmonth, self.startday)
        self.xyear = int(start_dt.year)
        self.xday = int(start_dt.timetuple().tm_yday)

    def check_settings(self) -> None:
        """Check time settings validity."""
        self._apply_datetime_inference()
        if self.xyear != self.startyear:
            logger.warning(
                "TimeModule: xyear (%d) does not match startyear (%d)",
                self.xyear,
                self.startyear,
            )
