import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from modular_dales.modular.dales_simulation import dales_simulation
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

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

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    iexpnr: int | None = field(default=None, metadata={"nml": "RUN", "key": "iexpnr"})
    runtime: float | None = field(
        default=None, metadata={"nml": "RUN", "key": "runtime"}
    )
    ladaptive: bool | None = field(
        default=None, metadata={"nml": "RUN", "key": "ladaptive"}
    )
    dtmax: float | None = field(default=None, metadata={"nml": "RUN", "key": "dtmax"})

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "RunModule"

    def prepare_calculation(self):
        """Set up run-related namelist parameters."""
        return

    def check_settings(self):
        """Check run settings validity."""
        return

    def write_files(self):
        """No files to write for run module."""
        return


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

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    xtime: float = field(
        default=0.0,
        metadata={
            "nml": "DOMAIN",
            "key": "xtime",
            "required": True,
            "doc": "UTC start time in hours within the day for DOMAIN:xtime.",
        },
    )
    xday: int = field(
        default=265,
        metadata={
            "nml": "DOMAIN",
            "key": "xday",
            "required": True,
            "doc": "Day of year (1-365/366) used by DOMAIN:xday.",
        },
    )
    xyear: int = field(
        default=2025,
        metadata={
            "nml": "DOMAIN",
            "key": "xyear",
            "required": True,
            "doc": "Simulation year used by DOMAIN:xyear.",
        },
    )
    enable_datetime: bool = field(
        default=True,
        metadata={
            "nml": "namdatetime",
            "key": "l_datetime",
            "required": True,
            "doc": "Enable NAMDATETIME-based calendar interpretation.",
        },
    )
    timezone: int = field(
        default=0,
        metadata={
            "nml": "namdatetime",
            "key": "timezone",
            "doc": "Timezone offset in hours for NAMDATETIME.",
        },
    )
    startyear: int = field(
        default=2025,
        metadata={
            "nml": "namdatetime",
            "key": "startyear",
            "required": True,
            "doc": "Calendar start year for NAMDATETIME.",
        },
    )
    startmonth: int = field(
        default=1,
        metadata={
            "nml": "namdatetime",
            "key": "startmonth",
            "required": True,
            "doc": "Calendar start month for NAMDATETIME.",
        },
    )
    startday: int = field(
        default=1,
        metadata={
            "nml": "namdatetime",
            "key": "startday",
            "required": True,
            "doc": "Calendar start day for NAMDATETIME.",
        },
    )
    inferfromdatetime: bool = field(
        default=False,
        metadata={
            "serialize": True,
            "doc": "If true, infer DOMAIN:xday and DOMAIN:xyear from NAMDATETIME start date.",
        },
    )

    runtime: float | None = field(
        default=None,
        metadata={
            "nml": "RUN",
            "key": "runtime",
            "required": True,
            "doc": "Total runtime in seconds for RUN:runtime.",
        },
    )
    trestart: float | None = field(
        default=None,
        metadata={
            "nml": "RUN",
            "key": "trestart",
            "doc": "Restart write interval in seconds for RUN:trestart.",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "TimeModule"

    def do_config(self):
        """Configure time-related namelist parameters."""
        return

    def _apply_datetime_inference(self) -> None:
        """Infer DOMAIN xyear/xday from NAMDATETIME start date when enabled."""
        if not self.inferfromdatetime:
            return

        start_dt = date(self.startyear, self.startmonth, self.startday)
        self.xyear = int(start_dt.year)
        self.xday = int(start_dt.timetuple().tm_yday)

    def check_settings(self):
        """Check time settings validity."""
        self._apply_datetime_inference()
        if self.xyear != self.startyear:
            logger.warning(
                "TimeModule: xyear (%d) does not match startyear (%d)",
                self.xyear,
                self.startyear,
            )

    def prepare_calculation(self):
        """No calculation work needed."""
        return

    def write_files(self):
        """No files to write."""
        return
