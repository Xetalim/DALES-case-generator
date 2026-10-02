import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import ls2d

from modular_dales.Atmosphere.external_forcing import ExternalForcingModule
from modular_dales.Atmosphere.les_input import LESInput
from modular_dales.MODULE_REGISTRY import register_module, register_singleton

logger = logging.getLogger(__name__)


def retry_exponential(
    func: Callable,
    *,
    max_attempts: int = 5,
    max_total_time: float = 60.0,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to execute
        max_attempts: Maximum number of attempts
        max_total_time: Maximum total retry time (seconds)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    """
    start_time = time.monotonic()
    attempt = 0

    while True:
        attempt += 1
        if func():
            return
        else:
            elapsed = time.monotonic() - start_time

            if attempt >= max_attempts:
                logger.error(
                    "Max attempts reached (%s). Giving up download.",
                    max_attempts,
                )
                raise RuntimeError("Max download attempts reached. Giving up download.")

            if elapsed >= max_total_time:
                logger.error(
                    "Max total retry time exceeded (%.2fs). Giving up download.",
                    max_total_time,
                )
                raise RuntimeError("Max total retry time exceeded. Giving up download.")

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            logger.warning(
                "Download attempt %s failed. (data not yet ready) Retrying in %.2fs...",
                attempt,
                delay,
            )

            time.sleep(delay)


@register_singleton
@register_module
@dataclass
class FromLS2D:
    """Marker class to enable LS2D-driven soil/roughness in LSM.

    When added to :class:`LSMModule` (``lsm += FromLS2D()``) and an
    :class:`LS2DAtmosphereModule` is present, soil temperature/moisture,
    soil type index and the bulk roughness lengths ``z0mav``/``z0hav`` are
    taken from LS2D.
    """


@register_module
@dataclass
class LS2DAtmosphereModule(ExternalForcingModule):
    """Atmosphere forcing from ERA5, processed by LS2D.

    Downloads ERA5 data, runs ``era.calculate_forcings`` and interpolates the
    result onto ``grid.zt``. Initial profiles, nudging targets and large-scale
    forcings are provided to :class:`AtmosphereModule` and
    :class:`TimedependentModule`; see :class:`ExternalForcingModule`.
    """

    area_size: float = field(
        default=1.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    case_name: str = field(
        default="ls2d_case",
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    era5_path: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    start_date: datetime | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    end_date: datetime | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    write_log: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    data_source: str = field(
        default="CDS",
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    n_av: int = field(
        default=0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    method: str = field(
        default="2nd",
        init=True,
        repr=True,
        metadata={"serialize": True},
    )

    def check_settings(self):
        """Basic validation of configuration before running LS2D."""

        super().check_settings()
        if (not self.case_name) and getattr(self.sim, "case_name", None):
            self.case_name = str(self.sim.case_name)

        missing: list[str] = [
            name
            for name in (
                "central_lat",
                "central_lon",
                "era5_path",
                "start_date",
                "end_date",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "LS2DAtmosphereModule missing required settings: " + ", ".join(missing)
            )

    def load_les_input(self) -> LESInput:

        settings = {
            "central_lat": self.central_lat,
            "central_lon": self.central_lon,
            "area_size": self.area_size,
            "case_name": self.case_name,
            "era5_path": self.era5_path,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "write_log": self.write_log,
            "data_source": self.data_source,
        }

        logger.info("LS2DAtmosphereModule: downloading ERA5 via LS2D")
        retry_exponential(
            lambda: ls2d.download_era5(settings, exit_when_waiting=False),
            max_attempts=30,
            max_total_time=3600,
            base_delay=60,
            max_delay=180,
        )

        logger.info("LS2DAtmosphereModule: reading ERA5 via LS2D.Read_era5")
        era = ls2d.Read_era5(settings)
        era.calculate_forcings(n_av=self.n_av, method=self.method)

        logger.info(
            "LS2DAtmosphereModule: building les_input on DALES grid (k=%d)",
            self.zt.size,
        )
        return era.get_les_input(self.zt)
