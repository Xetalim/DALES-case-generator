import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast

import ls2d
import numpy as np
import xarray as xr

from moda.Atmosphere.external_forcing import ExternalForcingModule
from moda.Atmosphere.les_input import LESInput
from moda.module_registry import register_module
from moda.modular.time_dependent_scalars import SOIL_DIM, value_at
from moda.vars import t_soil, theta_soil, thls, type_soil, z0h, z0m

if TYPE_CHECKING:
    from moda.Surface.LSM.lsm_sources.context import LSMSourceContext

logger = logging.getLogger(__name__)


def _soil_profile_on_levels(
    profile: xr.DataArray, nz_soil: int, target_depths: np.ndarray | None
) -> xr.DataArray:
    """Interpolate a soil profile by depth or relative level index."""
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


def retry_exponential(
    func: Callable[[], bool],
    *,
    max_attempts: int = 5,
    max_total_time: float = 60.0,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> None:
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


@register_module
@dataclass
class FromLS2D:
    """Marker class to enable LS2D-driven soil/roughness in LSM.

    Add it inside ``LSMSource`` with an explicit priority. It supports every
    land-use set because it only provides soil state and surface roughness.
    With an :class:`LS2DAtmosphereModule` present, soil temperature/moisture,
    soil type index and bulk roughness lengths ``z0mav``/``z0hav`` are taken
    from LS2D.
    """

    SUPPORTED_LAND_USE_SETS = frozenset({"basic", "build", "crop", "depac", "ifs"})

    def apply_source(self, context: "LSMSourceContext") -> None:
        module, writer = context.module, context.writer
        forcings = module.sim.forcings()
        soil = writer.soil
        nz_soil, jtot, itot = soil.t_soil.shape
        dz_soil = (
            np.asarray(module.dz_soil, dtype=float)
            if module.dz_soil is not None and len(module.dz_soil) == nz_soil
            else None
        )
        target_depths = None if dz_soil is None else np.cumsum(dz_soil) - 0.5 * dz_soil
        for name, target in (
            (t_soil.name, soil.t_soil),
            (theta_soil.name, soil.theta_soil),
        ):
            provided = forcings.initial.get(name)
            if provided is not None:
                profile = _soil_profile_on_levels(provided, nz_soil, target_depths)
                target[:, :, :] = profile.values[:, np.newaxis, np.newaxis]
        provided_type = forcings.initial.get(type_soil.name)
        if provided_type is not None:
            values = np.asarray(provided_type.values)
            if values.shape == (jtot, itot):
                soil.index_soil[:, :, :] = values.astype(int)[np.newaxis, :, :]
            else:
                soil.index_soil[:, :, :] = int(values.ravel()[0])
        for name, attr in ((z0m.name, "z0mav"), (z0h.name, "z0hav")):
            if name in forcings.initial:
                setattr(module, attr, float(forcings.initial[name]))
        skin = forcings.series.get(thls.name)
        if skin is not None:
            writer.set_skin_temperature(float(value_at(skin, 0.0)), lu_type="all")
            context.set_aquatic_skin_temperature()
        logger.info(
            "LSM: applied provided soil profiles, soil index, roughness "
            "and skin temperature where available"
        )


@register_module
@dataclass
class LS2DAtmosphereModule(ExternalForcingModule):
    """Atmosphere forcing from ERA5, processed by LS2D.

    Downloads ERA5 data, runs ``era.calculate_forcings`` and interpolates the
    result onto ``grid.zt``. Initial profiles, nudging targets and large-scale
    forcings are provided to :class:`AtmosphereModule` and collected by the
    simulation for time-dependent output; see :class:`ExternalForcingModule`.
    """

    module_name: str = field(
        default="LS2DAtmosphereModule", init=False, metadata={"serialize": False}
    )

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
    data_source: Literal["CDS", "ARCO"] = field(
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

    def check_settings(self) -> None:
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
        if self.start_date is None or self.end_date is None:
            raise ValueError("Start and end dates must be set")
        ls2d.disable_default_logging()
        if self.data_source == "CDS":
            settings = {
                "central_lat": self.central_lat,
                "central_lon": self.central_lon,
                "area_size": self.area_size,
                "case_name": self.case_name,
                "era5_path": self.era5_path,
                "start_date": self.start_date.replace(
                    tzinfo=None
                ),  # while LS2D expects naive datetime objects
                "end_date": self.end_date.replace(
                    tzinfo=None
                ),  # while LS2D expects naive datetime objects
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
            return cast(LESInput, era.get_les_input(self.zt))
        elif self.data_source == "ARCO":
            settings = {
                "central_lat": self.central_lat,
                "central_lon": self.central_lon,
                "area_size": self.area_size,
                "case_name": self.case_name,
                "era5_path": self.era5_path,
                "start_date": self.start_date.replace(tzinfo=None),
                "end_date": self.end_date.replace(tzinfo=None),
                "write_log": self.write_log,
            }
            logger.info("LS2DAtmosphereModule: reading local ERA5 via LS2D.Read_era5")
            ls2d.download_era5_arco(settings)
            ds_3d = ls2d.read_era5_arco(settings)
            ds_1d = ls2d.create_column_input(ds_3d, self.zt, n_av=self.n_av)
            return cast(LESInput, ds_1d)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
