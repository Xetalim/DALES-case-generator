from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr

from modular_dales.Configuration import TimeModule
from modular_dales.IO_helpers.atmosphere_writer import AtmosphereProfileWriter
from modular_dales.modular.forcing import var_attrs
from modular_dales.modular.time_dependent_scalars import (
    TIME_DIM,
    Z_DIM,
    resolve_time_axis,
)
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.vars import get_var_by_name

from .simulation_module import simulation_module

if TYPE_CHECKING:
    from . import dales_simulation


@register_module
@dataclass
class TimedependentModule(simulation_module):
    """Collects all time-dependent forcings and writes ``forcings.<exp_id>.nc``.

    Every module in the simulation can provide time series (e.g.
    ``TimeDependentScalar`` parameters, ``TimedAtmosphereProfile`` entries,
    ``TimeDependentScalar`` surface fields or LS2D/Harmonie forcings). Each
    series may use its own time points; all are linearly interpolated onto one
    common axis:

    * ``timesteps`` given: that axis is used.
    * ``timesteps`` empty (default): the union of all provided time points.

    Time 0 and the end of the run are always part of the axis.
    """

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    timesteps: list[float] = field(default_factory=list)
    timedep: xr.Dataset = field(
        default_factory=xr.Dataset,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    """Forcings on the common time axis, named as in ``forcings.<exp_id>.nc``."""
    profile_writer: AtmosphereProfileWriter = field(
        default_factory=AtmosphereProfileWriter,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    ntimedep: int = field(
        default=0,
        init=False,
        repr=False,
        metadata={
            "serialize": False,
            "nml": "PHYSICS",
            "key": "ntimedep",
        },
    )
    ltimedep: bool = field(
        default=True,
        init=True,
        repr=False,
        metadata={
            "serialize": False,
            "nml": "PHYSICS",
            "key": "ltimedep",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        if isinstance(self.timesteps, np.ndarray):
            self.timesteps = self.timesteps.tolist()
        self.module_name = "TimedependentModule"

    def check_settings(self):
        return None

    def prepare_calculation(self):
        self.timedep = xr.Dataset()
        if self.grid is None:
            return

        vars_by_name = get_var_by_name()
        forcings = self.sim.forcings()
        # Series without a timedep name are init-only (e.g. nudging targets).
        names = [
            name for name in forcings.series if vars_by_name[name].time_dependent_name
        ]
        if not names:
            return

        runtime = self.retrieve_module(TimeModule).runtime
        time_axis = resolve_time_axis(
            (forcings.series[name] for name in names),
            explicit=self.timesteps,
            runtime=runtime,
        )
        resampled = forcings.series_dataset(time_axis, names)
        self.timedep = xr.Dataset(
            {
                vars_by_name[name].time_dependent_name: resampled[name].assign_attrs(
                    var_attrs(
                        vars_by_name[name],
                        " (time-dependent)" if Z_DIM in resampled[name].dims else "",
                    )
                )
                for name in names
            }
        )
        if Z_DIM not in self.timedep.dims:
            self.timedep = self.timedep.assign_coords(
                {Z_DIM: np.asarray(self.grid.zt, dtype=float)}
            )

        # time 0 lives in the base profiles and is not counted
        self.ntimedep = self.timedep.sizes[TIME_DIM] - 1
        if self.ntimedep < 2:
            raise ValueError(
                f"TimedependentModule requires at least 3 time points (including time=0) for interpolation; got {time_axis.tolist()}"
            )
        return

    def write_files(self):
        if not self.timedep.data_vars:
            return
        self.profile_writer.write_forcings(
            self.timedep, self.output_path / "input", self.exp_id
        )
        return
