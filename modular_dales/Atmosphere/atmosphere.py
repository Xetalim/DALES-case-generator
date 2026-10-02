import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.shapes import SHAPE_FUNCTIONS
from modular_dales.IO_helpers import AtmosphereProfileWriter
from modular_dales.modular.forcing import (
    ForcingSet,
    collect_forcings,
    to_dataset,
    var_attrs,
)
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.modular.time_dependent_scalars import (
    TIME_DIM,
    Z_DIM,
    TimeDependentScalar,
    profile,
    resample_time,
    resolve_time_axis,
    stack_in_time,
)
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.vars import VariableDefinition, get_all_vars, get_var_by_name

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def _value_at(value, time_value: float):
    if isinstance(value, TimeDependentScalar):
        return value.at(time_value)
    return value


def _union_of_times(values: Iterable) -> list[float]:
    return sorted(
        {
            float(t)
            for value in values
            if isinstance(value, TimeDependentScalar)
            for t in value.times
        }
    )


@register_module
@dataclass
class AtmosphericProfile:
    """Single atmospheric profile configuration.

    Parameters may be ``TimeDependentScalar`` objects, each with its own time
    points; the profile is then evaluated on the union of those times, with
    every parameter linearly interpolated in time.
    """

    variable: VariableDefinition
    """Variable definition (see modular_dales.Atmosphere.vars)."""
    shape: str
    """Shape function (lin, exp, linmlsurf, expsinw, etc.)"""
    params: dict[str, float | TimeDependentScalar]
    """Shape-specific parameters"""

    def evaluate(self, z, time_value: float = 0.0) -> xr.DataArray:
        if self.shape not in SHAPE_FUNCTIONS:
            raise ValueError(
                f"Unknown shape function '{self.shape}' for variable '{self.variable.name}'"
            )
        params = {
            key: _value_at(value, time_value) for key, value in self.params.items()
        }
        z_arr = np.asarray(z, dtype=float)
        return profile(
            SHAPE_FUNCTIONS[self.shape](z_arr, **params),
            z_arr,
            name=self.variable.name,
            attrs=var_attrs(self.variable),
        )

    def timed_forcing_series(self, z) -> xr.DataArray | None:
        times = _union_of_times(self.params.values())
        if not times:
            return None
        return stack_in_time(times, [self.evaluate(z, t) for t in times])


@register_module
@dataclass
class InterpolatedProfile:
    """Profile interpolated between points; points may be ``TimeDependentScalar``."""

    variable: VariableDefinition
    """Variable definition (see modular_dales.Atmosphere.vars)."""
    z: list[float]
    """list of z coordinates for interpolation"""
    points: list[float | TimeDependentScalar]
    """Values at the specified z coordinates"""
    fill_value: float | str = "extrapolate"
    """Value to use for beyond the provided z range, or 'extrapolate' to extrapolate."""

    def evaluate(self, z, time_value: float = 0.0) -> xr.DataArray:
        points = xr.DataArray(
            [_value_at(point, time_value) for point in self.points],
            dims=(Z_DIM,),
            coords={Z_DIM: np.asarray(self.z, dtype=float)},
            name=self.variable.name,
            attrs=var_attrs(self.variable),
        )
        return points.interp(
            {Z_DIM: np.asarray(z, dtype=float)},
            kwargs={"fill_value": self.fill_value},
        )

    def timed_forcing_series(self, z) -> xr.DataArray | None:
        times = _union_of_times(self.points)
        if not times:
            return None
        return stack_in_time(times, [self.evaluate(z, t) for t in times])


@register_module
@dataclass
class TimedAtmosphereProfile:
    """Profile assignment at a specific simulation time (seconds)."""

    time: float
    profile: AtmosphericProfile | InterpolatedProfile


@register_module
@dataclass
class AtmosphereModule(simulation_module):
    """Atmosphere module for profile setup.

    User-configured profiles take precedence over profiles provided by other
    modules (e.g. :class:`LS2DAtmosphereModule`); missing variables are filled
    from those providers. Writes ``init.<exp_id>.nc``, including nudging
    targets interpolated to a common time axis.

    Args:
        sim: Parent simulation instance
        shaped_profiles: Mapping of variable name to shaped profile configuration
        interpolated_profiles: Mapping of variable name to interpolated profile configuration
        timed_profiles: Profiles at specific times; per variable these form a
            time series (each variable may use its own time points)
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    shaped_profiles: list[AtmosphericProfile] = field(default_factory=list)
    interpolated_profiles: list[InterpolatedProfile] = field(default_factory=list)
    timed_profiles: list[TimedAtmosphereProfile] = field(default_factory=list)
    profile_writer: AtmosphereProfileWriter = field(
        default_factory=AtmosphereProfileWriter,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )

    forcings: ForcingSet = field(
        default_factory=ForcingSet,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    """Merged forcings of this module and all other providers (set in prepare_calculation)."""

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "AtmosphereModule"

    def _initialize_from_sim(self, sim):
        sim.has_atmosphere_module = True
        return super()._initialize_from_sim(sim)

    # Generic add method
    def __add__(
        self,
        other: AtmosphericProfile
        | InterpolatedProfile
        | TimedAtmosphereProfile
        | list
        | tuple,
    ):
        if isinstance(other, AtmosphericProfile):
            self.shaped_profiles.append(other)
        elif isinstance(other, InterpolatedProfile):
            self.interpolated_profiles.append(other)
        elif isinstance(other, TimedAtmosphereProfile):
            self.timed_profiles.append(other)
        elif (
            isinstance(other, tuple)
            and len(other) == 2
            and isinstance(other[0], (int, float))
            and isinstance(other[1], (AtmosphericProfile, InterpolatedProfile))
        ):
            self.timed_profiles.append(
                TimedAtmosphereProfile(time=float(other[0]), profile=other[1])
            )
        elif isinstance(other, (list, tuple)):
            for item in other:
                self += item  # recursive
        else:
            raise TypeError(
                f"Cannot add object of type {type(other)} to AtmosphereModule"
            )
        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def _validate_profile_variable(
        self, configured: AtmosphericProfile | InterpolatedProfile
    ):
        variable = configured.variable
        if variable not in get_all_vars() or not variable.is_profile:
            raise ValueError(f"Unknown atmospheric variable '{variable.name}'")

    @staticmethod
    def _require_time_dependent(var_definition: VariableDefinition) -> None:
        if not var_definition.can_be_time_dependent:
            raise ValueError(
                f"Variable '{var_definition.name}' cannot be time-dependent"
            )

    def provide_forcings(self) -> ForcingSet:
        """Evaluate the user-configured profiles on the grid."""

        if self.grid is None:
            raise ValueError("AtmosphereModule requires a simulation grid")
        z = np.asarray(self.grid.zt, dtype=float)
        forcings = ForcingSet()

        base_profiles = {}
        for configured in self.shaped_profiles + self.interpolated_profiles:
            self._validate_profile_variable(configured)
            base_profiles[configured.variable] = configured

        param_series_vars = set()
        for var_definition, configured in base_profiles.items():
            name = var_definition.name
            values = configured.evaluate(z)
            series = configured.timed_forcing_series(z)
            if series is not None:
                self._require_time_dependent(var_definition)
                forcings.series[name] = series
                param_series_vars.add(var_definition)
            if var_definition.must_only_be_time_dependent:
                if series is None:
                    forcings.series[name] = values.expand_dims({TIME_DIM: [0.0]})
            else:
                forcings.initial[name] = values

        timed_by_var: dict[VariableDefinition, dict[float, xr.DataArray]] = {}
        for timed_profile in self.timed_profiles:
            configured = timed_profile.profile
            self._validate_profile_variable(configured)
            self._require_time_dependent(configured.variable)
            time_value = float(timed_profile.time)
            timed_by_var.setdefault(configured.variable, {})[time_value] = (
                configured.evaluate(z, time_value)
            )

        for var_definition, values_by_time in timed_by_var.items():
            if var_definition in param_series_vars:
                raise ValueError(
                    f"'{var_definition.name}' has both TimeDependentScalar parameters and "
                    "TimedAtmosphereProfile entries; use only one of the two"
                )
            times = sorted(values_by_time)
            forcings.series[var_definition.name] = stack_in_time(
                times, [values_by_time[t] for t in times]
            )

        return forcings

    def prepare_calculation(self):
        """Merge own profiles with those of other providers (e.g. LS2D)."""
        if self.grid is None:
            raise ValueError("AtmosphereModule requires a simulation grid")

        if any(module is self for module in self.sim.modules):
            self.forcings = self.sim.forcings()
        else:
            # An AtmosphereModule outside the simulation (e.g. an open-boundary
            # source) combines its own profiles with the other providers.
            self.forcings = collect_forcings(
                [m for m in self.sim.modules if not isinstance(m, AtmosphereModule)]
                + [self]
            )
        logger.info("AtmosphereModule: Profiles prepared")

    def check_settings(self):
        """Check atmosphere settings validity."""
        return

    def initial_profiles(self) -> xr.Dataset:
        """Initial profiles written to ``init.<exp_id>.nc`` (on ``zt``)."""
        vars_by_name = get_var_by_name()
        z = np.asarray(self.grid.zt, dtype=float)
        return to_dataset(
            {
                name: values
                for name, values in self.forcings.initial.items()
                if vars_by_name[name].is_initial_profile
            }
        ).assign_coords({Z_DIM: z})

    def init_dataset(self) -> xr.Dataset:
        """Contents of ``init.<exp_id>.nc``.

        Initial profiles plus the ``init_time_height`` variables (nudging
        targets and timescales) interpolated onto one common time axis.
        """
        vars_by_name = get_var_by_name()
        time_height = {
            name: series
            for name, series in self.forcings.series.items()
            if vars_by_name[name].init_time_height
        }
        for name, values in self.forcings.initial.items():
            if vars_by_name[name].init_time_height and name not in time_height:
                time_height[name] = values.expand_dims({TIME_DIM: [0.0]})

        dataset = self.initial_profiles()
        if time_height:
            times = resolve_time_axis(time_height.values())
            dataset = dataset.merge(
                to_dataset(
                    {
                        vars_by_name[name].init_name: resample_time(
                            series, times
                        ).assign_attrs(var_attrs(vars_by_name[name]))
                        for name, series in time_height.items()
                    }
                ),
                join="exact",
            )
        return dataset

    def write_files(self):
        """Write ``init.<exp_id>.nc`` (incl. nudging) and generate plots."""
        if self.grid is None:
            return

        output_input_path = self.output_path / "input"
        self.profile_writer.write_init(
            self.init_dataset(), output_input_path, self.exp_id
        )
        self.profile_writer.plot_profiles(
            self.initial_profiles(), output_input_path, self.exp_id
        )
