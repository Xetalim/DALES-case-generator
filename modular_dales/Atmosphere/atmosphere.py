import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.shapes import ShapeFunction
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
from modular_dales.vars import (
    ProfileVariable,
    VariableDefinition,
    get_all_vars,
    get_var_by_name,
)

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


@dataclass
class Profile:
    """Base class for atmospheric profiles."""

    variable: VariableDefinition
    """Variable definition (see modular_dales.Atmosphere.vars)."""

    def evaluate(self, z, time_value: float = 0.0) -> xr.DataArray:
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def profile_at(self, time_value: float = 0.0) -> "TimedAtmosphereProfile":
        raise NotImplementedError("Subclasses must implement the profile_at method.")

    def times(self) -> list[float]:
        raise NotImplementedError("Subclasses must implement the times method.")

    def timed_forcing_series(self) -> list["TimedAtmosphereProfile"]:
        times = self.times()
        if not times:
            times = [0.0]
        return [self.profile_at(t) for t in times]


@register_module
@dataclass
class AtmosphericProfile(Profile):
    """Single atmospheric profile configuration.
    Provide a variable name (VariableDefinition) and a shape function (ShapeFunction) to define the atmospheric profile.
    Parameters for the shape function may be ``TimeDependentScalar`` objects.

    Parameters may be ``TimeDependentScalar`` objects, each with its own time
    points; the profile is then evaluated on the union of those times, with
    every parameter linearly interpolated in time.
    """

    shape: ShapeFunction
    """Shape type (lin, exp, linmlsurf, expsinw, etc.)"""

    def _evaluate(self, z, time_value: float = 0.0) -> xr.DataArray:
        if not isinstance(self.shape, ShapeFunction):
            raise TypeError(
                f"Shape must be an instance of ShapeFunction for variable '{self.variable.name}'"
            )
        params = self.shape.params_at(time_value)

        z_arr = np.asarray(z, dtype=float)
        return profile(
            self.shape.evaluate(z=z_arr, **params),
            z_arr,
            name=self.variable.name,
            attrs=var_attrs(self.variable),
        )

    def evaluate(self, z):
        z_arr = np.asarray(z, dtype=float)
        return profile(
            self.shape.evaluate(z=z_arr, **self.shape.params),
            z_arr,
            name=self.variable.name,
            attrs=var_attrs(self.variable),
        )

    def times(self) -> list[float]:
        return _union_of_times(self.shape.__dict__.values())

    def profile_at(self, time_value: float = 0.0) -> "TimedAtmosphereProfile":
        """Return the profile at a specific time as a TimedAtmosphereProfile."""
        prof = AtmosphericProfile(
            variable=self.variable, shape=self.shape.at_time(time_value)
        )
        return TimedAtmosphereProfile(time=time_value, profile=prof)


@register_module
@dataclass
class InterpolatedProfile(Profile):
    """Profile interpolated between points; points --but not z-- may be ``TimeDependentScalar``.
    Values are interpolated to the used DALES grid automatically."""

    z: list[float]
    """list of z coordinates for interpolation"""
    points: list[float | TimeDependentScalar]
    """Values at the specified z coordinates"""
    fill_value: float | str = "extrapolate"
    """Value to use for beyond the provided z range, or 'extrapolate' to extrapolate."""

    def _evaluate(self, z, time_value: float = 0.0) -> xr.DataArray:
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

    def evaluate(self, z):
        points = xr.DataArray(
            self.points,
            dims=(Z_DIM,),
            coords={Z_DIM: np.asarray(self.z, dtype=float)},
            name=self.variable.name,
            attrs=var_attrs(self.variable),
        )
        return points.interp(
            {Z_DIM: np.asarray(z, dtype=float)},
            kwargs={"fill_value": self.fill_value},
        )

    def times(self) -> list[float]:
        return _union_of_times(self.points)

    def profile_at(self, time_value: float = 0) -> "TimedAtmosphereProfile":
        prof = InterpolatedProfile(
            z=self.z,
            points=[_value_at(point, time_value) for point in self.points],
            fill_value=self.fill_value,
            variable=self.variable,
        )
        return TimedAtmosphereProfile(time=time_value, profile=prof)


@register_module
@dataclass
class TimedAtmosphereProfile:
    """Profile assignment at a specific simulation time (seconds).
    The profile may not contain time-dependent values."""

    time: float
    profile: AtmosphericProfile | InterpolatedProfile

    def __post_init__(self):
        if isinstance(self.profile, InterpolatedProfile):
            for point in self.profile.points:
                if isinstance(point, TimeDependentScalar):
                    raise TypeError(
                        "TimedAtmosphereProfile cannot contain time-dependent values."
                    )
        if isinstance(self.profile, AtmosphericProfile):
            for var in self.profile.shape.__dict__.values():
                if isinstance(var, TimeDependentScalar):
                    raise TypeError(
                        "TimedAtmosphereProfile cannot contain time-dependent values."
                    )


def profiles_to_forcings(
    z: np.ndarray | list[float], profiles: list[TimedAtmosphereProfile]
) -> ForcingSet:
    """Evaluate the user-configured profiles on the grid."""

    z = np.asarray(z, dtype=float)
    forcings = ForcingSet()

    timed_by_var: dict[VariableDefinition, dict[float, xr.DataArray]] = {}
    for timed_profile in profiles:
        configured = timed_profile.profile
        _validate_profile_variable(configured)
        time_value = float(timed_profile.time)
        timed_by_var.setdefault(configured.variable, {})[time_value] = (
            configured.evaluate(z)
        )

    for var_definition, values_by_time in timed_by_var.items():
        times = sorted(values_by_time)
        if (
            len(times) == 1
            and times[0] == 0.0
            and not var_definition.must_only_be_time_dependent
        ):
            # Only one time point at t=0, treat as initial profile
            forcings.initial[var_definition.name] = values_by_time[times[0]]
            continue
        if len(times) == 1 and times[0] != 0.0:
            raise ValueError(
                f"'{var_definition.name}' has no time=0 entry but is required to have one. Did you mean to only add a time at {times[0]}?"
            )
        _require_time_dependent(var_definition)
        forcings.series[var_definition.name] = stack_in_time(
            times, [values_by_time[t] for t in times]
        )

    return forcings


def _require_time_dependent(var_definition: VariableDefinition) -> None:
    if not var_definition.can_be_time_dependent:
        raise ValueError(f"Variable '{var_definition.name}' cannot be time-dependent")


def _validate_profile_variable(
    configured: AtmosphericProfile | InterpolatedProfile,
):
    variable = configured.variable
    if variable not in get_all_vars() or not variable.is_profile:
        raise ValueError(f"Unknown atmospheric variable '{variable.name}'")


@register_module
@dataclass
class AtmosphereModule(simulation_module):
    """Main atmosphere module to add to the dales_simulation.
    Add physical profiles with an explicit destination:
    ```python
    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=0)))
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=5, ddz=0)),
             nudging=True)
    ```

    User-configured profiles take precedence over profiles provided by other
    modules (e.g. :class:`LS2DAtmosphereModule`); missing variables are filled
    from those providers. Writes ``init.<exp_id>.nc``, including nudging
    targets interpolated to a common time axis.

    Args:
        sim: Parent simulation instance
        profiles: Regular profiles at specific simulation times.
        nudging_profiles: Physical profiles used as nudging targets, independently
            of the initial state. No state-to-nudging fallback is applied.
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    profiles: list[TimedAtmosphereProfile] = field(default_factory=list)
    nudging_profiles: list[TimedAtmosphereProfile] = field(default_factory=list)
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

    def add(
        self,
        other: AtmosphericProfile
        | InterpolatedProfile
        | TimedAtmosphereProfile
        | list
        | tuple,
        *,
        nudging: bool = False,
    ):
        """Add profiles, optionally to nudging instead of the regular destination."""
        destination = self.nudging_profiles if nudging else self.profiles
        if isinstance(other, (AtmosphericProfile, InterpolatedProfile)):
            destination.extend(other.timed_forcing_series())
        elif isinstance(other, TimedAtmosphereProfile):
            destination.append(other)
        elif (
            isinstance(other, tuple)
            and len(other) == 2
            and isinstance(other[0], (int, float))
            and isinstance(other[1], (AtmosphericProfile, InterpolatedProfile))
        ):
            destination.append(
                TimedAtmosphereProfile(time=float(other[0]), profile=other[1])
            )
        elif isinstance(other, (list, tuple)):
            for item in other:
                self.add(item, nudging=nudging)
        else:
            raise TypeError(
                f"Cannot add object of type {type(other)} to AtmosphereModule"
            )
        return self

    def provide_forcings(self) -> ForcingSet:
        """Evaluate the user-configured profiles on the grid."""

        if self.grid is None:
            raise ValueError("AtmosphereModule requires a simulation grid")
        nudging_profiles = []
        for entry in self.nudging_profiles:
            entries = (
                [entry]
                if isinstance(entry, TimedAtmosphereProfile)
                else entry.timed_forcing_series()
            )
            for timed in entries:
                variable = timed.profile.variable
                if not isinstance(variable, ProfileVariable):
                    raise TypeError("Nudging requires an atmospheric state profile")
                nudging_profiles.append(
                    replace(
                        timed,
                        profile=replace(timed.profile, variable=variable.nudger),
                    )
                )
        return profiles_to_forcings(
            np.asarray(self.grid.zt, dtype=float), self.profiles + nudging_profiles
        )

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
