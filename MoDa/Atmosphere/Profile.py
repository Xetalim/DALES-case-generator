from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from moda.Atmosphere.shapes import ShapeFunction, ShapeSpec
from moda.modular.forcing import ForcingSet, var_attrs
from moda.modular.time_dependent_scalars import (
    Z_DIM,
    TimeDependentScalar,
    profile,
    stack_in_time,
)
from moda.module_registry import register_module
from moda.vars import ProfileVariable, VariableDefinition, get_all_vars


@register_module
@dataclass
class TimedAtmosphereProfile:
    """Profile assignment at a specific simulation time (seconds).
    The profile may not contain time-dependent values."""

    time: float
    profile: AtmosphericProfile | InterpolatedProfile

    def __post_init__(self) -> None:
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


@dataclass
class Profile:
    """Base class for atmospheric profiles."""

    variable: ProfileVariable
    """Variable definition (see moda.Atmosphere.vars)."""

    def __post_init__(self) -> None:
        if not isinstance(self.variable, ProfileVariable):
            raise TypeError(
                f"variable must be an instance of ProfileVariable, got {type(self.variable).__name__}"
            )

    def evaluate(self, z: np.ndarray | list[float]) -> xr.DataArray:
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


def _union_of_times(values: Iterable[float | TimeDependentScalar]) -> list[float]:
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
class AtmosphericProfile(Profile):
    """Single atmospheric profile configuration.
    Provide a variable name (VariableDefinition) and a shape function (ShapeFunction) to define the atmospheric profile.
    Parameters for the shape function may be ``TimeDependentScalar`` objects.

    Parameters may be ``TimeDependentScalar`` objects, each with its own time
    points; the profile is then evaluated on the union of those times, with
    every parameter linearly interpolated in time.
    """

    shape: ShapeSpec
    """Shape type (lin, exp, linmlsurf, expsinw, etc.)"""

    def _evaluate(
        self, z: np.ndarray | list[float], time_value: float = 0.0
    ) -> xr.DataArray:
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

    def evaluate(self, z: np.ndarray | list[float]) -> xr.DataArray:
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


def _value_at(value: float | TimeDependentScalar, time_value: float) -> float:
    if isinstance(value, TimeDependentScalar):
        return value.at(time_value)
    return value


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

    def _evaluate(
        self, z: np.ndarray | list[float], time_value: float = 0.0
    ) -> xr.DataArray:
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

    def evaluate(self, z: np.ndarray | list[float]) -> xr.DataArray:
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


def _validate_profile_variable(
    configured: AtmosphericProfile | InterpolatedProfile,
) -> None:
    variable = configured.variable
    if variable not in get_all_vars() or not variable.is_profile:
        raise ValueError(f"Unknown atmospheric variable '{variable.name}'")


def _require_time_dependent(var_definition: VariableDefinition) -> None:
    if not var_definition.can_be_time_dependent:
        raise ValueError(f"Variable '{var_definition.name}' cannot be time-dependent")


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
