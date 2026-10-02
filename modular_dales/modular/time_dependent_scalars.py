from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from modular_dales.MODULE_REGISTRY import MODULE_REGISTRY, register_special_serializing

TIME_DIM = "time"
"""Time dimension of all forcing series, in seconds since the start of the run."""
Z_DIM = "zt"
"""Vertical dimension of all profiles (DALES full levels)."""
SOIL_DIM = "zs"
"""Soil level dimension; its coordinate (when present) is depth in m."""


def profile(
    values, z, name: str | None = None, attrs: dict | None = None
) -> xr.DataArray:
    """Profile on the ``zt`` dimension."""
    z_arr = np.asarray(z, dtype=float)
    data = np.broadcast_to(np.asarray(values, dtype=float), z_arr.shape)
    return xr.DataArray(
        data.copy(), dims=(Z_DIM,), coords={Z_DIM: z_arr}, name=name, attrs=attrs or {}
    )


def check_time_series(series: xr.DataArray) -> xr.DataArray:
    """Sort a series by time and reject empty or duplicate time axes."""
    if TIME_DIM not in series.dims or series.sizes[TIME_DIM] == 0:
        raise ValueError(
            f"Time series '{series.name}' needs a non-empty '{TIME_DIM}' dimension"
        )
    series = series.sortby(TIME_DIM)
    if np.any(np.diff(series[TIME_DIM].values) == 0):
        raise ValueError(
            f"Time series '{series.name}' has duplicate times: {series[TIME_DIM].values}"
        )
    return series


def stack_in_time(
    times: Iterable[float], profiles: Iterable[xr.DataArray]
) -> xr.DataArray:
    """Combine profiles (or scalars) at the given times into one series."""
    times_arr = np.asarray(list(times), dtype=float)
    stacked = xr.concat(list(profiles), dim=TIME_DIM).assign_coords(
        {TIME_DIM: times_arr}
    )
    return check_time_series(stacked)


def resample_time(series: xr.DataArray, times: Iterable[float]) -> xr.DataArray:
    """Linearly interpolate ``series`` onto ``times``; values outside its range are held constant."""
    target = np.asarray(list(times), dtype=float)
    source = series[TIME_DIM].values
    if source.size == 1:
        resampled = series.isel({TIME_DIM: np.zeros(target.size, dtype=int)})
    else:
        resampled = series.interp({TIME_DIM: np.clip(target, source[0], source[-1])})
    return resampled.assign_coords({TIME_DIM: target}).assign_attrs(series.attrs)


def value_at(series: xr.DataArray, time_value: float) -> xr.DataArray:
    return resample_time(series, [time_value]).isel({TIME_DIM: 0}, drop=True)


def resolve_time_axis(
    series: Iterable[xr.DataArray],
    explicit: Iterable[float] | None = None,
    runtime: float | None = None,
) -> np.ndarray:
    """Common time axis: ``explicit`` if given, else the union of all series times.

    Time 0 and ``runtime`` (when beyond the last point) are always included.
    """

    explicit_list = [float(t) for t in explicit] if explicit is not None else []
    if explicit_list:
        times = set(explicit_list)
    else:
        times = {float(t) for s in series for t in s[TIME_DIM].values}
    times.add(0.0)
    if runtime is not None and max(times) < float(runtime):
        times.add(float(runtime))
    return np.asarray(sorted(times), dtype=float)


@register_special_serializing
@dataclass
class TimeDependentScalar:
    """Time-dependent scalar value defined on explicit timesteps.

    Different instances may use different (numbers of) time points; they are
    interpolated onto a common time axis when the forcings are written.
    """

    times: list[float] = field(
        default_factory=list, metadata={"serialize": True}, init=True
    )
    values: list[float] = field(
        default_factory=list, metadata={"serialize": True}, init=True
    )

    def __post_init__(self):
        if isinstance(self.times, np.ndarray):
            self.times = self.times.tolist()
        if isinstance(self.values, np.ndarray):
            self.values = self.values.tolist()
        if len(self.times) != len(self.values):
            raise ValueError(
                f"TimeDependentScalar times(len={len(self.times)}) and values(len={len(self.values)}) must have equal length"
            )
        if len(self.times) == 0:
            raise ValueError("TimeDependentScalar requires at least one point")
        normalized = [float(t) for t in self.times]
        if len(set(normalized)) != len(normalized):
            raise ValueError("TimeDependentScalar times contain duplicates")

    def as_series(self, name: str | None = None) -> xr.DataArray:
        return check_time_series(
            xr.DataArray(
                np.asarray(self.values, dtype=float),
                dims=(TIME_DIM,),
                coords={TIME_DIM: np.asarray(self.times, dtype=float)},
                name=name,
            )
        )

    def at(self, time_value: float) -> float:
        return float(value_at(self.as_series(), time_value))

    def value_at_start(self) -> float:
        return self.at(0.0)


# Register TimeDependentScalar for YAML deserialization without making it a simulation module
MODULE_REGISTRY["TimeDependentScalar"] = TimeDependentScalar
