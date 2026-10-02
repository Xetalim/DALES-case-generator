"""Common container and merge logic for data exchanged between modules.

Every simulation module can act as a provider by implementing
``provide_forcings()``; it is evaluated once per module. Requesters use
``sim.forcings()`` (all modules in the simulation, merged once) or
:func:`collect_forcings` for a custom set of providers.

All data are ``xr.DataArray`` objects keyed by variable name (see
:mod:`modular_dales.vars` for what each variable is used for):

* atmospheric profiles have the dimension ``zt``,
* soil profiles ``zs`` and radiation soundings ``lev`` (pressure),
* series have ``time`` (seconds) and, for profiles, ``zt``. Every series
  keeps its own time axis until it is resampled with :meth:`ForcingSet.series_dataset`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import xarray as xr

from modular_dales.modular.time_dependent_scalars import resample_time
from modular_dales.vars import backrad_o3, backrad_q, backrad_T

if TYPE_CHECKING:
    from modular_dales.modular.simulation_module import simulation_module
    from modular_dales.vars import VariableDefinition

USER_FORCING_PRIORITY = 100
EXTERNAL_FORCING_PRIORITY = 50


def var_attrs(var_definition: VariableDefinition, suffix: str = "") -> dict:
    return {
        "long_name": var_definition.long_name + suffix,
        "units": var_definition.unit,
    }


def named(values: xr.DataArray, var_definition: VariableDefinition) -> xr.DataArray:
    """``values`` named and annotated as ``var_definition``."""
    out = values.rename(var_definition.name)
    out.attrs = var_attrs(var_definition)
    return out


def sounding_forcings(sounding: xr.Dataset) -> dict[str, xr.DataArray]:
    """Radiation sounding (``T``, ``q``, optional ``o3`` on pressure ``lev``) as provided variables."""
    return {
        var_definition.name: named(sounding[name].astype(float), var_definition)
        for var_definition, name in (
            (backrad_T, "T"),
            (backrad_q, "q"),
            (backrad_o3, "o3"),
        )
        if name in sounding
    }


@dataclass
class ForcingSet:
    """Time-independent values (``initial``) and time series (``series``) per variable."""

    initial: dict[str, xr.DataArray] = field(default_factory=dict)
    series: dict[str, xr.DataArray] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.initial or self.series)

    def initial_dataset(self, names: Iterable[str] | None = None) -> xr.Dataset:
        selected = (
            self.initial if names is None else {n: self.initial[n] for n in names}
        )
        return to_dataset(selected)

    def series_dataset(
        self, times, names: Iterable[str] | None = None
    ) -> xr.Dataset:
        """Selected series, linearly interpolated onto the common axis ``times``."""
        selected = self.series if names is None else {n: self.series[n] for n in names}
        return to_dataset(
            {name: resample_time(da, times) for name, da in selected.items()}
        )


def to_dataset(arrays: Mapping[str, xr.DataArray]) -> xr.Dataset:
    """Dataset of named arrays; their coordinates (``zt``, ``time``) must match exactly."""
    names = list(arrays)
    aligned = xr.align(*(arrays[name] for name in names), join="exact")
    return xr.Dataset(
        {
            name: da.reset_coords(drop=True).rename(name)
            for name, da in zip(names, aligned)
        }
    )


def collect_forcings(providers: Iterable[simulation_module]) -> ForcingSet:
    """Merge the (cached) forcings of ``providers``.

    Per variable (and per category) the provider with the highest
    ``forcing_priority`` wins as a whole; equal priorities conflict.
    """

    merged = ForcingSet()
    owners: dict[tuple[str, str], tuple[int, str]] = {}
    for module in providers:
        provided = module.provided_forcings()
        if provided.is_empty():
            continue
        priority = int(module.forcing_priority)
        name = module.module_name or type(module).__name__
        for category in ("initial", "series"):
            target = getattr(merged, category)
            for key, value in getattr(provided, category).items():
                owner = owners.get((category, key))
                if owner is not None and owner[0] == priority:
                    raise ValueError(
                        f"Forcing '{key}' ({category}) is provided by both '{owner[1]}' and '{name}' with equal priority"
                    )
                if owner is None or priority > owner[0]:
                    target[key] = value
                    owners[(category, key)] = (priority, name)
    return merged
