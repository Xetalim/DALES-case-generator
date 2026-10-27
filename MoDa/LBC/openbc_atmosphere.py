"""Atmosphere-profile open-boundary module."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import xarray as xr

from MoDa.Atmosphere import (
    InterpolatedProfile,
    TimedAtmosphereProfile,
    profiles_to_forcings,
)
from MoDa.Atmosphere.Profile import AtmosphericProfile
from MoDa.LBC.nest_dales_in_dales import boundary_fields_fine
from MoDa.LBC.openbc import OpenBoundaryModule
from MoDa.LBC.openboundary_config import OpenBoundaryConfig
from MoDa.modular.time_dependent_scalars import (
    TIME_DIM,
    Z_DIM,
    resample_time,
    resolve_time_axis,
)
from MoDa.module_registry import register_module
from MoDa.vars import ProfileVariable

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class _AtmosphereProfilesOpenBCOptions:
    """Explicit homogeneous boundary and initialization profiles."""

    u: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    v: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    w: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    thl: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    qt: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    e120: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    tracers: dict[
        str, AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = ["u", "v", "w", "thl", "qt", "e120"]
        if set(self.tracers) & set(expected):
            raise ValueError("Tracer names must not replace u, v, w, thl, qt or e120")

        inputs: dict[
            str, AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
        ] = {name: getattr(self, name) for name in expected}
        inputs.update(self.tracers)

        for name, source in inputs.items():
            if not isinstance(source, (AtmosphericProfile, InterpolatedProfile, list)):
                raise TypeError(
                    f"Boundary profile '{name}' requires a profile or a list of timed profiles"
                )

            entries = (
                source if isinstance(source, list) else source.timed_forcing_series()
            )
            if not entries:
                raise ValueError(f"Boundary profile '{name}' must not be empty")
            if isinstance(source, list) and not all(
                isinstance(entry, TimedAtmosphereProfile) for entry in entries
            ):
                raise TypeError(
                    f"Boundary profile '{name}' lists require TimedAtmosphereProfile entries"
                )

            for entry in entries:
                if not isinstance(
                    entry.profile, (AtmosphericProfile, InterpolatedProfile)
                ):
                    raise TypeError(
                        f"Boundary profile '{name}' requires AtmosphericProfile or InterpolatedProfile"
                    )
                variable = entry.profile.variable
                actual = (
                    variable.boundary_name
                    if isinstance(variable, ProfileVariable) and name in expected
                    else variable.name
                )
                if actual != name:
                    raise ValueError(
                        f"Boundary field '{name}' requires variable '{name}', got '{entry.profile.variable.name}'"
                    )

    def provide_profiles(self) -> list[TimedAtmosphereProfile]:
        expected = ["u", "v", "w", "thl", "qt", "e120"]
        inputs: dict[
            str, AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
        ] = {name: getattr(self, name) for name in expected}
        inputs.update(self.tracers)

        profiles: list[TimedAtmosphereProfile] = []
        for source in inputs.values():
            entries = (
                source if isinstance(source, list) else source.timed_forcing_series()
            )
            profiles.extend(entries)
        return profiles

    add_to_top_thl: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_std: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_seed: int | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_boundaries: list[str] | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_variables: list[str] | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_minzt: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_maxzt: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )


@register_module
@dataclass
class Nest_in_AtmosphereProfiles(OpenBoundaryModule, _AtmosphereProfilesOpenBCOptions):
    module_name: str = field(default="Nest_in_AtmosphereProfiles", init=False)

    def prepare_calculation(self) -> None:
        self._prepare_from_atmosphere()

    def _prepare_from_atmosphere(self) -> None:
        initial, series = self._evaluate_profiles()
        ds, boundaries, base_vars = self._build_atmosphere_boundaries_dataset(
            initial, series
        )
        ds = self._apply_atmosphere_boundary_noise(ds, boundaries, base_vars)

        config = OpenBoundaryConfig(
            e12=self.e12,
            tracernames=list(self.tracernames or []),
            tchunk=self.tchunk,
            start=self.start,
            time0=self.time0,
            author="author",
            end=self.end,
        )
        self.initfields = self._build_initfields_dataset(
            initial, list(self.tracernames or [])
        )
        self.boundaries = boundary_fields_fine.set_openboundary_attrs(config, ds)

    def _build_initfields_dataset(
        self,
        profiles_1d: dict[str, xr.DataArray],
        tracernames: list[str],
    ) -> xr.Dataset:
        dims_by_var = {
            "u0": ("u", ("zt", "yt", "xm")),
            "v0": ("v", ("zt", "ym", "xt")),
            "w0": ("w", ("zm", "yt", "xt")),
            "thl0": ("thl", ("zt", "yt", "xt")),
            "qt0": ("qt", ("zt", "yt", "xt")),
            "e120": ("e120", ("zt", "yt", "xt")),
        }
        initfields_ds = xr.Dataset(
            {
                name: self._broadcast_profile(profiles_1d[var], dims)
                for name, (var, dims) in dims_by_var.items()
            }
        )
        initfields_ds = initfields_ds.assign_attrs(
            {
                "history": f"Created on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                "author": "author",
                "time0": self.time0,
            }
        )
        for name in tracernames:
            initfields_ds[f"{name}0"] = self._broadcast_profile(
                profiles_1d[name], ("zt", "yt", "xt")
            )
        return initfields_ds

    def _grid_coord(self, dim: str) -> xr.DataArray:
        values = np.asarray(getattr(self.openBCgrid, dim), dtype=float)
        return xr.DataArray(values, dims=(dim,), coords={dim: values})

    def _broadcast_profile(
        self,
        prof: xr.DataArray,
        dims: tuple[str, ...],
    ) -> xr.DataArray:
        if "zm" in dims:
            zm = self._grid_coord("zm").values
            zt = prof[Z_DIM].values
            prof = (
                prof.interp({Z_DIM: np.clip(zm, zt.min(), zt.max())})
                .assign_coords({Z_DIM: zm})
                .rename({Z_DIM: "zm"})
            )
        elif "zt" not in dims:
            prof = prof.isel({Z_DIM: -1}, drop=True)
        horizontal = [self._grid_coord(dim) for dim in dims if dim not in ("zt", "zm")]
        out = xr.broadcast(prof, *horizontal)[0]
        leading = (TIME_DIM,) if TIME_DIM in out.dims else ()
        return out.transpose(*leading, *dims)

    def _evaluate_profiles(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        heights = self._grid_coord(Z_DIM).values
        profiles = self.provide_profiles()
        forcings = profiles_to_forcings(heights, profiles)
        names = {
            entry.profile.variable.name: entry.profile.variable.boundary_name
            for entry in profiles
        }
        initial = {names[name]: values for name, values in forcings.initial.items()}
        series = {names[name]: values for name, values in forcings.series.items()}
        for name, values in series.items():
            initial[name] = values.isel({TIME_DIM: 0}, drop=True)
        return initial, series

    def _boundary_dims(self, var: str, bnd: str) -> tuple[str, ...]:
        var_dims = {
            "u": {"x": "xm", "y": "yt", "z": "zt"},
            "v": {"x": "xt", "y": "ym", "z": "zt"},
            "w": {"x": "xt", "y": "yt", "z": "zm"},
        }.get(var, {"x": "xt", "y": "yt", "z": "zt"})
        base_dims = {
            "west": ("z", "y"),
            "east": ("z", "y"),
            "south": ("z", "x"),
            "north": ("z", "x"),
            "top": ("y", "x"),
        }[bnd]
        return tuple(var_dims[dim] for dim in base_dims)

    def _build_atmosphere_boundaries_dataset(
        self,
        initial_by_var: dict[str, xr.DataArray],
        series_by_var: dict[str, xr.DataArray],
    ) -> tuple[xr.Dataset, list[str], list[str]]:
        for var, initial in initial_by_var.items():
            if var not in series_by_var:
                series_by_var[var] = initial.expand_dims({TIME_DIM: [0.0]})
        all_times = resolve_time_axis(series_by_var.values())

        if self.time0 is None:
            raise ValueError(
                "Nest_in_AtmosphereProfiles requires 'time0' on OpenBoundaryModule"
            )
        time_points = np.datetime64(self.time0) + np.round(all_times).astype(
            "timedelta64[s]"
        )

        boundaries = ["west", "east", "south", "north", "top"]
        base_vars = list(series_by_var)
        ds = xr.Dataset()
        for var in base_vars:
            series = series_by_var[var]
            in_time = resample_time(series, all_times).assign_coords(
                {TIME_DIM: time_points}
            )
            for bnd in boundaries:
                prof = in_time
                if var == "thl" and bnd == "top" and self.add_to_top_thl is not None:
                    prof = prof + self.add_to_top_thl
                field_name = "e12" if var == "e120" else var
                ds[f"{field_name}{bnd}"] = self._broadcast_profile(
                    prof, self._boundary_dims(var, bnd)
                )

        return ds, boundaries, base_vars

    def _apply_atmosphere_boundary_noise(
        self,
        ds: xr.Dataset,
        boundaries: list[str],
        base_vars: list[str],
    ) -> xr.Dataset:
        noise_std = self.noise_std
        noise_minzt = self.noise_minzt
        noise_maxzt = self.noise_maxzt

        if noise_std is not None and (
            noise_minzt is not None or noise_maxzt is not None
        ):
            zt = self.openBCgrid.zt
            zm = self.openBCgrid.zm
            mask = np.ones_like(zt, dtype=bool)
            mask_zm = np.ones_like(zm, dtype=bool)
            if noise_minzt is not None:
                mask &= zt >= noise_minzt
                mask_zm &= zm >= noise_minzt
            if noise_maxzt is not None:
                mask &= zt <= noise_maxzt
                mask_zm &= zm <= noise_maxzt
            if (not np.any(mask)) or (not np.any(mask_zm)):
                logger.warning(
                    "Noise std is set but no vertical levels are within the specified min/max zt bounds; skipping noise addition."
                )
                noise_std = None
            else:
                logger.info(
                    "Applying noise with std=%.3f to levels where zt is between %.2f and %.2f",
                    noise_std,
                    zt[mask][0],
                    zt[mask][-1],
                )

        if noise_std is None or noise_std <= 0.0:
            return ds

        rng = np.random.default_rng(self.noise_seed)
        active_bounds = (
            set(boundaries)
            if self.noise_boundaries is None
            else {b for b in self.noise_boundaries if b in boundaries}
        )
        active_vars = (
            set(base_vars)
            if self.noise_variables is None
            else {v for v in self.noise_variables if v in base_vars}
        )

        for var in active_vars:
            for bnd in active_bounds:
                name = f"{var}{bnd}"
                if name not in ds:
                    continue
                arr = ds[name]
                mask_3d = xr.ones_like(arr, dtype=bool)
                if "zt" in arr.dims:
                    if noise_minzt is not None:
                        mask_3d = mask_3d.where(ds.zt >= noise_minzt, other=False)
                    if noise_maxzt is not None:
                        mask_3d = mask_3d.where(ds.zt <= noise_maxzt, other=False)
                if "zm" in arr.dims:
                    if noise_minzt is not None:
                        mask_3d = mask_3d.where(ds.zm >= noise_minzt, other=False)
                    if noise_maxzt is not None:
                        mask_3d = mask_3d.where(ds.zm <= noise_maxzt, other=False)
                noise = rng.normal(loc=0.0, scale=noise_std, size=arr.shape)
                ds[name] = arr + (mask_3d * noise)

        return ds
