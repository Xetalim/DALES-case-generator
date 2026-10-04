"""Worker for building open-boundary fields from atmosphere profiles."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from modular_dales.Atmosphere import (
    AtmosphericProfile,
    InterpolatedProfile,
    TimedAtmosphereProfile,
)
from modular_dales.LBC.nest_dales_in_dales import boundary_fields_fine
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.modular.time_dependent_scalars import (
    TIME_DIM,
    Z_DIM,
    resample_time,
    resolve_time_axis,
    stack_in_time,
)

if TYPE_CHECKING:
    from modular_dales.LBC.openbc import do_openboundary


logger = logging.getLogger(__name__)


class OpenBCAtmosphereWorker:
    """Build initial fields and boundaries solely from explicitly supplied profiles."""

    def __init__(self, module: do_openboundary) -> None:
        self.module = module

    def prepare(self) -> tuple[xr.Dataset, xr.Dataset]:
        profiles_1d, series = self._evaluate_profiles()
        ds, boundaries, base_vars = self._build_atmosphere_boundaries_dataset(
            series,
        )
        ds = self._apply_atmosphere_boundary_noise(ds, boundaries, base_vars)

        config = OpenBoundaryConfig(
            e12=self.module.e12,
            tracernames=list(self.module.tracernames or []),
            tchunk=self.module.tchunk,
            start=self.module.start,
            time0=self.module.time0,
            author="author",
            end=self.module.end,
        )

        initfields = self._build_initfields_dataset(profiles_1d)
        boundaries_ds = boundary_fields_fine.set_openboundary_attrs(
            config,
            ds,
        )
        return boundaries_ds, initfields

    def _build_initfields_dataset(
        self,
        profiles_1d: dict[str, xr.DataArray],
    ) -> xr.Dataset:
        dims_by_var = {
            "u0": ("u", ("zt", "yt", "xm")),
            "v0": ("v", ("zt", "ym", "xt")),
            "w0": ("w", ("zm", "yt", "xt")),
            "thl0": ("thl", ("zt", "yt", "xt")),
            "qt0": ("qt", ("zt", "yt", "xt")),
            "e120": ("e12", ("zt", "yt", "xt")),
        }
        initfields = xr.Dataset(
            {
                name: self._broadcast_profile(profiles_1d[var], dims)
                for name, (var, dims) in dims_by_var.items()
            }
        )
        initfields = initfields.assign_attrs(
            {
                "history": f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                "author": "author",
                "time0": self.module.time0,
            }
        )
        for name in self.module.nest_in_atmosphere.tracers:
            initfields[f"{name}0"] = self._broadcast_profile(
                profiles_1d[name], ("zt", "yt", "xt")
            )
        return initfields

    def _grid_coord(self, dim: str) -> xr.DataArray:
        values = np.asarray(getattr(self.module.openBCgrid, dim), dtype=float)
        return xr.DataArray(values, dims=(dim,), coords={dim: values})

    def _broadcast_profile(
        self,
        prof: xr.DataArray,
        dims: tuple[str, ...],
    ) -> xr.DataArray:
        """Broadcast a ``zt`` profile (optionally with ``time``) to ``dims``.

        Profiles are interpolated to ``zm`` when needed; without a vertical
        dimension (top boundary) the highest level is used.
        """
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
        horizontal = [self._grid_coord(d) for d in dims if d not in ("zt", "zm")]
        out = xr.broadcast(prof, *horizontal)[0]
        leading = (TIME_DIM,) if TIME_DIM in out.dims else ()
        return out.transpose(*leading, *dims)

    def _evaluate_profiles(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        configured = self.module.nest_in_atmosphere
        expected = {
            "u": "ua",
            "v": "va",
            "w": "w",
            "thl": "thetal",
            "qt": "qt",
            "e12": "tke",
        }
        inputs = {name: getattr(configured, name) for name in expected}
        if set(configured.tracers) & set(expected):
            raise ValueError("Tracer names must not replace u, v, w, thl, qt or e12")
        inputs.update(configured.tracers)
        initial = {}
        series = {}
        heights = self._grid_coord(Z_DIM).values
        for name, source in inputs.items():
            entries = source if isinstance(source, list) else [source]
            if not entries:
                raise ValueError(f"Boundary profile '{name}' must not be empty")
            for entry in entries:
                profile = (
                    entry.profile
                    if isinstance(entry, TimedAtmosphereProfile)
                    else entry
                )
                if not isinstance(profile, (AtmosphericProfile, InterpolatedProfile)):
                    raise TypeError(
                        f"Boundary profile '{name}' must be a shaped or interpolated profile"
                    )
                required_name = expected.get(name, name)
                if profile.variable.name != required_name:
                    raise ValueError(
                        f"Boundary field '{name}' requires variable '{required_name}', got '{profile.variable.name}'"
                    )
            if isinstance(source, list):
                if not all(
                    isinstance(entry, TimedAtmosphereProfile) for entry in entries
                ):
                    raise TypeError(
                        f"Boundary profile '{name}' lists must contain TimedAtmosphereProfile entries"
                    )
                times = [float(entry.time) for entry in entries]
                if 0.0 not in times or any(time < 0 for time in times):
                    raise ValueError(
                        f"Timed boundary profile '{name}' requires time zero and nonnegative times"
                    )
                values = [
                    entry.profile.evaluate(heights, entry.time) for entry in entries
                ]
                series[name] = stack_in_time(times, values).rename(name)
                initial[name] = series[name].sel({TIME_DIM: 0.0}, drop=True)
            else:
                initial[name] = source.evaluate(heights, 0.0).rename(name)
                timed = source.timed_forcing_series(heights)
                if timed is not None:
                    if 0.0 not in timed[TIME_DIM].values or bool(
                        (timed[TIME_DIM] < 0).any()
                    ):
                        raise ValueError(
                            f"Timed boundary profile '{name}' requires time zero and nonnegative times"
                        )
                    series[name] = timed.rename(name)
                else:
                    series[name] = initial[name].expand_dims({TIME_DIM: [0.0]})
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
        return tuple(var_dims[d] for d in base_dims)

    def _build_atmosphere_boundaries_dataset(
        self,
        series_by_var: dict[str, xr.DataArray],
    ) -> tuple[xr.Dataset, list[str], list[str]]:
        all_times = resolve_time_axis(series_by_var.values())

        base_time_str = self.module.time0
        if base_time_str is None:
            raise ValueError(
                "Nest_in_AtmosphereProfiles requires 'time0' on do_openboundary"
            )
        time_points = np.datetime64(base_time_str) + np.round(all_times).astype(
            "timedelta64[s]"
        )

        boundaries = ["west", "east", "south", "north", "top"]
        base_vars = list(series_by_var)

        add_to_top_thl = getattr(self.module.nest_in_atmosphere, "add_to_top_thl", None)
        ds = xr.Dataset()
        for var in base_vars:
            series = series_by_var[var]
            in_time = resample_time(series, all_times).assign_coords(
                {TIME_DIM: time_points}
            )
            for bnd in boundaries:
                prof = in_time
                if var == "thl" and bnd == "top" and add_to_top_thl is not None:
                    prof = prof + add_to_top_thl
                ds[f"{var}{bnd}"] = self._broadcast_profile(
                    prof, self._boundary_dims(var, bnd)
                )

        return ds, boundaries, base_vars

    def _apply_atmosphere_boundary_noise(
        self,
        ds: xr.Dataset,
        boundaries: list[str],
        base_vars: list[str],
    ) -> xr.Dataset:
        noise_std = getattr(self.module.nest_in_atmosphere, "noise_std", None)
        noise_minzt = getattr(self.module.nest_in_atmosphere, "noise_minzt", None)
        noise_maxzt = getattr(self.module.nest_in_atmosphere, "noise_maxzt", None)

        if noise_std is not None and (
            noise_minzt is not None or noise_maxzt is not None
        ):
            zt = self.module.openBCgrid.zt
            zm = self.module.openBCgrid.zm
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

        rng = np.random.default_rng(
            getattr(self.module.nest_in_atmosphere, "noise_seed", None)
        )
        requested_bounds = getattr(
            self.module.nest_in_atmosphere, "noise_boundaries", None
        )
        requested_vars = getattr(
            self.module.nest_in_atmosphere, "noise_variables", None
        )

        active_bounds = (
            set(boundaries)
            if requested_bounds is None
            else {b for b in requested_bounds if b in boundaries}
        )
        active_vars = (
            set(base_vars)
            if requested_vars is None
            else {v for v in requested_vars if v in base_vars}
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
