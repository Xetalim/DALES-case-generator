"""Worker for building open-boundary fields from atmosphere profiles."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from modular_dales.Atmosphere import AtmosphereModule
from modular_dales.LBC.nest_dales_in_dales import boundary_fields_fine
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.modular.forcing import ForcingSet, collect_forcings
from modular_dales.modular.time_dependent_scalars import (
    TIME_DIM,
    Z_DIM,
    resample_time,
    resolve_time_axis,
    value_at,
)
from modular_dales.vars import get_all_vars, get_var_by_name

if TYPE_CHECKING:
    from modular_dales.LBC.openbc import do_openboundary


logger = logging.getLogger(__name__)


class OpenBCAtmosphereWorker:
    """Build open boundaries from the forcings of an atmosphere module.

    The atmosphere module (usually an external :class:`AtmosphereModule`)
    merges its own profiles with other providers in the simulation (e.g.
    LS2D); user-configured profiles take precedence. Each variable's time
    series is linearly interpolated onto the union of all time points.
    """

    def __init__(self, module: do_openboundary) -> None:
        self.module = module

    def prepare(self) -> tuple[xr.Dataset, xr.Dataset]:
        forcings = self._collect_forcings()
        profile_mapping = self._build_mapping()
        mapping = self._enforce_nudging_mapping(dict(profile_mapping))

        profiles_1d = self._extract_openbc_base_profiles(mapping, forcings)
        init_profiles_1d = self._extract_openbc_base_profiles(profile_mapping, forcings)
        ds, boundaries, base_vars = self._build_atmosphere_boundaries_dataset(
            mapping,
            profiles_1d,
            forcings,
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

        initfields = self._build_initfields_dataset(init_profiles_1d)
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

    def _collect_forcings(self) -> ForcingSet:
        atmo_module = self.module.nest_in_atmosphere.atmosphere_module
        if atmo_module is None:
            raise ValueError(
                "Nest_in_AtmosphereProfiles requires 'atmosphere_module' to be set; "
                "using AtmosphereModule instances from dales_simulation is not supported."
            )
        if atmo_module.sim is None or atmo_module.sim.grid is None:
            raise ValueError(
                "AtmosphereModule must be associated with a dales_simulation with a defined grid before preparing open boundary profiles."
            )

        if isinstance(atmo_module, AtmosphereModule):
            atmo_module.ensure_prepared()
            return atmo_module.forcings

        # Any other provider (e.g. an external LS2DAtmosphereModule) is used directly.
        if not atmo_module.check_settings_done:
            atmo_module.check_settings()
            atmo_module.check_settings_done = True
        return collect_forcings([atmo_module])

    def _build_mapping(self) -> dict[str, str]:
        mapping = dict(self.module.nest_in_atmosphere.variable_mapping or {})
        mapping.setdefault("u", "ua")
        mapping.setdefault("v", "va")
        mapping.setdefault("w", "w")
        mapping.setdefault("thl", "thetal")
        mapping.setdefault("qt", "qt")
        mapping.setdefault("e12", "tke")
        return mapping

    def _enforce_nudging_mapping(
        self,
        mapping: dict[str, str],
    ) -> dict[str, str]:
        """Use the nudging target of each mapped state variable for the boundaries."""
        nudging_target_of = {
            var.fallback: var.name
            for var in get_all_vars()
            if var.init_time_height and var.fallback
        }
        return {
            obc_var: nudging_target_of.get(atmo_name, atmo_name)
            for obc_var, atmo_name in mapping.items()
        }

    def _lookup(
        self,
        forcings: ForcingSet,
        atmo_name: str,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        vars_by_name = get_var_by_name()
        if atmo_name not in vars_by_name:
            raise ValueError(f"Unknown atmosphere variable '{atmo_name}' in mapping")
        series = forcings.series.get(atmo_name)
        initial = forcings.initial.get(atmo_name)
        fallback = vars_by_name[atmo_name].fallback
        if series is None and initial is None and fallback is not None:
            return self._lookup(forcings, fallback)
        return self._on_openbc_grid(series), self._on_openbc_grid(initial)

    def _on_openbc_grid(self, da: xr.DataArray | None) -> xr.DataArray | None:
        if da is None:
            return None
        zt = self._grid_coord("zt").values
        if da.sizes.get(Z_DIM) != zt.size:
            raise ValueError(
                f"Profile '{da.name}' has {da.sizes.get(Z_DIM)} levels, expected {zt.size} (len(grid.zt))"
            )
        return da.assign_coords({Z_DIM: zt})

    def _extract_openbc_base_profiles(
        self,
        mapping: dict[str, str],
        forcings: ForcingSet,
    ) -> dict[str, xr.DataArray]:
        profiles_1d: dict[str, xr.DataArray] = {}
        for obc_var, atmo_name in mapping.items():
            series, initial = self._lookup(forcings, atmo_name)
            if initial is not None:
                profiles_1d[obc_var] = initial
            elif series is not None:
                profiles_1d[obc_var] = value_at(series, 0.0)
                logger.info(
                    "_prepare_from_atmosphere: no base profile for '%s'; using t=0 of its time series",
                    atmo_name,
                )
            else:
                raise ValueError(
                    f"Atmosphere variable '{atmo_name}' (for '{obc_var}') is not provided by any module"
                )
        return profiles_1d

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
        mapping: dict[str, str],
        profiles_1d: dict[str, xr.DataArray],
        forcings: ForcingSet,
    ) -> tuple[xr.Dataset, list[str], list[str]]:
        series_by_var = {
            obc_var: self._lookup(forcings, atmo_name)[0]
            for obc_var, atmo_name in mapping.items()
        }
        all_times = resolve_time_axis(
            [series for series in series_by_var.values() if series is not None]
        )

        base_time_str = self.module.time0 or self.module.start
        if base_time_str is None:
            raise ValueError(
                "Nest_in_AtmosphereProfiles requires 'time0' or 'start' to be set on do_openboundary"
            )
        time_points = np.datetime64(base_time_str) + np.round(all_times).astype(
            "timedelta64[s]"
        )

        boundaries = ["west", "east", "south", "north", "top"]
        base_vars = ["u", "v", "w", "thl", "qt", "e12"]
        for var in mapping:
            if var not in base_vars:
                logger.info("Adding variable '%s' to base_vars", var)
                base_vars.append(var)

        add_to_top_thl = getattr(self.module.nest_in_atmosphere, "add_to_top_thl", None)
        ds = xr.Dataset()
        for var in base_vars:
            series = series_by_var.get(var)
            if series is None:
                series = profiles_1d[var].expand_dims({TIME_DIM: [0.0]})
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
