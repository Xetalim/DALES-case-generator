"""Worker for building open-boundary fields from atmosphere profiles."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from modular_dales.Atmosphere import AtmosphereModule
from modular_dales.Atmosphere.ls2d_atmosphere import LS2DAtmosphereModule
from modular_dales.LBC.nest_dales_in_dales import boundary_fields_fine
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.vars import get_var_by_name

if TYPE_CHECKING:
    from modular_dales.LBC.openbc import do_openboundary


logger = logging.getLogger(__name__)


class OpenBCAtmosphereWorker:
    """Build open boundaries from an AtmosphereModule with optional LS2D overrides.

    Precedence rules for time-dependent series:
    1. AtmosphereModule timed forcings are the baseline.
    2. Explicit AtmosphereModule nudging profiles (base or timed) are authoritative.
    3. LS2D nudging fields only fill missing nudging series.
    """

    def __init__(self, module: "do_openboundary") -> None:
        self.module = module

    def prepare(self) -> Tuple[xr.Dataset, xr.Dataset]:
        atmo_module, ls2d_module = self._prepare_modules()
        profile_mapping = self._build_mapping()
        mapping = dict(profile_mapping)
        mapping = self._enforce_nudging_mapping(mapping)

        timed_forcings_by_name = self._collect_atmo_timed_forcings_by_name(atmo_module)
        self._merge_ls2d_timed_forcings(
            timed_forcings_by_name,
            ls2d_module,
            atmo_module,
        )

        profiles_1d = self._extract_openbc_base_profiles(
            mapping,
            atmo_module,
            timed_forcings_by_name,
        )
        init_profiles_1d = self._extract_openbc_base_profiles(
            profile_mapping,
            atmo_module,
            timed_forcings_by_name,
        )
        ds, boundaries, base_vars = self._build_atmosphere_boundaries_dataset(
            mapping,
            profiles_1d,
            timed_forcings_by_name,
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
        profiles_1d: Dict[str, np.ndarray],
    ) -> xr.Dataset:
        coords = {
            "xt": ("xt", self.module.openBCgrid.xt),
            "xm": ("xm", self.module.openBCgrid.xm),
            "yt": ("yt", self.module.openBCgrid.yt),
            "ym": ("ym", self.module.openBCgrid.ym),
            "zt": ("zt", self.module.openBCgrid.zt),
            "zm": ("zm", self.module.openBCgrid.zm),
        }

        data_vars = {
            "u0": (
                ("zt", "yt", "xm"),
                self._broadcast_profile(
                    np.asarray(profiles_1d["u"], dtype=float),
                    ("zt", "yt", "xm"),
                    coords,
                ),
            ),
            "v0": (
                ("zt", "ym", "xt"),
                self._broadcast_profile(
                    np.asarray(profiles_1d["v"], dtype=float),
                    ("zt", "ym", "xt"),
                    coords,
                ),
            ),
            "w0": (
                ("zm", "yt", "xt"),
                self._broadcast_profile(
                    self._interpolate_profile(
                        np.asarray(profiles_1d["w"], dtype=float),
                        self.module.openBCgrid.zt,
                        self.module.openBCgrid.zm,
                    ),
                    ("zm", "yt", "xt"),
                    coords,
                ),
            ),
            "thl0": (
                ("zt", "yt", "xt"),
                self._broadcast_profile(
                    np.asarray(profiles_1d["thl"], dtype=float),
                    ("zt", "yt", "xt"),
                    coords,
                ),
            ),
            "qt0": (
                ("zt", "yt", "xt"),
                self._broadcast_profile(
                    np.asarray(profiles_1d["qt"], dtype=float),
                    ("zt", "yt", "xt"),
                    coords,
                ),
            ),
            "e120": (
                ("zt", "yt", "xt"),
                self._broadcast_profile(
                    np.asarray(profiles_1d["e12"], dtype=float),
                    ("zt", "yt", "xt"),
                    coords,
                ),
            ),
        }

        initfields = xr.Dataset(data_vars=data_vars, coords=coords)
        initfields = initfields.assign_attrs(
            {
                "history": f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                "author": "author",
                "time0": self.module.time0,
            }
        )
        return initfields

    def _interpolate_profile(
        self,
        values: np.ndarray,
        source_z: np.ndarray,
        target_z: np.ndarray,
    ) -> np.ndarray:
        src_z = np.asarray(source_z, dtype=float)
        prof = np.asarray(values, dtype=float)
        if src_z.shape[0] != prof.shape[0]:
            raise ValueError(
                f"Vertical coordinate length {src_z.shape[0]} does not match profile length {prof.shape[0]}"
            )

        sort_idx = np.argsort(src_z)
        src_z = src_z[sort_idx]
        prof = prof[sort_idx]
        return np.interp(np.asarray(target_z, dtype=float), src_z, prof)

    def _broadcast_profile(
        self,
        profile: np.ndarray,
        dims: Tuple[str, ...],
        coords: Dict[str, Tuple[str, np.ndarray]],
    ) -> np.ndarray:
        shape = [len(coords[d][1]) for d in dims]
        out = np.zeros(shape, dtype=float)
        if "zm" in dims:
            vertical_dim = "zm"
        else:
            vertical_dim = "zt"
        kdim = dims.index(vertical_dim)
        for k in range(len(profile)):
            idx = [slice(None)] * len(shape)
            idx[kdim] = k
            out[tuple(idx)] = profile[k]
        return out

    def _prepare_modules(
        self,
    ) -> Tuple[AtmosphereModule, Optional[LS2DAtmosphereModule]]:
        atmo_module: Optional[AtmosphereModule] = (
            self.module.nest_in_atmosphere.atmosphere_module
        )
        if atmo_module is None:
            raise ValueError(
                "Nest_in_AtmosphereProfiles requires 'atmosphere_module' to be set; "
                "using AtmosphereModule instances from dales_simulation is not supported."
            )
        if atmo_module.sim is None or atmo_module.sim.grid is None:
            raise ValueError(
                "AtmosphereModule must be associated with a dales_simulation with a defined grid before preparing open boundary profiles."
            )

        ls2d_module: Optional[LS2DAtmosphereModule] = None
        if isinstance(atmo_module, LS2DAtmosphereModule):
            ls2d_module = atmo_module
        if self.module.module_exists(LS2DAtmosphereModule):
            ls2d_module = self.module.retrieve_module(LS2DAtmosphereModule)
            if not ls2d_module.prepare_calculation_done:
                ls2d_module.prepare_calculation()
                ls2d_module.prepare_calculation_done = True

        if not atmo_module.prepare_calculation_done:
            atmo_module.prepare_calculation()
            atmo_module.prepare_calculation_done = True

        return atmo_module, ls2d_module

    def _build_mapping(self) -> Dict[str, str]:
        mapping = dict(self.module.nest_in_atmosphere.variable_mapping or {})
        mapping.setdefault("u", "ua")
        mapping.setdefault("v", "va")
        mapping.setdefault("w", "w")
        mapping.setdefault("thl", "thetal")
        mapping.setdefault("qt", "qt")
        mapping.setdefault("e12", "tke")
        return mapping

    def _collect_atmo_timed_forcings_by_name(
        self,
        atmo_module: AtmosphereModule,
    ) -> Dict[str, Dict[float, np.ndarray]]:
        timed_forcings = atmo_module.get_timedep_atmosphere_forcings() or {}
        timed_forcings_by_name: Dict[str, Dict[float, np.ndarray]] = {}
        for key, series in timed_forcings.items():
            if isinstance(key, str):
                name = key
            else:
                name = getattr(key, "name", None)
            if name is None:
                continue
            target = timed_forcings_by_name.setdefault(name, {})
            for time_value, values in series.items():
                target[float(time_value)] = np.asarray(values, dtype=float)
        return timed_forcings_by_name

    def _enforce_nudging_mapping(
        self,
        mapping: Dict[str, str],
    ) -> Dict[str, str]:
        nudge_mapping = {
            "u": "ua_nudge",
            "v": "va_nudge",
            "w": "wa_nudge",
            "thl": "thl_nudge",
            "qt": "qt_nudge",
        }

        resolved_mapping = dict(mapping)
        for obc_var, nudge_name in nudge_mapping.items():
            if obc_var not in resolved_mapping:
                continue
            resolved_mapping[obc_var] = nudge_name

        return resolved_mapping

    def _normalize_ls2d_time_height(
        self,
        raw: Any,
        n_times: int,
        n_levels: int,
        field_name: str,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(raw, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim != 2:
            logger.warning(
                "_prepare_from_atmosphere: unexpected LS2D shape for '%s': %s, skipping",
                field_name,
                arr.shape,
            )
            return None
        if arr.shape == (n_times, n_levels):
            return arr
        if arr.shape == (n_levels, n_times):
            return arr.T
        logger.warning(
            "_prepare_from_atmosphere: unexpected LS2D shape for '%s': %s, skipping",
            field_name,
            arr.shape,
        )
        return None

    def _merge_ls2d_timed_forcings(
        self,
        timed_forcings_by_name: Dict[str, Dict[float, np.ndarray]],
        ls2d_module: Optional[LS2DAtmosphereModule],
        atmo_module: AtmosphereModule,
    ) -> None:
        if ls2d_module is None:
            return

        ls2d_times = list(getattr(ls2d_module, "_times_with_zero", []))
        if not ls2d_times:
            return

        n_levels = len(self.module.openBCgrid.zt)
        n_times = len(ls2d_times)

        shaped_profiles = getattr(atmo_module, "shaped_profiles", [])
        interpolated_profiles = getattr(atmo_module, "interpolated_profiles", [])
        timed_profiles = getattr(atmo_module, "timed_profiles", [])

        explicit_nudging_names = set()
        explicit_nudging_names.update(
            profile.variable.name
            for profile in shaped_profiles + interpolated_profiles
            if profile.variable.name
            in {"ua_nudge", "va_nudge", "wa_nudge", "thl_nudge", "qt_nudge"}
        )
        explicit_nudging_names.update(
            timed_profile.profile.variable.name
            for timed_profile in timed_profiles
            if timed_profile.profile.variable.name
            in {"ua_nudge", "va_nudge", "wa_nudge", "thl_nudge", "qt_nudge"}
        )

        nudging_profiles = getattr(ls2d_module, "_nudging_var_dic", {})
        nudging_name_map = {
            "ua": "ua_nudge",
            "va": "va_nudge",
            "wa": "wa_nudge",
            "thetal": "thl_nudge",
            "qt": "qt_nudge",
        }
        for source_name, target_name in nudging_name_map.items():
            # Respect explicit AtmosphereModule configuration for this nudging target.
            if target_name in explicit_nudging_names:
                continue
            # Only fill missing timed series; never overwrite existing forcings.
            if target_name in timed_forcings_by_name:
                continue
            raw = nudging_profiles.get(source_name)
            arr = self._normalize_ls2d_time_height(
                raw,
                n_times,
                n_levels,
                source_name,
            )
            if arr is None:
                continue
            timed_forcings_by_name[target_name] = {
                float(time_value): arr[idx, :]
                for idx, time_value in enumerate(ls2d_times)
            }

        if ls2d_module.les_input is None:
            return

        ls2d_state = {
            "ua_nudge": "u",
            "va_nudge": "v",
            "wa_nudge": "wls",
            "thl_nudge": "thl",
            "qt_nudge": "qt",
        }
        for target_name, les_field in ls2d_state.items():
            if target_name in explicit_nudging_names:
                continue
            if target_name in timed_forcings_by_name:
                continue
            if not hasattr(ls2d_module.les_input, les_field):
                continue
            raw = getattr(ls2d_module.les_input, les_field).values
            arr = self._normalize_ls2d_time_height(raw, n_times, n_levels, les_field)
            if arr is None:
                continue
            timed_forcings_by_name[target_name] = {
                float(time_value): arr[idx, :]
                for idx, time_value in enumerate(ls2d_times)
            }

    def _extract_openbc_base_profiles(
        self,
        mapping: Dict[str, str],
        atmo_module: AtmosphereModule,
        timed_forcings_by_name: Dict[str, Dict[float, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        profiles_1d: Dict[str, np.ndarray] = {}
        vars_by_name = get_var_by_name()
        atmo_variables = getattr(atmo_module, "variables", None)
        for obc_var, atmo_name in mapping.items():
            if atmo_name not in vars_by_name:
                raise ValueError(
                    f"Unknown atmosphere variable '{atmo_name}' in mapping for '{obc_var}'"
                )
            atmo_definition = vars_by_name[atmo_name]
            profile = None
            if atmo_variables is not None and atmo_definition in atmo_variables:
                var_container = atmo_variables[atmo_definition]
                if var_container.values is not None:
                    profile = np.asarray(var_container.values, dtype=float)

            # For LS2D-driven nudging boundaries, prefer explicit evaluated
            # nudging profiles over timed-series fallback.
            if profile is None and isinstance(atmo_module, LS2DAtmosphereModule):
                if atmo_name in {
                    "ua_nudge",
                    "va_nudge",
                    "wa_nudge",
                    "thl_nudge",
                    "qt_nudge",
                }:
                    profile = self._extract_ls2d_profile(atmo_module, atmo_name)

            if profile is not None:
                pass
            elif (
                atmo_name in timed_forcings_by_name
                and 0.0 in timed_forcings_by_name[atmo_name]
            ):
                profile = np.asarray(
                    timed_forcings_by_name[atmo_name][0.0], dtype=float
                )
                logger.info(
                    "_prepare_from_atmosphere: base profile for '%s' not evaluated; using t=0 of timed series as fallback",
                    atmo_name,
                )
            elif isinstance(atmo_module, LS2DAtmosphereModule):
                profile = self._extract_ls2d_profile(atmo_module, atmo_name)
            else:
                raise ValueError(
                    f"AtmosphereModule variable '{atmo_name}' has no evaluated values; "
                    "ensure AtmosphereModule.prepare_calculation() has run."
                )
            profiles_1d[obc_var] = profile

        nz = len(self.module.openBCgrid.zt)
        for obc_var, profile in profiles_1d.items():
            if profile.shape[0] != nz:
                raise ValueError(
                    f"Profile for '{obc_var}' has length {profile.shape[0]}, expected {nz} (len(grid.zt))"
                )

        return profiles_1d

    def _extract_ls2d_profile(
        self,
        ls2d_module: LS2DAtmosphereModule,
        atmo_name: str,
    ) -> np.ndarray:
        zt = np.asarray(self.module.openBCgrid.zt, dtype=float)
        nz = len(zt)

        if atmo_name == "tke":
            return np.ones(nz, dtype=float) * float(ls2d_module.init_tke)

        nudging_name_map = {
            "ua_nudge": "ua",
            "va_nudge": "va",
            "wa_nudge": "wa",
            "thl_nudge": "thetal",
            "qt_nudge": "qt",
        }
        if atmo_name in nudging_name_map:
            raw_nudge = ls2d_module._nudging_var_dic.get(nudging_name_map[atmo_name])
            if raw_nudge is None:
                raise ValueError(
                    f"LS2DAtmosphereModule is missing nudging profile for '{atmo_name}'"
                )
            arr = np.asarray(raw_nudge, dtype=float)
            if arr.ndim == 2:
                if arr.shape[0] == nz:
                    return arr[:, 0]
                if arr.shape[1] == nz:
                    return arr[0, :]
            raise ValueError(
                f"Unexpected nudging profile shape for '{atmo_name}': {arr.shape}"
            )

        source_field_map = {
            "ua": "u",
            "va": "v",
            "w": "wls",
            "thetal": "thl",
            "qt": "qt",
        }
        field_name = source_field_map.get(atmo_name)
        if (
            ls2d_module.les_input is not None
            and field_name is not None
            and hasattr(ls2d_module.les_input, field_name)
        ):
            values = np.asarray(
                getattr(ls2d_module.les_input, field_name).values,
                dtype=float,
            )
            if values.ndim == 2:
                if (
                    values.shape[0] == len(ls2d_module._times_with_zero)
                    and values.shape[1] == nz
                ):
                    return values[0, :]
                if (
                    values.shape[1] == len(ls2d_module._times_with_zero)
                    and values.shape[0] == nz
                ):
                    return values[:, 0]

        raise ValueError(
            f"LS2DAtmosphereModule could not provide profile for '{atmo_name}'"
        )

    def _build_uniform_boundary(
        self,
        prof_z: np.ndarray,
        var: str,
        bnd: str,
    ) -> xr.DataArray:
        var_dims_dic = {
            "u": {"x": "xm", "y": "yt", "z": "zt"},
            "v": {"x": "xt", "y": "ym", "z": "zt"},
            "w": {"x": "xt", "y": "yt", "z": "zm"},
            "default": {"x": "xt", "y": "yt", "z": "zt"},
        }
        boundary_var_assign_dic = {
            "west": {"default": ["z", "y"]},
            "east": {"default": ["z", "y"]},
            "south": {"default": ["z", "x"]},
            "north": {"default": ["z", "x"]},
            "top": {"default": ["y", "x"]},
        }
        assign_grid_dic = {
            "xt": self.module.openBCgrid.xt,
            "xm": self.module.openBCgrid.xm,
            "yt": self.module.openBCgrid.yt,
            "ym": self.module.openBCgrid.ym,
            "zt": self.module.openBCgrid.zt,
            "zm": self.module.openBCgrid.zm,
        }

        var_dims = var_dims_dic.get(var, var_dims_dic["default"])
        base_dims = boundary_var_assign_dic[bnd].get(
            var, boundary_var_assign_dic[bnd]["default"]
        )
        dims = [var_dims[d] for d in base_dims]

        vertical_dim = None
        for dim in ("zt", "zm"):
            if dim in dims:
                vertical_dim = dim
                break

        if vertical_dim is None:
            fill_value = float(prof_z[-1])
            shape = [len(assign_grid_dic[d]) for d in dims]
            data = np.full(shape, fill_value, dtype=float)
        else:
            if vertical_dim == "zt":
                prof_on_vert = prof_z
                vert_coords = assign_grid_dic["zt"]
            else:
                prof_on_vert = np.interp(
                    assign_grid_dic["zm"], self.module.openBCgrid.zt, prof_z
                )
                vert_coords = assign_grid_dic["zm"]

            shape = [len(assign_grid_dic[d]) for d in dims]
            data = np.zeros(shape, dtype=float)
            vert_index = dims.index(vertical_dim)
            for k in range(len(vert_coords)):
                idx = [slice(None)] * len(shape)
                idx[vert_index] = k
                data[tuple(idx)] = prof_on_vert[k]

        coords = {d: assign_grid_dic[d] for d in dims}
        return xr.DataArray(data, coords=coords, dims=dims)

    def _build_atmosphere_boundaries_dataset(
        self,
        mapping: Dict[str, str],
        profiles_1d: Dict[str, np.ndarray],
        timed_forcings_by_name: Dict[str, Dict[float, np.ndarray]],
    ) -> Tuple[xr.Dataset, List[str], List[str]]:
        times_set = {0.0}
        for atmo_name in mapping.values():
            series = timed_forcings_by_name.get(atmo_name)
            if series:
                times_set.update(float(t) for t in series.keys())
        all_times = sorted(times_set)

        base_time_str = self.module.time0 or self.module.start
        if base_time_str is None:
            raise ValueError(
                "Nest_in_AtmosphereProfiles requires 'time0' or 'start' to be set on do_openboundary"
            )

        time_points = [
            np.datetime64(base_time_str) + np.timedelta64(int(round(t)), "s")
            for t in all_times
        ]
        time_index_by_seconds = {
            t: np.datetime64(base_time_str) + np.timedelta64(int(round(t)), "s")
            for t in all_times
        }

        ds = xr.Dataset(coords={"time": ("time", time_points)})
        ds = ds.assign_coords(
            {
                "xt": ("xt", self.module.openBCgrid.xt),
                "xm": ("xm", self.module.openBCgrid.xm),
                "yt": ("yt", self.module.openBCgrid.yt),
                "ym": ("ym", self.module.openBCgrid.ym),
                "zt": ("zt", self.module.openBCgrid.zt),
                "zm": ("zm", self.module.openBCgrid.zm),
            }
        )

        boundaries = ["west", "east", "south", "north", "top"]
        base_vars = ["u", "v", "w", "thl", "qt", "e12"]
        for var in mapping.keys():
            if var not in base_vars:
                logger.info("Adding variable '%s' to base_vars", var)
                base_vars.append(var)

        add_to_top_thl = getattr(self.module.nest_in_atmosphere, "add_to_top_thl", None)
        for var in base_vars:
            atmo_name = mapping[var]
            series = timed_forcings_by_name.get(atmo_name, {})
            for bnd in boundaries:
                da_list: List[xr.DataArray] = []
                for t in all_times:
                    prof_z = np.asarray(series.get(t, profiles_1d[var]), dtype=float)
                    if var == "thl" and bnd == "top" and add_to_top_thl is not None:
                        prof_z = prof_z.copy()
                        prof_z[-1] += add_to_top_thl
                    da2d = self._build_uniform_boundary(prof_z, var, bnd)
                    da3d = da2d.expand_dims({"time": [time_index_by_seconds[t]]})
                    da_list.append(da3d)
                ds[f"{var}{bnd}"] = xr.concat(da_list, dim="time")

        return ds, boundaries, base_vars

    def _apply_atmosphere_boundary_noise(
        self,
        ds: xr.Dataset,
        boundaries: List[str],
        base_vars: List[str],
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
