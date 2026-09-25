import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.ls2d_atmosphere import LS2DAtmosphereModule
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.LBC.nest_dales_in_HARMONIE import prep_harmonie
from modular_dales.LBC.nest_dales_in_HARMONIE.nest_dales_in_KNMI import (
    KNMIPrepper,
)
from modular_dales.LBC.nest_dales_in_HARMONIE.knmi_harmonie_download import (
    KNMIHarmonieForecastDownloadModule,
    resolve_knmi_harmonie_download_module,
)
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class HarmonieAtmosphereModule(LS2DAtmosphereModule):
    """LS2D-like atmosphere forcing module sourced from Harmonie NetCDF.

    This module runs the existing Harmonie preprocessing pipeline and then
    converts domain-mean Harmonie fields to an LS2D-like ``les_input`` object.
    The downstream forcing/profile generation reuses the same logic as
    ``LS2DAtmosphereModule`` so that output structure matches LS2D-driven runs.
    """

    harmonie_ml_glob: Optional[str] = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_sfc_glob: Optional[str] = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_start: Optional[str] = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_end: Optional[str] = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_time0: Optional[str] = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_tchunk: int = field(
        default=1,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )

    use_harmonie_vertical_velocity: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    use_wind_as_geostrophic: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    infer_adv_tendencies: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "HarmonieAtmosphereModule"

    def check_settings(self):
        if self.grid is None:
            raise ValueError(
                "HarmonieAtmosphereModule requires a GridDales grid (sim.grid) to be set"
            )

        knmi_download_module = None
        if self.sim is not None and self.sim.module_exists(
            KNMIHarmonieForecastDownloadModule
        ):
            knmi_download_module = self.sim.retrieve_module(
                KNMIHarmonieForecastDownloadModule
            )

        missing = []
        for name in (
            "harmonie_ml_glob",
            "harmonie_sfc_glob",
            "harmonie_start",
            "harmonie_end",
            "harmonie_time0",
        ):
            if getattr(self, name) in (None, ""):
                missing.append(name)

        if missing and knmi_download_module is None:
            raise ValueError(
                "HarmonieAtmosphereModule missing required settings: "
                + ", ".join(missing)
            )

        if int(self.harmonie_tchunk) <= 0:
            raise ValueError("HarmonieAtmosphereModule.harmonie_tchunk must be > 0")

        return None

    def prepare_calculation(self):
        if self.grid is None:
            raise ValueError(
                "HarmonieAtmosphereModule requires grid to be initialized before prepare_calculation"
            )

        knmi_download_module = None
        if self.sim is not None:
            knmi_download_module = resolve_knmi_harmonie_download_module(self.sim)

        if knmi_download_module is not None:
            if not self.harmonie_ml_glob:
                self.harmonie_ml_glob = knmi_download_module.ml_glob
            if not self.harmonie_sfc_glob:
                self.harmonie_sfc_glob = knmi_download_module.sfc_glob

        openbc_grid = (
            self.grid.as_openbc() if hasattr(self.grid, "as_openbc") else self.grid
        )

        config = OpenBoundaryConfig(
            start=str(self.harmonie_start),
            time0=str(self.harmonie_time0),
            end=str(self.harmonie_end),
            KNMI_ml_glob=str(self.harmonie_ml_glob),
            KNMI_sfc_glob=str(self.harmonie_sfc_glob),
            tchunk=int(self.harmonie_tchunk),
        )

        logger.info(
            "HarmonieAtmosphereModule: preprocessing KNMI NetCDF via KNMIPrepper"
        )
        prepper = KNMIPrepper(config, openbc_grid)
        prepper.load_data()
        data, _ = prepper.prep_harmonie()

        z_dales = np.asarray(self.grid.zt, dtype=float)
        les_input = self._build_harmonie_les_input(data, z_dales, prepper)
        self.les_input = les_input

        self._prepare_from_les_input(les_input, z_dales)
        return None

    def _build_harmonie_les_input(
        self,
        data: xr.Dataset,
        z_dales: np.ndarray,
        prepper: prep_harmonie.harmoniePrepper,
    ) -> xr.Dataset:
        required = ("u", "v", "thl", "qt", "z", "time")
        missing = [
            name for name in required if name not in data and name not in data.coords
        ]
        if missing:
            raise ValueError(
                "HarmonieAtmosphereModule: missing required Harmonie fields: "
                + ", ".join(missing)
            )

        profile_ds = data
        for dim in ("x", "y"):
            if dim in profile_ds.dims:
                profile_ds = profile_ds.mean(dim=dim, skipna=True)

        times = np.asarray(profile_ds["time"].values)
        if times.ndim != 1 or times.size < 2:
            raise ValueError(
                "HarmonieAtmosphereModule expects at least 2 time samples in Harmonie data"
            )
        times_sec = (
            (times.astype("datetime64[ns]") - times[0].astype("datetime64[ns]"))
            / np.timedelta64(1, "s")
        ).astype(float)

        z_src = np.asarray(profile_ds["z"].values, dtype=float)
        if z_src.ndim != 1 or z_src.size < 2:
            raise ValueError(
                "HarmonieAtmosphereModule expects a 1D vertical coordinate 'z' with at least 2 levels"
            )

        order = np.argsort(z_src)
        z_src_sorted = z_src[order]

        def _interp_profile(name: str) -> np.ndarray:
            if name not in profile_ds:
                raise ValueError(
                    f"HarmonieAtmosphereModule missing profile variable '{name}'"
                )
            da = profile_ds[name]
            if "time" not in da.dims or "z" not in da.dims:
                raise ValueError(
                    f"HarmonieAtmosphereModule expects '{name}' to have dimensions including time and z"
                )
            arr = np.asarray(da.transpose("time", "z").values, dtype=float)
            out = np.zeros((arr.shape[0], z_dales.size), dtype=float)
            for t_idx in range(arr.shape[0]):
                out[t_idx, :] = np.interp(
                    z_dales,
                    z_src_sorted,
                    arr[t_idx, order],
                    left=arr[t_idx, order][0],
                    right=arr[t_idx, order][-1],
                )
            return out

        u_tz = _interp_profile("u")
        v_tz = _interp_profile("v")
        thl_tz = _interp_profile("thl")
        qt_tz = _interp_profile("qt")

        if self.use_harmonie_vertical_velocity and "w" in profile_ds:
            wls_tz = _interp_profile("w")
        else:
            wls_tz = np.zeros_like(u_tz)

        if self.use_wind_as_geostrophic:
            geostrophic = self._compute_geostrophic_profiles_constant_pressure(
                prepper, z_dales
            )
            if geostrophic is None:
                geostrophic = self._compute_geostrophic_profiles(data)
            if geostrophic is not None:
                ug_prof, vg_prof = geostrophic
                # KNMIPrepper/prep_harmonie profiles may live on a different
                # vertical dimension than DALES. Map geostrophic profiles to
                # z_dales explicitly instead of assigning onto profile_ds['z'].
                if ug_prof.ndim == 2 and vg_prof.ndim == 2:
                    if (
                        ug_prof.shape[1] == z_dales.size
                        and vg_prof.shape[1] == z_dales.size
                    ):
                        ug_tz = ug_prof
                        vg_tz = vg_prof
                    elif (
                        ug_prof.shape[1] == z_src.size
                        and vg_prof.shape[1] == z_src.size
                    ):
                        ug_tz = np.zeros((ug_prof.shape[0], z_dales.size), dtype=float)
                        vg_tz = np.zeros((vg_prof.shape[0], z_dales.size), dtype=float)
                        for t_idx in range(ug_prof.shape[0]):
                            ug_tz[t_idx, :] = np.interp(
                                z_dales,
                                z_src_sorted,
                                ug_prof[t_idx, order],
                                left=ug_prof[t_idx, order][0],
                                right=ug_prof[t_idx, order][-1],
                            )
                            vg_tz[t_idx, :] = np.interp(
                                z_dales,
                                z_src_sorted,
                                vg_prof[t_idx, order],
                                left=vg_prof[t_idx, order][0],
                                right=vg_prof[t_idx, order][-1],
                            )
                    else:
                        logger.warning(
                            "HarmonieAtmosphereModule: unexpected geostrophic profile shape %s; falling back to u/v",
                            ug_prof.shape,
                        )
                        ug_tz = u_tz.copy()
                        vg_tz = v_tz.copy()
                else:
                    logger.warning(
                        "HarmonieAtmosphereModule: unexpected geostrophic profile rank; falling back to u/v"
                    )
                    ug_tz = u_tz.copy()
                    vg_tz = v_tz.copy()
            else:
                logger.warning(
                    "HarmonieAtmosphereModule: could not derive geostrophic wind from Harmonie geopotential; falling back to u/v"
                )
                ug_tz = u_tz.copy()
                vg_tz = v_tz.copy()
        else:
            ug_tz = np.zeros_like(u_tz)
            vg_tz = np.zeros_like(v_tz)

        if self.infer_adv_tendencies and times_sec.size >= 2:
            dtu_tz = np.gradient(u_tz, times_sec, axis=0, edge_order=1)
            dtv_tz = np.gradient(v_tz, times_sec, axis=0, edge_order=1)
            dtthl_tz = np.gradient(thl_tz, times_sec, axis=0, edge_order=1)
            dtqt_tz = np.gradient(qt_tz, times_sec, axis=0, edge_order=1)
        else:
            dtu_tz = np.zeros_like(u_tz)
            dtv_tz = np.zeros_like(v_tz)
            dtthl_tz = np.zeros_like(thl_tz)
            dtqt_tz = np.zeros_like(qt_tz)

        ps_value = float(prepper.ps) if prepper.ps is not None else 101325.0
        thls_value = (
            float(prepper.thls)
            if prepper.thls is not None
            else float(np.nanmean(thl_tz[:, 0]))
        )

        nt = times_sec.size
        zeros_t = np.zeros(nt, dtype=float)

        return xr.Dataset(
            data_vars={
                "u": (("time_sec", "z"), u_tz),
                "v": (("time_sec", "z"), v_tz),
                "thl": (("time_sec", "z"), thl_tz),
                "qt": (("time_sec", "z"), qt_tz),
                "wls": (("time_sec", "z"), wls_tz),
                "ug": (("time_sec", "z"), ug_tz),
                "vg": (("time_sec", "z"), vg_tz),
                "dtu_advec": (("time_sec", "z"), dtu_tz),
                "dtv_advec": (("time_sec", "z"), dtv_tz),
                "dtthl_advec": (("time_sec", "z"), dtthl_tz),
                "dtqt_advec": (("time_sec", "z"), dtqt_tz),
                "ps": (("time_sec",), np.full(nt, ps_value, dtype=float)),
                "ts": (("time_sec",), np.full(nt, thls_value, dtype=float)),
                "wth": (("time_sec",), zeros_t.copy()),
                "wq": (("time_sec",), zeros_t.copy()),
            },
            coords={
                "time_sec": times_sec,
                "z": z_dales,
            },
        )

    def _compute_geostrophic_profiles_constant_pressure(
        self,
        prepper: prep_harmonie.harmoniePrepper,
        z_dales: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Compute geostrophic wind on constant-pressure levels, then map to ``z_dales``.

        This follows the LS2D idea more closely than a direct model-level
        geopotential gradient: compute gradients on pressure surfaces first.
        """

        p_da = getattr(prepper, "p", None)
        z_da = getattr(prepper, "z", None)
        if p_da is None or z_da is None:
            return None
        if not hasattr(p_da, "dims") or not hasattr(z_da, "dims"):
            return None
        if not {"time", "lev", "y", "x"}.issubset(set(p_da.dims)):
            return None
        if not {"time", "lev", "y", "x"}.issubset(set(z_da.dims)):
            return None

        lat = self.central_lat
        if lat is None and self.grid is not None:
            lat = getattr(self.grid, "xlat", None)
        if lat is None:
            return None

        fc = 2.0 * 7.2921e-5 * np.sin(np.deg2rad(float(lat)))
        if np.isclose(fc, 0.0):
            return None

        x_vals = np.asarray(p_da.coords["x"].values, dtype=float)
        y_vals = np.asarray(p_da.coords["y"].values, dtype=float)
        if x_vals.ndim != 1 or y_vals.ndim != 1 or x_vals.size < 2 or y_vals.size < 2:
            return None
        dx = float(np.nanmean(np.diff(x_vals)))
        dy = float(np.nanmean(np.diff(y_vals)))
        if dx == 0.0 or dy == 0.0:
            return None

        p4d = np.asarray(p_da.transpose("time", "lev", "y", "x").values, dtype=float)
        z4d = np.asarray(z_da.transpose("time", "lev", "y", "x").values, dtype=float)
        if p4d.shape != z4d.shape or p4d.ndim != 4:
            return None

        nt, _nlev, ny, nx = p4d.shape

        p_ref = np.nanmean(p4d, axis=(0, 2, 3))
        finite = np.isfinite(p_ref)
        if finite.sum() < 3:
            return None
        p_ref = p_ref[finite]

        # Interpolate column heights onto common pressure surfaces.
        z_on_p = np.full((nt, p_ref.size, ny, nx), np.nan, dtype=float)
        for t in range(nt):
            for j in range(ny):
                for i in range(nx):
                    p_col = p4d[t, :, j, i]
                    z_col = z4d[t, :, j, i]
                    mask = np.isfinite(p_col) & np.isfinite(z_col)
                    if mask.sum() < 3:
                        continue
                    p_valid = p_col[mask]
                    z_valid = z_col[mask]
                    order = np.argsort(p_valid)
                    p_sorted = p_valid[order]
                    z_sorted = z_valid[order]
                    p_unique, uniq_idx = np.unique(p_sorted, return_index=True)
                    z_unique = z_sorted[uniq_idx]
                    if p_unique.size < 3:
                        continue
                    z_on_p[t, :, j, i] = np.interp(
                        p_ref,
                        p_unique,
                        z_unique,
                        left=np.nan,
                        right=np.nan,
                    )

        if not np.isfinite(z_on_p).any():
            return None

        dzdx = np.gradient(z_on_p, axis=3) / dx
        dzdy = np.gradient(z_on_p, axis=2) / dy
        ug_p = -(9.81 / fc) * dzdy
        vg_p = (9.81 / fc) * dzdx

        ug_p_mean = np.nanmean(ug_p, axis=(2, 3))
        vg_p_mean = np.nanmean(vg_p, axis=(2, 3))
        z_p_mean = np.nanmean(z_on_p, axis=(2, 3))

        ug_tz = np.full((nt, z_dales.size), np.nan, dtype=float)
        vg_tz = np.full((nt, z_dales.size), np.nan, dtype=float)
        for t in range(nt):
            z_line = z_p_mean[t, :]
            ug_line = ug_p_mean[t, :]
            vg_line = vg_p_mean[t, :]
            mask = np.isfinite(z_line) & np.isfinite(ug_line) & np.isfinite(vg_line)
            if mask.sum() < 3:
                continue
            z_valid = z_line[mask]
            ug_valid = ug_line[mask]
            vg_valid = vg_line[mask]
            order = np.argsort(z_valid)
            z_sorted = z_valid[order]
            ug_sorted = ug_valid[order]
            vg_sorted = vg_valid[order]
            ug_tz[t, :] = np.interp(
                z_dales,
                z_sorted,
                ug_sorted,
                left=ug_sorted[0],
                right=ug_sorted[-1],
            )
            vg_tz[t, :] = np.interp(
                z_dales,
                z_sorted,
                vg_sorted,
                left=vg_sorted[0],
                right=vg_sorted[-1],
            )

        if not np.isfinite(ug_tz).any() or not np.isfinite(vg_tz).any():
            return None

        # Fill occasional NaN time rows by nearest valid profile.
        valid_rows = np.where(
            np.isfinite(ug_tz).all(axis=1) & np.isfinite(vg_tz).all(axis=1)
        )[0]
        if valid_rows.size == 0:
            return None
        for t in range(nt):
            if np.isfinite(ug_tz[t]).all() and np.isfinite(vg_tz[t]).all():
                continue
            nearest = valid_rows[np.abs(valid_rows - t).argmin()]
            ug_tz[t, :] = ug_tz[nearest, :]
            vg_tz[t, :] = vg_tz[nearest, :]

        return ug_tz, vg_tz

    def _compute_geostrophic_profiles(
        self,
        data: xr.Dataset,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Compute geostrophic wind from horizontal geopotential gradients.

        This mirrors the LS2D strategy: derive geostrophic wind from height
        gradients and Coriolis, then average over the target area.
        """

        if "z3d" not in data:
            return None
        if "x" not in data.coords or "y" not in data.coords:
            return None
        if not {"time", "z", "y", "x"}.issubset(data["z3d"].dims):
            return None

        lat = self.central_lat
        if lat is None and self.grid is not None:
            lat = getattr(self.grid, "xlat", None)
        if lat is None:
            return None

        fc = 2.0 * 7.2921e-5 * np.sin(np.deg2rad(float(lat)))
        if np.isclose(fc, 0.0):
            return None

        x = np.asarray(data["x"].values, dtype=float)
        y = np.asarray(data["y"].values, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
            return None

        dx = float(np.nanmean(np.diff(x)))
        dy = float(np.nanmean(np.diff(y)))
        if dx == 0.0 or dy == 0.0:
            return None

        z3d = np.asarray(data["z3d"].values, dtype=float)
        if z3d.ndim != 4:
            return None

        # Geostrophic wind from geopotential height gradients.
        dzdx = np.gradient(z3d, axis=3) / dx
        dzdy = np.gradient(z3d, axis=2) / dy

        ug = -(9.81 / fc) * dzdy
        vg = (9.81 / fc) * dzdx

        ug_mean = np.nanmean(ug, axis=(2, 3))
        vg_mean = np.nanmean(vg, axis=(2, 3))
        return ug_mean, vg_mean
