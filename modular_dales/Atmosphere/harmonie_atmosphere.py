import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.external_forcing import ExternalForcingModule
from modular_dales.Atmosphere.les_input import LESInput
from modular_dales.LBC.nest_dales_in_HARMONIE import prep_harmonie
from modular_dales.LBC.nest_dales_in_HARMONIE.knmi_harmonie_download import (
    KNMIHarmonieForecastDownloadModule,
    resolve_knmi_harmonie_download_module,
)
from modular_dales.LBC.nest_dales_in_HARMONIE.nest_dales_in_KNMI import (
    KNMIPrepper,
)
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class HarmonieAtmosphereModule(ExternalForcingModule):
    """Atmosphere forcing sourced from Harmonie NetCDF.

    Runs the Harmonie preprocessing pipeline and converts domain-mean fields
    to an LS2D-like ``les_input``; profiles, nudging and forcings are then
    provided exactly as for :class:`LS2DAtmosphereModule`.
    """

    harmonie_ml_glob: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_sfc_glob: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_start: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_end: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    harmonie_time0: str | None = field(
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
    _sounding: xr.Dataset | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )

    def check_settings(self):
        super().check_settings()

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

    def load_les_input(self) -> LESInput:
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
        self._sounding = prepper.backrad_profile

        return self._build_harmonie_les_input(data, prepper)

    def backrad_sounding(self) -> xr.Dataset | None:
        return self._sounding

    def _build_harmonie_les_input(
        self,
        data: xr.Dataset,
        prepper: prep_harmonie.harmoniePrepper,
    ) -> LESInput:
        required = ("u", "v", "thl", "qt", "z", "time")
        missing = [
            name for name in required if name not in data and name not in data.coords
        ]
        if missing:
            raise ValueError(
                "HarmonieAtmosphereModule: missing required Harmonie fields: "
                + ", ".join(missing)
            )

        profile_ds = data.mean([d for d in ("x", "y") if d in data.dims], skipna=True)
        if profile_ds.sizes.get("time", 0) < 2:
            raise ValueError(
                "HarmonieAtmosphereModule expects at least 2 time samples in Harmonie data"
            )
        if "z" not in profile_ds.dims or profile_ds.sizes["z"] < 2:
            raise ValueError(
                "HarmonieAtmosphereModule expects a vertical dimension 'z' with at least 2 levels"
            )
        times_sec = (
            (profile_ds["time"] - profile_ds["time"][0]) / np.timedelta64(1, "s")
        ).values.astype(float)

        def at_dales_heights(name: str) -> xr.DataArray:
            if name not in profile_ds:
                raise ValueError(
                    f"HarmonieAtmosphereModule missing profile variable '{name}'"
                )
            da = profile_ds[name]
            if not {"time", "z"} <= set(da.dims):
                raise ValueError(
                    f"HarmonieAtmosphereModule expects '{name}' to have dimensions including time and z"
                )
            return _to_heights(da, "z", self.zt).assign_coords(time=times_sec)

        u = at_dales_heights("u")
        v = at_dales_heights("v")
        thl = at_dales_heights("thl")
        qt = at_dales_heights("qt")
        if self.use_harmonie_vertical_velocity and "w" in profile_ds:
            wls = at_dales_heights("w")
        else:
            wls = xr.zeros_like(u)

        if self.use_wind_as_geostrophic:
            ug, vg = self._geostrophic_wind_at_heights(data, prepper, u, v)
        else:
            ug, vg = xr.zeros_like(u), xr.zeros_like(v)

        fields = {"u": u, "v": v, "thl": thl, "qt": qt, "wls": wls, "ug": ug, "vg": vg}
        for name in ("u", "v", "thl", "qt"):
            fields[f"dt{name}_advec"] = (
                fields[name].differentiate("time")
                if self.infer_adv_tendencies
                else xr.zeros_like(fields[name])
            )

        surface = u.isel(z=0, drop=True)
        ps_value = float(prepper.ps) if prepper.ps is not None else 101325.0
        thls_value = (
            float(prepper.thls)
            if prepper.thls is not None
            else float(thl.isel(z=0).mean())
        )
        fields.update(
            ps=xr.full_like(surface, ps_value),
            ts=xr.full_like(surface, thls_value),
        )
        fields.update(_surface_fluxes(profile_ds, prepper, times_sec))
        return xr.Dataset(fields).assign(
            time_sec=xr.DataArray(
                times_sec,
                dims=("time",),
                attrs={"long_name": "Seconds since simulation start", "units": "s"},
            )
        )

    def _coriolis_parameter(self) -> float | None:
        lat = self.central_lat
        if lat is None and self.grid is not None:
            lat = getattr(self.grid, "xlat", None)
        if lat is None:
            return None
        fc = 2.0 * 7.2921e-5 * np.sin(np.deg2rad(float(lat)))
        return None if np.isclose(fc, 0.0) else fc

    def _geostrophic_wind_at_heights(
        self,
        data: xr.Dataset,
        prepper: prep_harmonie.harmoniePrepper,
        u: xr.DataArray,
        v: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Geostrophic wind on ``(time, zt)``; falls back to ``u``/``v``."""
        geostrophic = self._compute_geostrophic_profiles_constant_pressure(prepper)
        if geostrophic is None:
            geostrophic = self._compute_geostrophic_profiles(data)
            if geostrophic is not None:
                geostrophic = tuple(_to_heights(g, "z", self.zt) for g in geostrophic)
        if geostrophic is None:
            logger.warning(
                "HarmonieAtmosphereModule: could not derive geostrophic wind from Harmonie geopotential; falling back to u/v"
            )
            return u.copy(), v.copy()
        if any(g.sizes["time"] != u.sizes["time"] for g in geostrophic):
            logger.warning(
                "HarmonieAtmosphereModule: geostrophic wind has %d times, expected %d; falling back to u/v",
                geostrophic[0].sizes["time"],
                u.sizes["time"],
            )
            return u.copy(), v.copy()
        ug, vg = (
            g.transpose("time", "z").assign_coords(time=u["time"]) for g in geostrophic
        )
        return ug, vg

    def _compute_geostrophic_profiles_constant_pressure(
        self,
        prepper: prep_harmonie.harmoniePrepper,
    ) -> tuple[xr.DataArray, xr.DataArray] | None:
        """Compute geostrophic wind on constant-pressure levels, then map to :attr:`zt`.

        This follows the LS2D idea more closely than a direct model-level
        geopotential gradient: compute gradients on pressure surfaces first.
        """

        p_da = getattr(prepper, "p", None)
        z_da = getattr(prepper, "z", None)
        required_dims = {"time", "lev", "y", "x"}
        if not isinstance(p_da, xr.DataArray) or not isinstance(z_da, xr.DataArray):
            return None
        if not (required_dims <= set(p_da.dims) and required_dims <= set(z_da.dims)):
            return None
        if "x" not in p_da.coords or "y" not in p_da.coords:
            return None
        if p_da.sizes["x"] < 2 or p_da.sizes["y"] < 2:
            return None
        fc = self._coriolis_parameter()
        if fc is None:
            return None

        p_da = p_da.astype(float).compute()
        z_da = z_da.astype(float).compute()
        p_ref = p_da.mean(["time", "y", "x"]).dropna("lev")
        if p_ref.size < 3:
            return None

        # Column heights on common pressure surfaces.
        z_on_p = xr.apply_ufunc(
            _interp_column,
            p_da,
            z_da,
            input_core_dims=[["lev"], ["lev"]],
            output_core_dims=[["p_ref"]],
            kwargs={"target": p_ref.values, "outside": np.nan},
            vectorize=True,
        ).assign_coords(p_ref=p_ref.values)
        if not z_on_p.notnull().any():
            return None

        ug_p, vg_p = _geostrophic_wind(z_on_p, fc)
        z_p = z_on_p.mean(["y", "x"])

        def to_heights(values: xr.DataArray) -> xr.DataArray:
            return xr.apply_ufunc(
                _interp_column,
                z_p,
                values,
                input_core_dims=[["p_ref"], ["p_ref"]],
                output_core_dims=[["z"]],
                kwargs={"target": self.zt, "outside": None},
                vectorize=True,
            ).assign_coords(z=self.zt)

        ug_tz = to_heights(ug_p)
        vg_tz = to_heights(vg_p)

        # Fill occasional NaN time rows by the nearest valid profile.
        valid = (ug_tz.notnull().all("z") & vg_tz.notnull().all("z")).values
        if not valid.any():
            return None
        valid_idx = np.flatnonzero(valid)
        nearest = valid_idx[
            np.abs(valid_idx[:, None] - np.arange(valid.size)).argmin(axis=0)
        ]
        return tuple(
            da.isel(time=nearest).assign_coords(time=da["time"])
            for da in (ug_tz, vg_tz)
        )

    def _compute_geostrophic_profiles(
        self,
        data: xr.Dataset,
    ) -> tuple[xr.DataArray, xr.DataArray] | None:
        """Compute geostrophic wind from horizontal geopotential gradients.

        This mirrors the LS2D strategy: derive geostrophic wind from height
        gradients and Coriolis, then average over the target area. The result
        stays on the source ``z`` levels.
        """

        if "z3d" not in data:
            return None
        if "x" not in data.coords or "y" not in data.coords:
            return None
        if not {"time", "z", "y", "x"} <= set(data["z3d"].dims):
            return None
        if data.sizes["x"] < 2 or data.sizes["y"] < 2:
            return None
        fc = self._coriolis_parameter()
        if fc is None:
            return None
        return _geostrophic_wind(data["z3d"].astype(float), fc)


def _surface_fluxes(
    profile_ds: xr.Dataset,
    prepper: prep_harmonie.harmoniePrepper,
    times_sec: np.ndarray,
) -> dict[str, xr.DataArray]:
    """Kinematic ``wth``/``wq`` from Harmonie ``hfss``/``hfls`` (W m-2, upward) when loaded."""
    if prepper.ps is None or prepper.thls is None or prepper.exnrs is None:
        return {}
    exner = float(prepper.exnrs)
    rho = float(prepper.ps) / (prep_harmonie.Rd * float(prepper.thls) * exner)
    fluxes = {}
    if "hfss" in profile_ds:
        fluxes["wth"] = profile_ds["hfss"] / (rho * prep_harmonie.cp * exner)
    if "hfls" in profile_ds:
        fluxes["wq"] = profile_ds["hfls"] / (rho * prep_harmonie.Lv)
    if not fluxes:
        logger.info(
            "HarmonieAtmosphereModule: no surface fluxes in Harmonie data; wtsurf/wqsurf are not provided"
        )
    return {
        name: flux.reset_coords(drop=True).assign_coords(time=times_sec)
        for name, flux in fluxes.items()
    }


def _to_heights(da: xr.DataArray, z_dim: str, z_target: np.ndarray) -> xr.DataArray:
    """Interpolate along ``z_dim`` onto ``z_target``; values outside are held constant."""
    da = da.sortby(z_dim)
    z_src = da[z_dim].values
    return da.interp({z_dim: np.clip(z_target, z_src[0], z_src[-1])}).assign_coords(
        {z_dim: z_target}
    )


def _interp_column(x: np.ndarray, y: np.ndarray, target: np.ndarray, outside):
    """1D interpolation of ``y(x)`` onto ``target`` ignoring NaNs; ``outside=None`` holds end values."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_unique, first = np.unique(x[mask], return_index=True)
    if x_unique.size < 3:
        return np.full(np.shape(target), np.nan)
    return np.interp(target, x_unique, y[mask][first], left=outside, right=outside)


def _geostrophic_wind(
    height: xr.DataArray, fc: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Area-mean geostrophic wind from geopotential height on a horizontal ``(y, x)`` grid."""
    ug = -(9.81 / fc) * height.differentiate("y")
    vg = (9.81 / fc) * height.differentiate("x")
    return ug.mean(["y", "x"]), vg.mean(["y", "x"])
