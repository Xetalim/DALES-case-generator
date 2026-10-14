import logging
from dataclasses import dataclass, field

import dask
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

        ug_mean, vg_mean = dask.compute(
            prepper.ug.mean(["x", "y"]), prepper.vg.mean(["x", "y"])
        )
        ug, vg = (
            _to_heights(g, "z", self.zt)
            .assign_coords(time=times_sec)
            .transpose(*u.dims)
            for g in (ug_mean, vg_mean)
        )

        fields = {"u": u, "v": v, "thl": thl, "qt": qt, "wls": wls, "ug": ug, "vg": vg}
        fields.update(
            _advective_tendencies_from_harmonie(
                data=data,
                z_target=self.zt,
                times_sec=times_sec,
                infer_adv_tendencies=self.infer_adv_tendencies,
            )
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


def _advective_tendencies_from_harmonie(
    data: xr.Dataset,
    z_target: np.ndarray,
    times_sec: np.ndarray,
    infer_adv_tendencies: bool,
) -> dict[str, xr.DataArray]:
    """Compute domain-mean advective tendencies from horizontal gradients."""
    names = ("u", "v", "thl", "qt")
    if not infer_adv_tendencies:
        profile_ds = data.mean([d for d in ("x", "y") if d in data.dims], skipna=True)
        out = {}
        for name in names:
            out[f"dt{name}_advec"] = xr.zeros_like(
                _to_heights(profile_ds[name], "z", z_target).assign_coords(
                    time=times_sec
                )
            )
        return out

    required = {"time", "z", "x", "y", *names}
    missing = [name for name in required if name not in data and name not in data.dims]
    if missing:
        raise ValueError(
            "Cannot compute advective tendencies from Harmonie fields; missing: "
            + ", ".join(sorted(missing))
        )
    if data.sizes["x"] < 2 or data.sizes["y"] < 2:
        raise ValueError(
            "Cannot compute advective tendencies: need at least 2 points in both x and y"
        )

    wind_u = data["u"]
    wind_v = data["v"]
    out = {}
    for name in names:
        var = data[name]
        tendency = -(wind_u * var.differentiate("x") + wind_v * var.differentiate("y"))
        profile = tendency.mean(["x", "y"], skipna=True)
        out[f"dt{name}_advec"] = _to_heights(profile, "z", z_target).assign_coords(
            time=times_sec
        )
    return out


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
