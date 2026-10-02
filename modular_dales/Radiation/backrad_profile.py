"""Backrad profile helpers for DALES radiation schemes."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from modular_dales.IO_helpers.dales_external_data import resolve_rrtmg_data_paths
from modular_dales.IO_helpers.external_data_cache import cache_root


@dataclass
class BackradPressureProfile:
    """Pressure-coordinate background sounding for radiation input.

    Pressure is provided in Pascal and should be ordered from surface to top
    (high to low pressure).
    """

    pressure_pa: list[float]
    temperature_k: list[float]
    specific_humidity_kgkg: list[float]
    ozone_kgkg: list[float] | None = None
    liquid_water_kgkg: list[float] | None = None

    def normalized(self) -> BackradPressureProfile:
        p = np.asarray(self.pressure_pa, dtype=float)
        t = np.asarray(self.temperature_k, dtype=float)
        q = np.asarray(self.specific_humidity_kgkg, dtype=float)

        if not (p.size == t.size == q.size):
            raise ValueError(
                "pressure_pa, temperature_k, and specific_humidity_kgkg must have equal length"
            )
        if p.size < 2:
            raise ValueError("Backrad profile must contain at least 2 pressure levels")
        if np.any(p <= 0):
            raise ValueError("pressure_pa must be positive")

        o3 = None
        if self.ozone_kgkg is not None:
            o3 = np.asarray(self.ozone_kgkg, dtype=float)
            if o3.size != p.size:
                raise ValueError("ozone_kgkg length must match pressure_pa")
        lwc = None
        if self.liquid_water_kgkg is not None:
            lwc = np.asarray(self.liquid_water_kgkg, dtype=float)
            if lwc.size != p.size:
                raise ValueError("liquid_water_kgkg length must match pressure_pa")

        order = np.argsort(p)[::-1]
        p = p[order]
        t = t[order]
        q = q[order]
        if o3 is not None:
            o3 = o3[order]
        if lwc is not None:
            lwc = lwc[order]

        return BackradPressureProfile(
            pressure_pa=p.tolist(),
            temperature_k=t.tolist(),
            specific_humidity_kgkg=q.tolist(),
            ozone_kgkg=None if o3 is None else o3.tolist(),
            liquid_water_kgkg=None if lwc is None else lwc.tolist(),
        )

    def to_netcdf_dataset(self) -> xr.Dataset:
        profile = self.normalized()
        ds = xr.Dataset(
            data_vars={
                "T": ("lev", np.asarray(profile.temperature_k, dtype=float)),
                "q": ("lev", np.asarray(profile.specific_humidity_kgkg, dtype=float)),
            },
            coords={"lev": ("lev", np.asarray(profile.pressure_pa, dtype=float))},
        )
        if profile.ozone_kgkg is not None:
            ds["o3"] = ("lev", np.asarray(profile.ozone_kgkg, dtype=float))
        return ds

    def to_ascii(self) -> str:
        profile = self.normalized()
        pressure = np.asarray(profile.pressure_pa, dtype=float)
        temp = np.asarray(profile.temperature_k, dtype=float)
        humidity = np.asarray(profile.specific_humidity_kgkg, dtype=float)
        ozone = (
            np.asarray(profile.ozone_kgkg, dtype=float)
            if profile.ozone_kgkg is not None
            else np.zeros_like(pressure)
        )
        liquid = (
            np.asarray(profile.liquid_water_kgkg, dtype=float)
            if profile.liquid_water_kgkg is not None
            else np.zeros_like(pressure)
        )

        lines = [f"{temp[0]:.6f} {pressure.size}"]
        for p, t, q, o3, ql in zip(pressure, temp, humidity, ozone, liquid):
            lines.append(f"{p:.6f} {t:.6f} {q:.8e} {o3:.8e} {ql:.8e}")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_netcdf(cls, path: pathlib.Path) -> BackradPressureProfile:
        with xr.open_dataset(path) as ds:
            return cls.from_dataset(ds)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> BackradPressureProfile:
        """Profile from ``T``, ``q`` (and optional ``o3``) on pressure coordinate ``lev``."""
        profile = cls(
            pressure_pa=np.asarray(ds["lev"].values, dtype=float).tolist(),
            temperature_k=np.asarray(ds["T"].values, dtype=float).tolist(),
            specific_humidity_kgkg=np.asarray(ds["q"].values, dtype=float).tolist(),
            ozone_kgkg=(
                np.asarray(ds["o3"].values, dtype=float).tolist()
                if "o3" in ds
                else None
            ),
        )
        return profile.normalized()

    @classmethod
    def from_ascii(cls, path: pathlib.Path) -> BackradPressureProfile:
        with path.open("r", encoding="utf-8") as handle:
            rows = [line.strip() for line in handle.readlines() if line.strip()]

        if len(rows) < 2:
            raise ValueError(f"Invalid ASCII backrad file: {path}")

        first = rows[0].split()
        if len(first) < 2:
            raise ValueError(f"Invalid ASCII backrad header in {path}")
        nlev = int(float(first[1]))

        p: list[float] = []
        t: list[float] = []
        q: list[float] = []
        o3: list[float] = []
        ql: list[float] = []

        for row in rows[1 : 1 + nlev]:
            cols = row.split()
            if len(cols) < 3:
                raise ValueError(f"Invalid ASCII backrad row in {path}: {row}")
            p.append(float(cols[0]))
            t.append(float(cols[1]))
            q.append(float(cols[2]))
            o3.append(float(cols[3]) if len(cols) > 3 else 0.0)
            ql.append(float(cols[4]) if len(cols) > 4 else 0.0)

        return cls(
            pressure_pa=p,
            temperature_k=t,
            specific_humidity_kgkg=q,
            ozone_kgkg=o3,
            liquid_water_kgkg=ql,
        ).normalized()


@dataclass
class BackradInterpolatedProfile:
    """Interpolated-profile style builder for backrad pressure profiles.

    Provide anchor points on pressure levels and interpolate onto a target
    pressure grid. If ``target_pressure_pa`` is omitted, the anchor pressure
    levels are used as output levels.
    """

    pressure_pa: list[float]
    temperature_points: list[float]
    specific_humidity_points: list[float]
    ozone_points: list[float] | None = None
    liquid_water_points: list[float] | None = None
    target_pressure_pa: list[float] | None = None
    fill_value: float | str = "extrapolate"

    def _interpolate(
        self, x: np.ndarray, y: np.ndarray, x_target: np.ndarray
    ) -> np.ndarray:
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        return interp1d(
            x_sorted,
            y_sorted,
            fill_value=self.fill_value,
            bounds_error=False,
        )(x_target)

    def to_profile(
        self,
        template_profile: BackradPressureProfile | None = None,
    ) -> BackradPressureProfile:
        p = np.asarray(self.pressure_pa, dtype=float)
        t = np.asarray(self.temperature_points, dtype=float)
        q = np.asarray(self.specific_humidity_points, dtype=float)

        if not (p.size == t.size == q.size):
            raise ValueError(
                "pressure_pa, temperature_points, and specific_humidity_points must have equal length"
            )
        if p.size < 2:
            raise ValueError(
                "BackradInterpolatedProfile requires at least two anchor points"
            )

        if self.target_pressure_pa is not None:
            target_p = np.asarray(self.target_pressure_pa, dtype=float)
        elif template_profile is not None:
            target_p = np.asarray(template_profile.pressure_pa, dtype=float)
        else:
            target_p = p.copy()

        t_out = self._interpolate(p, t, target_p)
        q_out = self._interpolate(p, q, target_p)

        o3_out = None
        if self.ozone_points is not None:
            o3 = np.asarray(self.ozone_points, dtype=float)
            if o3.size != p.size:
                raise ValueError("ozone_points length must match pressure_pa")
            o3_out = self._interpolate(p, o3, target_p)

        ql_out = None
        if self.liquid_water_points is not None:
            ql = np.asarray(self.liquid_water_points, dtype=float)
            if ql.size != p.size:
                raise ValueError("liquid_water_points length must match pressure_pa")
            ql_out = self._interpolate(p, ql, target_p)

        return BackradPressureProfile(
            pressure_pa=target_p.tolist(),
            temperature_k=t_out.tolist(),
            specific_humidity_kgkg=q_out.tolist(),
            ozone_kgkg=None if o3_out is None else o3_out.tolist(),
            liquid_water_kgkg=None if ql_out is None else ql_out.tolist(),
        ).normalized()


def profile_from_path(path: pathlib.Path) -> BackradPressureProfile:
    if path.suffix == ".nc":
        return BackradPressureProfile.from_netcdf(path)
    return BackradPressureProfile.from_ascii(path)


def default_profile() -> BackradPressureProfile:
    """Return a default profile, preferring pre-existing repository files."""
    candidates = (
        pathlib.Path.cwd() / "extra_data" / "backrad.inp.001.nc",
        pathlib.Path.cwd() / "extra_data" / "backrad.inp.001",
    )
    for candidate in candidates:
        if candidate.exists():
            return profile_from_path(candidate)

    pressure = [100000.0, 92500.0, 85000.0, 70000.0, 50000.0, 30000.0, 10000.0]
    temperature = [290.0, 285.0, 279.0, 265.0, 250.0, 230.0, 210.0]
    humidity = [1.2e-2, 8.0e-3, 4.0e-3, 1.2e-3, 3.0e-4, 8.0e-5, 1.0e-5]
    ozone = [3.0e-8, 3.0e-8, 4.0e-8, 1.2e-7, 3.5e-7, 7.0e-7, 1.0e-6]
    return BackradPressureProfile(
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kgkg=humidity,
        ozone_kgkg=ozone,
    )


def write_profile(profile: BackradPressureProfile, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".nc":
        profile.to_netcdf_dataset().to_netcdf(path)
    else:
        path.write_text(profile.to_ascii(), encoding="utf-8")
    return path


def profile_from_forcings(
    initial: Mapping[str, xr.DataArray],
) -> BackradPressureProfile | None:
    """Sounding provided by other modules as ``backrad_T``/``backrad_q``/``backrad_o3`` on ``lev``."""
    if "backrad_T" not in initial or "backrad_q" not in initial:
        return None
    sounding = {"T": initial["backrad_T"], "q": initial["backrad_q"]}
    if "backrad_o3" in initial:
        sounding["o3"] = initial["backrad_o3"]
    return BackradPressureProfile.from_dataset(xr.Dataset(sounding))


def select_backrad_profile(module) -> BackradPressureProfile:
    """Backrad sounding for a radiation module.

    Precedence: the module's own ``backrad_*`` setting, then a sounding
    provided by another module (e.g. LS2D or Harmonie), then the default.
    """
    user_sources = [
        value
        for value in (
            module.backrad_profile,
            module.backrad_source_file,
            module.backrad_interpolated_profile,
        )
        if value is not None
    ]
    if len(user_sources) > 1:
        raise ValueError(
            f"{module.module_name}: provide at most one of backrad_profile, backrad_source_file, or backrad_interpolated_profile."
        )
    if module.backrad_profile is not None:
        return module.backrad_profile
    if module.backrad_source_file is not None:
        return profile_from_path(pathlib.Path(module.backrad_source_file))
    if module.backrad_interpolated_profile is not None:
        return module.backrad_interpolated_profile.to_profile(
            template_profile=default_profile()
        )
    provided = profile_from_forcings(module.sim.forcings().initial)
    return provided if provided is not None else default_profile()


def register_radiation_files(module, iradiation: int) -> None:
    """Write the backrad sounding and register the RRTMG(P) input files for ``iradiation`` 4/5."""
    if iradiation not in (4, 5):
        return
    sim = module.sim
    backrad_nc = write_profile(
        select_backrad_profile(module),
        cache_root("backrad", sim) / f"backrad.inp.{sim.exp_id:03d}.nc",
    )
    sim.required_files[backrad_nc.name] = backrad_nc.as_posix()
    external = resolve_rrtmg_data_paths(sim)
    sim.required_files["rrtmg_lw.nc"] = external.rrtmg_lw.as_posix()
    sim.required_files["rrtmg_sw.nc"] = external.rrtmg_sw.as_posix()
    if iradiation == 5:
        for file in external.rrtmgp_data_dir.glob("*.nc"):
            sim.required_files[file.name] = file.as_posix()
