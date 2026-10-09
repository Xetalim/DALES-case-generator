from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    BackradInterpolatedProfile,
    BackradPressureProfile,
    ConstantSurfaceTemperatureModule,
    DefaultNamelistModule,
    EasyOutputModule,
    GridDales,
    LSMHomogeneousModule,
    RadiationModule,
    RRTMGRadiationModule,
    SprayingModule,
    TimeModule,
    dales_simulation,
)
from modular_dales.Atmosphere import InterpolatedProfile
from modular_dales.vars import *  # noqa: F401,F403


def _base_runtime_case(machine_conf: dict, case_name: str) -> dales_simulation:
    sim = dales_simulation(case_name, machine_conf)
    sim += DefaultNamelistModule()
    sim += GridDales(
        itot=8,
        jtot=8,
        kmax=64,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0.0,
        y0=0.0,
        alpha=1.0,
        dz0=10.0,
    )

    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(
        variable=ua, shape="lin", params=dict(surf_val=2.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=va, shape="lin", params=dict(surf_val=0.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=thetal,
        shape="lin",
        params=dict(surf_val=293.15, ddz=5e-3),
    )
    atmo += AtmosphericProfile(
        variable=qt, shape="lin", params=dict(surf_val=0.008, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=wa, shape="lin", params=dict(surf_val=0.0, ddz=0.0)
    )
    atmo += InterpolatedProfile(
        variable=tke, z=[0, 1000, 3000], points=[0.2, 0.02, 0.01]
    )
    sim += atmo

    sim += TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=120)

    if sim.nml.get("RUN") is None:
        sim.nml["RUN"] = {}
    sim.nml["RUN"]["nprocx"] = 1
    sim.nml["RUN"]["nprocy"] = 1

    return sim


def radiation_type_4_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "radiation_type_4_case")
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        ps=100000.0,
        albedoav=0.22,
    )
    sim += RadiationModule(iradiation=4)
    sim += EasyOutputModule(output_interval=30)
    return sim


def radiation_type_5_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "radiation_type_5_case")
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        ps=100000.0,
        albedoav=0.22,
    )
    sim += RadiationModule(iradiation=5)
    sim += EasyOutputModule(output_interval=30)
    return sim


def radiation_backrad_profile_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "radiation_backrad_profile_case")
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        ps=100000.0,
        albedoav=0.22,
    )
    sim += RadiationModule(
        iradiation=4,
        backrad_profile=BackradPressureProfile(
            pressure_pa=[100000.0, 85000.0, 70000.0, 50000.0, 30000.0, 10000.0],
            temperature_k=[290.0, 281.0, 272.0, 257.0, 235.0, 212.0],
            specific_humidity_kgkg=[1.1e-2, 6.0e-3, 3.0e-3, 9.0e-4, 1.5e-4, 2.0e-5],
        ),
    )
    sim += EasyOutputModule(output_interval=30)
    return sim


def radiation_backrad_source_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "radiation_backrad_source_case")
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        ps=100000.0,
        albedoav=0.22,
    )
    sim += RadiationModule(
        iradiation=4,
        backrad_source_file=_reference_backrad_path(),
    )
    sim += EasyOutputModule(output_interval=30)
    return sim


def radiation_backrad_interpolated_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "radiation_backrad_interpolated_case")
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        ps=100000.0,
        albedoav=0.22,
    )
    sim += RadiationModule(
        iradiation=4,
        backrad_interpolated_profile=BackradInterpolatedProfile(
            pressure_pa=[100000.0, 70000.0, 40000.0, 10000.0],
            temperature_points=[289.0, 268.0, 246.0, 214.0],
            specific_humidity_points=[1.0e-2, 2.8e-3, 4.5e-4, 2.0e-5],
            target_pressure_pa=[
                100000.0,
                90000.0,
                80000.0,
                70000.0,
                60000.0,
                40000.0,
                20000.0,
                10000.0,
            ],
        ),
    )
    sim += EasyOutputModule(output_interval=30)
    return sim


def lsm_homogeneous_runtime_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "lsm_homogeneous_runtime_case")
    sim += LSMHomogeneousModule(
        ps=100000.0,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        albedoav=0.22,
        iinterp_t=1,
        iinterp_theta=1,
        kmax_soil=4,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        c_low=0.35,
        c_high=0.25,
        c_bare=0.30,
        c_water=0.05,
        c_asph=0.05,
        z0m_low=0.05,
        z0m_high=0.5,
        z0m_bare=0.01,
        z0m_water=2.0e-4,
        z0m_asph=0.02,
        z0h_low=0.005,
        z0h_high=0.05,
        z0h_bare=0.001,
        z0h_water=2.0e-5,
        z0h_asph=0.002,
        lambda_s_low=0.5,
        lambda_s_high=0.6,
        lambda_s_bare=0.4,
        lambda_s_asph=0.7,
        lambda_us_low=0.2,
        lambda_us_high=0.25,
        lambda_us_bare=0.15,
        lambda_us_asph=0.3,
        lai_low=2.0,
        lai_high=4.0,
        rs_min_low=100.0,
        rs_min_high=80.0,
        rs_min_bare=200.0,
        rs_min_asph=250.0,
        t_soil_p=[290.0, 289.0, 288.0, 287.0],
        theta_soil_p=[0.30, 0.29, 0.28, 0.27],
        soil_index_p=[5, 5, 5, 5],
        ar_low=3.0,
        br_low=2.0,
        ar_high=4.0,
        br_high=2.5,
        gD_high=0.05,
        tskin_water=289.0,
        thls=293.15,
        lheterogeneous=False,
    )
    sim += RRTMGRadiationModule()
    sim += EasyOutputModule(output_interval=30)
    return sim


def spraying_runtime_case(machine_conf: dict) -> dales_simulation:
    sim = _base_runtime_case(machine_conf, "spraying_runtime_case")
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=1.0e-4,
        z0hav=1.0e-4,
        ps=100000.0,
        albedoav=0.22,
    )
    sim += RadiationModule(iradiation=0)
    sim += SprayingModule(
        lsalt_spraying=True,
        lwater_spraying=False,
        x_idx=4,
        y_idx=4,
        z_idx=2,
        salt_spray_rate=2.0e-2,
        salinity=1.0,
        tracer="salt",
        lsalt_sponge=False,
    )
    sim += EasyOutputModule(output_interval=10, enable_output=True)
    return sim


def assert_sprayed_salt_in_fielddump(output_path: Path) -> None:
    run_dir = output_path / "run_001"
    candidates = [
        run_dir / "fielddump.001.nc",
        output_path / "fielddump.nc",
    ]
    fielddump_path = next((path for path in candidates if path.is_file()), None)
    if fielddump_path is None:
        raise AssertionError(
            f"No fielddump output found in expected locations: {candidates}"
        )

    with xr.open_dataset(fielddump_path) as ds:
        if "salt" not in ds.variables:
            raise AssertionError(
                f"Expected sprayed tracer variable 'salt' in {fielddump_path.name}, got {list(ds.variables)}"
            )

        salt = np.asarray(ds["salt"].values, dtype=float)
        if salt.ndim < 2:
            raise AssertionError(
                f"Expected time-dependent salt field in {fielddump_path.name}, got shape {salt.shape}"
            )

        time_axis = 0
        salt_means = np.nanmean(salt.reshape(salt.shape[time_axis], -1), axis=1)
        if salt_means.size < 2:
            raise AssertionError(
                f"Need at least two fielddump times to verify spraying trend, got {salt_means.size}"
            )

        if not (salt_means[-1] > salt_means[0]):
            raise AssertionError(
                "Expected mean salt concentration to increase over time due to spraying, "
                f"but got start={salt_means[0]:.6e}, end={salt_means[-1]:.6e}."
            )
