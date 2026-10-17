from __future__ import annotations

from modular_dales import dales_simulation
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    exp,
    expsinw,
    lin,
    linmlsurf,
)
from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.Configuration.output_modules import EasyOutputModule
from modular_dales.Configuration.run_and_time import TimeModule
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.LSM import FromLCZ, LandUseModification, LSMModule
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.Surface.LSM.SLuRB.slurb import (
    SLURBModification,
    SLURBModule,
    SLURBVariableModification,
)
from modular_dales.vars import *


def SLURB_LCZ_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation combining SLURB and LCZ-based LSM configuration."""
    sim = dales_simulation("SLURB_case", machine_conf)

    sim += DefaultNamelistModule()

    domain_info = GridDales(
        itot=32,
        jtot=32,
        kmax=40,
        xsize=1600.0,
        ysize=1600.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=85325.531122,
        y0=446137.619111,
        alpha=1.0,
        dz0=10,
        proj4="+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.4171,50.3319,465.5524,1.9342,-1.6677,9.1019,4.0725 +units=m +no_defs +type=crs",
    )
    sim += domain_info

    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3))
    atmo += AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(
        variable=thetal,
        shape=linmlsurf(surf_val=293, offset_val=1.25, lapse_rate=1e-3, z_ml=0),
    )
    atmo += AtmosphericProfile(
        variable=qt, shape=exp(surf_val=0.016, lambda_val=1500, z_ml=0)
    )
    atmo += AtmosphericProfile(
        variable=wa,
        shape=expsinw(surf_val=4.0e-3, amp=0, H=2500, Hp=5300),
    )
    atmo += InterpolatedProfile(
        variable=tke,
        z=[0, 4000, 5000],
        points=[1, 1e-8, 1e-8],
    )

    sim += atmo
    lsm = LSMModule(
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
    )
    lsm += UniformSkinTemperature(291)
    lsm += UniformSoilTemperature([288, 288, 288, 288])
    lsm += UniformSoilMoisture(
        [
            0.36867549,
            0.25300502,
            0.14997292,
            0.16459982,
        ]
    )
    lsm += LandUseModification(geometry=AllGeometry(), type="grs")
    lsm += FromLCZ()
    sim += lsm

    slurb = SLURBModule(deep_soil_temperature=293)
    slurb += SLURBModification(
        geometry=AllGeometry(),
        vars=[SLURBVariableModification(varname="albedo_av", value=10)],
    )
    sim += slurb

    sim += RadiationModule(
        iradiation=4,
    )

    sim += EasyOutputModule(output_interval=10)

    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)
    sim += time_module
    return sim


def assert_slurb_lcz_files_written(machine_conf) -> None:
    """Test that SLURB_LCZ files are written correctly for SLURB_LCZ_case."""

    sim = SLURB_LCZ_case(machine_conf("slurb_LCZ"))
    sim.sim_preprocessing_pipeline()

    slurb_files = list(sim.output_path.glob("input/inslurb.*.nc"))
    assert len(slurb_files) == 1, f"Expected 1 SLURB file, found {len(slurb_files)}"
    for slurb_file in slurb_files:
        assert slurb_file.is_file(), f"Expected SLURB file {slurb_file} to exist"
        assert slurb_file.stat().st_size > 0, (
            f"Expected SLURB file {slurb_file} to be non-empty"
        )
    lsm_files = list(sim.output_path.glob("input/lsm.inp_*.nc"))
    assert len(lsm_files) == 1, f"Expected 1 LSM file, found {len(lsm_files)}"
    for lsm_file in lsm_files:
        assert lsm_file.is_file(), f"Expected LSM file {lsm_file} to exist"
        assert lsm_file.stat().st_size > 0, (
            f"Expected LSM file {lsm_file} to be non-empty"
        )
