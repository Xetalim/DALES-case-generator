from __future__ import annotations

from moda import DefaultNamelistModule, TimeModule, dales_simulation
from moda.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    exp,
    expsinw,
    lin,
    linmlsurf,
)
from moda.Configuration.output_modules import EasyOutputModule
from moda.Geometry.grids import GridDales
from moda.Geometry.masks import AllMask
from moda.Radiation.radiation import RadiationModule
from moda.Surface.LSM import LandUseSet
from moda.Surface.LSM.lsm_module import LSMModule
from moda.Surface.LSM.lsm_sources.lsm_sources import (
    FromLCZ,
    SlurbModification,
)
from moda.Surface.LSM.lsm_sources.lu_modifications import LandUseModification
from moda.Surface.LSM.lsm_sources.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from moda.vars import *


def SLURB_LCZ_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation combining SLURB and LCZ-based LSM configuration."""
    sim = dales_simulation("SLURB_case", machine_conf)

    sim.add(DefaultNamelistModule())

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
    sim.add(domain_info)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(
        AtmosphericProfile(
            variable=thetal,
            shape=linmlsurf(surf_val=293, offset_val=1.25, lapse_rate=1e-3, z_ml=0),
        )
    )
    atmo.add(
        AtmosphericProfile(
            variable=qt, shape=exp(surf_val=0.016, lambda_val=1500, z_ml=0)
        )
    )
    atmo.add(
        AtmosphericProfile(
            variable=wa,
            shape=expsinw(surf_val=4.0e-3, amp=0, H=2500, Hp=5300),
        )
    )
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )

    sim.add(atmo)
    lsm = LSMModule(
        include_slurb=True,
        deep_soil_temperature=293,
        building_indoor_temperature=293.15,
        land_use_set=LandUseSet.DEPAC,
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
    )
    lsm.add(UniformSkinTemperature(291), priority=600)
    lsm.add(UniformSoilTemperature([288, 288, 288, 288]), priority=600)
    lsm.add(
        UniformSoilMoisture(
            [
                0.36867549,
                0.25300502,
                0.14997292,
                0.16459982,
            ]
        ),
        priority=600,
    )
    lsm.add(LandUseModification(mask=AllMask(), type="grs"), priority=0)
    lsm.add(FromLCZ(), priority=100)
    sim.add(lsm)

    lsm.add(
        SlurbModification(
            mask=AllMask(),
            varname="albedo_av",
            value=10,
        ),
        priority=200,
    )

    sim.add(
        RadiationModule(
            iradiation=4,
        )
    )

    sim.add(EasyOutputModule(output_interval=10))

    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)
    sim.add(time_module)
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
