from __future__ import annotations

from moda import DefaultNamelistModule, TimeModule, dales_simulation
from moda.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    lin,
)
from moda.Geometry.grids import GridDales
from moda.Geometry.masks import AllMask
from moda.Radiation.radiation import RadiationModule
from moda.Surface.LSM import LandUseSet
from moda.Surface.LSM.lsm_module import LSMModule
from moda.Surface.LSM.lsm_sources.lu_modifications import LandUseModification
from moda.Surface.LSM.lsm_sources.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from moda.vars import *


def sim_2_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation from a YAML string, for testing deserialization."""
    sim = dales_simulation("complicated", machine_conf)

    sim.add(DefaultNamelistModule())

    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=20,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=5.0,
    )
    sim.add(domain_info)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0, ddz=0)))
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )

    sim.add(atmo)
    sim.add(TimeModule(xday=0, xtime=12.0, xyear=2023, runtime=60))

    sim.add(
        RadiationModule(
            iradiation=4,
        )
    )
    lsm = LSMModule(
        land_use_set=LandUseSet.DEPAC,
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
    )
    lsm.add(UniformSkinTemperature(293), priority=600)
    lsm.add(UniformSoilTemperature([293, 293, 293, 293]), priority=600)
    lsm.add(UniformSoilMoisture([0.3, 0.3, 0.3, 0.3]), priority=600)
    lsm.add(LandUseModification(mask=AllMask(), type="grs"), priority=0)
    sim.add(lsm)
    return sim
