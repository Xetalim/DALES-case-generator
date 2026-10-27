from __future__ import annotations

from modular_dales import dales_simulation
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    lin,
)
from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.Configuration.output_modules import EasyOutputModule
from modular_dales.Configuration.run_and_time import TimeModule
from modular_dales.Geometry.grids import GridDales
from modular_dales.modular.time_dependent import (
    TimedependentModule,
)
from modular_dales.modular.time_dependent_scalars import TimeDependentScalar
from modular_dales.Surface.surface import ConstantSurfaceTemperatureModule
from modular_dales.vars import *


def timedep_atmosphere_with_params_case_long(machine_conf: dict) -> dales_simulation:
    """Construct a simulation with time-dependent scalar params inside AtmosphericProfile."""
    sim = dales_simulation("timedep_atmosphere_params", machine_conf)

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
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )
    atmo.add(
        AtmosphericProfile(
            variable=wa,
            shape=lin(
                surf_val=TimeDependentScalar(times=[0.0, 60.0], values=[0.0, 0.02]),
                ddz=0.0,
            ),
        )
    )
    atmo.add(
        AtmosphericProfile(
            variable=ug,
            shape=lin(
                surf_val=TimeDependentScalar(times=[0.0, 60.0], values=[5.0, 8.0]),
                ddz=TimeDependentScalar(times=[0.0, 60.0], values=[1e-3, 2e-3]),
            ),
        )
    )

    sim.add(TimedependentModule(timesteps=[0.0, 60.0]))
    sim.add(ConstantSurfaceTemperatureModule(
        thls=TimeDependentScalar(times=[0.0, 60.0], values=[293.15, 294.15]),
        z0mav=0.0001,
        z0hav=0.0001,
        ps=100000,
        albedoav=0.22,
    ))
    sim.add(atmo)
    sim.add(TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60))

    sim.add(EasyOutputModule(output_interval=10))
    return sim
