from __future__ import annotations

from MoDa import dales_simulation
from MoDa.Atmosphere import AtmosphereModule, lin
from MoDa.Atmosphere.Profile import AtmosphericProfile
from MoDa.Configuration.defaultnamelist import DefaultNamelistModule
from MoDa.Configuration.output_modules import EasyOutputModule
from MoDa.Configuration.run_and_time import TimeModule
from MoDa.Emission.emission import (
    EmissionModule,
    EmissionPointSource,
    EmissionTracer,
)
from MoDa.Geometry.grids import GridDales
from MoDa.Surface.surface import ConstantSurfaceTemperatureModule
from MoDa.vars import *


def emission_case(machine_conf: dict) -> dales_simulation:
    """Simple simulation including the EmissionModule."""

    sim = dales_simulation("emission_case", machine_conf)

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

    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=3600)
    sim.add(time_module)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=0.0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0.0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=0)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.0, ddz=0)))
    sim.add(atmo)

    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15, z0mav=0.0001, z0hav=0.0001, ps=100000, albedoav=0.22
        )
    )

    emis = EmissionModule()

    emis.add(
        EmissionTracer(
            name="co2",
            long_name="Carbon Dioxide (CO2)",
            unit="ppm",
            molar_mass=44.009,
            lemis=True,
        )
    )
    emis.add(
        EmissionTracer(
            name="so2",
            long_name="Sulfur Dioxide (SO2)",
            unit="ppb",
            molar_mass=64.066,
            lemis=True,
        )
    )

    emis.add(
        EmissionPointSource(
            tracer_name="co2",
            x_idx=4,
            y_idx=4,
            height=10.0,
            temperature=293.0,
            volume=1.0,
            emission=10.0,
            stack_exit_area=1.0,
        )
    )
    emis.add(
        EmissionPointSource(
            tracer_name="co2",
            x_idx=8,
            y_idx=8,
            height=20.0,
            temperature=293.0,
            volume=1.0,
            emission=5.0,
            stack_exit_area=1.0,
        )
    )
    emis.add(
        EmissionPointSource(
            tracer_name="so2",
            x_idx=2,
            y_idx=2,
            height=15.0,
            temperature=293.0,
            volume=1.0,
            emission=3.0,
            stack_exit_area=1.0,
        )
    )
    emis.add(
        EmissionPointSource(
            tracer_name="so2",
            x_idx=10,
            y_idx=10,
            height=25.0,
            temperature=293.0,
            volume=1.0,
            emission=7.0,
            stack_exit_area=1.0,
        )
    )

    sim.add(emis)
    sim.add(EasyOutputModule())
    return sim
