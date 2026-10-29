from __future__ import annotations

from typing import Any

from moda import DefaultNamelistModule, TimeModule
from moda.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    lin,
)
from moda.Geometry.grids import GridDales
from moda.modular import dales_simulation
from moda.Surface.surfacemodules import ConstantSurfaceTemperatureModule
from moda.vars import qt, thetal, tke, ua, va, wa


def build_warmstart_base_sim(
    machine_conf: dict[str, Any],
    case_name: str,
    domain: dict[str, Any],
    tfinal: int,
    trestart: int,
    lwarmstart: bool,
    extra_modules: list[Any] | None = None,
) -> dales_simulation:
    """Construct a tiny DALES case suitable for warmstart continuity tests."""
    sim = dales_simulation(case_name, machine_conf)
    sim.add(DefaultNamelistModule(lwarmstart=bool(lwarmstart), trestart=int(trestart)))

    sim.add(
        GridDales(
            itot=int(domain["itot"]),
            jtot=int(domain["jtot"]),
            kmax=int(domain["kmax"]),
            xsize=float(domain["xsize"]),
            ysize=float(domain["ysize"]),
            kmax_soil=4,
            xlat=52.25,
            xlon=5.45,
            x0=0,
            y0=0,
            alpha=1.0,
            dz0=5.0,
        )
    )

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3.0, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(
        InterpolatedProfile(variable=tke, z=[0, 4000, 5000], points=[1.0, 1e-8, 1e-8])
    )
    sim.add(atmo)

    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15,
            z0mav=1e-4,
            z0hav=1e-4,
            ps=100000.0,
            albedoav=0.22,
        )
    )
    sim.add(
        TimeModule(
            xtime=0.0,
            xday=1,
            xyear=2025,
            runtime=float(tfinal),
            trestart=float(trestart),
        )
    )
    if extra_modules:
        for module in extra_modules:
            sim.add(module)

    return sim
