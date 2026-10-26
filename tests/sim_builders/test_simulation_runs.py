import subprocess

import pytest

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    ConstantSurfaceTemperatureModule,
    DefaultNamelistModule,
    GridDales,
    Nest_in_Dales,
    NestingTopology,
    TimeModule,
    dales_simulation,
)
from modular_dales.Atmosphere import InterpolatedProfile, lin
from modular_dales.Configuration.output_modules import EasyOutputModule
from modular_dales.vars import *


@pytest.mark.slow
def test_triple_nested_simulation_setup(machine_conf):
    """Build a triple-nested simulation (coarse -> mid -> fine) and run preprocessing.

    This test verifies the nested topology can be constructed and the
    preprocessing pipeline runs without invoking the external DALES binary.
    """

    conf = machine_conf("triple_nested_sim")
    conf["job_conf"]["numcores"] = 8

    sims = []

    # Create three separate simulations: coarse, mid, fine
    grids = [
        GridDales(
            itot=64,
            jtot=64,
            kmax=70,
            xsize=640.0,
            ysize=640.0,
            dz0=20.0,
            alpha=1.0,
            x0=0,
            y0=0,
            kmax_soil=4,
            xlat=52.0,  # Example latitude value
            xlon=5.0,  # Example longitude value
        ),
        GridDales(
            itot=32,
            jtot=32,
            kmax=60,
            xsize=320.0,
            ysize=320.0,
            x0=160.0,
            y0=160.0,
            dz0=20.0,
            alpha=1.0,
            kmax_soil=4,
            xlat=52.0,  # Example latitude value
            xlon=5.0,  # Example longitude value
        ),
        GridDales(
            itot=16,
            jtot=16,
            kmax=50,
            xsize=160.0,
            ysize=160.0,
            x0=240.0,
            y0=240.0,
            dz0=20.0,
            alpha=1.0,
            kmax_soil=4,
            xlat=52.0,  # Example latitude value
            xlon=5.0,  # Example longitude value
        ),
    ]

    for idx, grid in enumerate(grids):
        simname = f"triple_{idx}"
        sim = dales_simulation(simname, conf)
        sim += DefaultNamelistModule()

        atmo = AtmosphereModule()
        atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=0)))
        atmo.add(AtmosphericProfile(variable=ug, shape=lin(surf_val=3, ddz=0)))
        atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=3, ddz=0)))
        atmo.add(AtmosphericProfile(variable=vg, shape=lin(surf_val=3, ddz=0)))
        atmo.add(AtmosphericProfile(variable=ua_nudge, shape=lin(surf_val=3, ddz=0)))
        atmo.add(AtmosphericProfile(variable=va_nudge, shape=lin(surf_val=3, ddz=0)))
        # atmo += AtmosphericProfile(
        #     variable=thetal, shape=lin(surf_val=293.15,ddz=1e-2)
        # )
        atmo.add(
            InterpolatedProfile(
                variable=thetal,
                z=[0, 400, 410, 1600],
                points=[293.15, 293.15, 298.15, 301.15],
            )
        )
        atmo.add(
            InterpolatedProfile(
                variable=thl_nudge,
                z=[0, 400, 410, 1600],
                points=[293.15, 293.15, 298.15, 301.15],
            )
        )
        atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.0018, ddz=0)))
        atmo.add(
            AtmosphericProfile(variable=qt_nudge, shape=lin(surf_val=0.0018, ddz=0))
        )
        atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0, ddz=0)))
        atmo.add(AtmosphericProfile(variable=wa_nudge, shape=lin(surf_val=0, ddz=0)))
        atmo.add(
            InterpolatedProfile(
                variable=tke,
                z=[0, 4000, 5000],
                points=[1, 1e-8, 1e-8],
            )
        )
        atmo.add(
            AtmosphericProfile(
                variable=w,
                shape=lin(surf_val=0.0, ddz=0.0),
            )
        )
        sim += atmo

        sim += TimeModule(
            xday=1,
            xtime=12.0,
            xyear=2023,
            runtime=60,
            startyear=2023,
            startmonth=1,
            startday=1,
        )
        sim += ConstantSurfaceTemperatureModule(
            thls=293.15, z0mav=1e-4, z0hav=1e-4, ps=100000
        )

        sim += grid

        # Build simple nesting topology containing all grids up to current
        nesting = NestingTopology()
        for g in grids:
            nesting += g
        nesting.my_idx = nesting.nestings.index(grid)
        sim += nesting

        sim += EasyOutputModule(output_interval=5)

        if idx != 0:
            openbc = Nest_in_Dales(
                time0="2023-01-01T12:00:00",
                start="2023-01-01T12:00:00",
                end="2023-01-02T12:00:00",
                e12=0.01,
                dxint=grid.xsize / grid.itot,
                dyint=grid.ysize / grid.jtot,
                tauh=20,
                taum=0,
                inpath_coarse=sims[-1].output_path / "input/",
                outpath_coarse=sims[-1].output_path / "run_001/",
                outpath_coarse_old=sims[-1].output_path / "run_001/",
            )
            sim += openbc

        # Run preprocessing to prepare outputs
        sim.sim_preprocessing_pipeline()

        outdir = sim.output_path
        subprocess.run(["./job.001"], check=True, cwd=outdir.as_posix())

        sims.append(sim)
