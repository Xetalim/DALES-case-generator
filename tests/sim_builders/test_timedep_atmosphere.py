from __future__ import annotations

import netCDF4

from MoDa import dales_simulation
from MoDa.Atmosphere import (
    AtmosphereModule,
    lin,
)
from MoDa.Atmosphere.Profile import (
    AtmosphericProfile,
    InterpolatedProfile,
    TimedAtmosphereProfile,
)
from MoDa.Configuration.defaultnamelist import DefaultNamelistModule
from MoDa.Configuration.output_modules import EasyOutputModule
from MoDa.Configuration.run_and_time import TimeModule
from MoDa.Geometry.grids import GridDales
from MoDa.modular.time_dependent import (
    TimedependentModule,
)
from MoDa.modular.time_dependent_scalars import TimeDependentScalar
from MoDa.Surface.surface import ConstantSurfaceTemperatureModule
from MoDa.vars import *


def timedep_atmosphere_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation with time-dependent atmosphere forcing entries."""
    sim = dales_simulation("timedep_atmosphere", machine_conf)

    sim.add(DefaultNamelistModule())
    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=60,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=8.0,
    )
    sim.add(domain_info)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=1.0, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.01, ddz=0)))
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )
    atmo.add(
        [
            TimedAtmosphereProfile(
                time=0.0,
                profile=AtmosphericProfile(
                    variable=dpdx, shape=lin(surf_val=1.0, ddz=1e-3)
                ),
            ),
            TimedAtmosphereProfile(
                time=30.0,
                profile=AtmosphericProfile(
                    variable=dpdx, shape=lin(surf_val=2.0, ddz=1e-3)
                ),
            ),
            TimedAtmosphereProfile(
                time=60.0,
                profile=AtmosphericProfile(
                    variable=dpdx, shape=lin(surf_val=3, ddz=1e-3)
                ),
            ),
        ]
    )

    sim.add(TimedependentModule(timesteps=[0.0, 30.0, 60.0]))
    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=TimeDependentScalar(
                times=[0.0, 30.0, 60.0], values=[293.15, 294.15, 295.15]
            ),
            z0mav=0.0001,
            z0hav=0.0001,
            ps=100000,
            albedoav=0.22,
        )
    )
    sim.add(atmo)
    sim.add(TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60))

    sim.add(EasyOutputModule(output_interval=10))
    return sim


def assert_timedep_atmosphere_files_written(machine_conf) -> None:
    """Test that atmosphere time-dependent file is written correctly."""

    sim = timedep_atmosphere_case(machine_conf("timedep_atmosphere_files"))
    sim.sim_preprocessing_pipeline()

    timedep_files = list(sim.output_path.glob("input/forcings.*.nc"))
    assert len(timedep_files) == 1, (
        f"Expected 1 timedep atmosphere file, found {len(timedep_files)}"
    )

    with netCDF4.Dataset(timedep_files[0], "r") as ds:
        assert "zh" in ds.dimensions
        assert "time" in ds.dimensions
        assert len(ds.dimensions["time"]) == 2
        assert "time" in ds.variables
        assert "dpdx_ls_timedep" in ds.variables
        assert "thlsurf_timedep" in ds.variables
        assert ds.variables["dpdx_ls_timedep"].dimensions == ("zh", "time")
        assert ds.variables["thlsurf_timedep"].dimensions == ("time",)
