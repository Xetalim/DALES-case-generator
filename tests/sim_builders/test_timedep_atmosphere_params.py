from __future__ import annotations

import netCDF4
import numpy as np

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
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.modular.time_dependent import (
    TimedependentModule,
)
from modular_dales.modular.time_dependent_scalars import TimeDependentScalar
from modular_dales.Surface.surface import ConstantSurfaceTemperatureModule
from modular_dales.vars import *


def timedep_atmosphere_with_params_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation with time-dependent scalar params inside AtmosphericProfile."""
    sim = dales_simulation("timedep_atmosphere_params", machine_conf)

    sim += DefaultNamelistModule()
    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=90,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=20.0,
    )
    sim += domain_info
    timesteps = np.append(np.insert(np.linspace(0, 300, num=5), 1, 0.1), 720)
    sim += TimedependentModule(timesteps=timesteps)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=2, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=5e-4)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0, ddz=0)))
    atmo.add(InterpolatedProfile(
        variable=tke,
        z=[0, 4000, 5000],
        points=[1, 1e-8, 1e-8],
    ))

    sim += ConstantSurfaceTemperatureModule(
        thls=TimeDependentScalar(
            times=timesteps,
            values=np.append(
                np.insert(
                    np.linspace(293.15, 295.15, num=len(timesteps) - 2), 0, 293.15
                ),
                295.15,
            ),
        ),
        z0mav=0.0001,
        z0hav=0.0001,
        ps=100000,
        albedoav=0.22,
    )
    sim += atmo
    sim += TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=720)

    sim += EasyOutputModule(output_interval=10)
    return sim


def assert_timedep_atmosphere_with_params_files_written(machine_conf) -> None:
    """Test that timedep atmosphere file with TimeDependentScalar params works."""

    sim = timedep_atmosphere_with_params_case(machine_conf("timedep_atmosphere_params"))
    sim.sim_preprocessing_pipeline()

    timedep_files = list(sim.output_path.glob("input/forcings.*.nc"))
    assert len(timedep_files) == 1, (
        f"Expected 1 timedep atmosphere file, found {len(timedep_files)}"
    )

    with netCDF4.Dataset(timedep_files[0], "r") as ds:
        assert "zh" in ds.dimensions
        assert "time" in ds.dimensions
        assert "time" in ds.variables
        assert "thlsurf_timedep" in ds.variables
        assert ds.variables["thlsurf_timedep"].dimensions == ("time",)
        assert len(ds.variables["time"][:]) == 6
        assert ds.variables["time"][0] == 0.1
        assert ds.variables["time"][-1] == 720
