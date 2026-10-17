from __future__ import annotations

from modular_dales import dales_simulation
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    lin,
)
from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.Configuration.run_and_time import TimeModule
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.LSM import LandUseModification, LSMModule
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


def simple_SLURB_case(machine_conf: dict) -> dales_simulation:
    """Construct a simple simulation including a SLURB configuration."""
    sim = dales_simulation("SLURB_case", machine_conf)

    sim += DefaultNamelistModule()

    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=30,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=10.0,
    )
    sim += domain_info

    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3))
    atmo += AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2))
    atmo += AtmosphericProfile(variable=qt, shape=lin(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(variable=wa, shape=lin(surf_val=0, ddz=0))
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
    lsm += UniformSkinTemperature(293)
    lsm += UniformSoilTemperature([293, 293, 293, 293])
    lsm += UniformSoilMoisture([0.3, 0.3, 0.3, 0.3])
    lsm += LandUseModification(geometry=AllGeometry(), type="grs")
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

    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)
    sim += time_module
    return sim


def assert_slurb_files_written(machine_conf) -> None:
    """Test that SLURB files are written correctly for simple_SLURB_case."""

    sim = simple_SLURB_case(machine_conf("slurb_files"))
    sim.sim_preprocessing_pipeline()

    slurb_files = list(sim.output_path.glob("input/inslurb.*.nc"))
    assert len(slurb_files) == 1, f"Expected 1 SLURB file, found {len(slurb_files)}"
    for slurb_file in slurb_files:
        assert slurb_file.is_file(), f"Expected SLURB file {slurb_file} to exist"
        assert slurb_file.stat().st_size > 0, (
            f"Expected SLURB file {slurb_file} to be non-empty"
        )
