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
from modular_dales.Geometry.geometry_modification import (
    AllGeometry,
    RectangleIdxGeometry,
)
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.IBM.IBM import IBMModification, IBMModule
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.LandUseModifications import LandUseModification
from modular_dales.Surface.LSM.LSM import LSMModule
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.vars import *


def IBM_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation including an IBM configuration."""
    sim = dales_simulation("IBM", machine_conf)

    sim += DefaultNamelistModule()

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
    sim += domain_info

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=0.5, ddz=1e-3)))
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

    sim += atmo
    sim += TimeModule(xday=2, xtime=12.0, xyear=2023, runtime=60)

    sim += RadiationModule(
        iradiation=2,
    )
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

    ibm = IBMModule(thlibm=293.15, thlroof=293.15)
    ibm += IBMModification(geometry=AllGeometry(), height=0)
    ibm += IBMModification(
        geometry=RectangleIdxGeometry(minx=4, maxx=6, miny=4, maxy=4),
        height=20,
    )
    sim += ibm
    return sim


def assert_ibm_files_written(machine_conf) -> None:
    """Test that IBM files are written correctly."""

    sim = IBM_case(machine_conf("ibm_files"))
    sim.sim_preprocessing_pipeline()

    ibm_files = list(sim.output_path.glob("input/ibm.inp_*.nc"))
    assert len(ibm_files) == 1, f"Expected 1 IBM file, found {len(ibm_files)}"
    for ibm_file in ibm_files:
        assert ibm_file.is_file(), f"Expected IBM file {ibm_file} to exist"
        assert ibm_file.stat().st_size > 0, (
            f"Expected IBM file {ibm_file} to be non-empty"
        )
