"""Example demonstrating complete DALES simulation module usage.

This example shows how to:
1. Create a simulation instance with configuration
2. Add all available modules
3. Configure, validate, prepare, and write output

The modular system uses dataclasses where:
- Each module is a dataclass with optional `sim` parent reference
- Modules expose properties to access sim.config, sim.grid, sim.output_path, sim.nml
- Namelist values are declared via field metadata: metadata={"nml": "SECTION", "key": "name"}
- Runtime-only fields use: field(default=None, init=False, repr=False)
"""

import datetime
import logging
import os
import subprocess

import numpy as np
import pytest
import xarray as xr
import yaml

import modular_dales.Emission.emission as emission
import modular_dales.Surface.LSM as LSM
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    FromLS2D,
    InterpolatedProfile,
    LS2DAtmosphereModule,
)
from modular_dales.Atmosphere.atmosphere import TimedAtmosphereProfile
from modular_dales.Atmosphere.ls2d_atmosphere import LS2DAtmosphereModule
from modular_dales.Configuration import (
    CapeModule,
    ColumnStatisticsOutputModule,
    CrossSectionOutputModule,
    DefaultNamelistModule,
    EasyOutputModule,
    FielddumpModule,
    LSMCrossModule,
    RadfieldModule,
    SamplingModule,
    StatsModule,
    TimestatModule,
    TimeModule,
)
from modular_dales.Configuration.output_modules import (
    CheckSimulationModule,
    VirtualMeasurementOutputModule,
)
from modular_dales.Emission.emission import EmissionModule, EmissionTracer
from modular_dales.Geometry import GridDales
from modular_dales.IBM.IBM import FromAHN, IBMModule
from modular_dales.LBC import Nest_in_Dales, NestingTopology, do_openboundary
from modular_dales.LBC.openbc import Nest_in_AtmosphereProfiles
from modular_dales.logging_wrapper import setup_logging
from modular_dales.modular import (
    TimedependentModule,
    TimeDependentScalar,
    dales_simulation,
)
from modular_dales.modular.simulation_module import set_nml_section
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface import ConstantSurfaceTemperatureModule
from modular_dales.Surface.LSM.LSM import FromLCZ
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.Surface.LSM.SLuRB.slurb import SLURBModification, SLURBModule
from modular_dales.vars import *
from modular_dales.vars import get_all_vars, register_var

setup_logging("logging.yaml")
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # pytest.skip("unsupported case")
    with open("machine_conf.yaml", "r") as file:
        machine_conf = yaml.safe_load(file)

    # Create minimal configuration with just case name and output directory
    case_name = "002_may_2026"
    output_directory = None
    xsize = 25600
    ysize = 25600
    middle = 123366.770928, 442217.642717
    x0, y0 = middle[0] - xsize / 2, middle[1] - ysize / 2
    x1, y1 = middle[0] + xsize / 2, middle[1] + ysize / 2
    print(f"Size in km: {(x1 - x0) / 1000} x {(y1 - y0) / 1000}")

    grid = GridDales(
        itot=256,
        jtot=256,
        kmax=128,
        xsize=xsize,
        ysize=ysize,
        kmax_soil=4,
        xlat=51.968090,
        xlon=4.929183,
        x0=x0,
        y0=y0,
        alpha=1.019,
        dz0=20,
        proj4="epsg:28992",
    )
    # Machine configuration (would normally come from machine_conf.yaml)
    # Create simulation instance
    sim = dales_simulation(case_name, machine_conf)
    logger.info(f"Created simulation with case: {case_name}")

    # Add modules - this demonstrates the += operator for module registration
    sim += DefaultNamelistModule()
    logger.info("Added DefaultNamelistModule")

    sim += grid
    logger.info("Added GridModule")

    sim += TimeModule(
        xtime=0,
        xday=121,
        xyear=2026,
        runtime=3600 * 24 * 31 - 3600,  # 4 hours in seconds
        startyear=2026,
        startmonth=5,
        startday=1,
    )

    lsm = LSM.LSMModule(
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        albedoav=0.2,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
    )
    lsm += FromLS2D()
    lsm += FromLCZ()

    lsm += UniformSkinTemperature(293)
    lsm += UniformSoilTemperature([293, 293, 293, 293])
    lsm += UniformSoilMoisture([0.2, 0.2, 0.2, 0.2])

    sim += lsm

    sim += SLURBModule(
        deep_soil_temperature=283.15, building_indoor_temperature=273.15 + 20
    )

    sim += FielddumpModule(lfielddump=True, dtav=1200)
    sim += CapeModule(enabled=True, dtav=300)
    sim += LSMCrossModule(enabled=True, dtav=300)
    sim += CrossSectionOutputModule(cross_enabled=True)
    sim += StatsModule(enabled=True)
    sim += TimestatModule(enabled=True)
    sim += RadfieldModule(enabled=True, timeav=300)

    sim += TimedependentModule(ltimedep=True)

    atmo_ls2d = LS2DAtmosphereModule(
        era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
            "era5_path",
            "/Users/andrevanginkel/Documents/20_Code/28_dales_input/28.01_Dales_LSM_generator/jupyter_tests/era5_data",
        ),
        start_date=datetime.datetime(2026, 5, 1, hour=0),
        end_date=datetime.datetime(2026, 5, 31, hour=23),
        write_log=False,
        data_source=sim.machine_conf.get("ls2d_conf", {}).get("data_source", "CDS"),
        n_av=0,
        method="2nd",
        do_nudging=True,
    )
    sim += atmo_ls2d
    atmo = AtmosphereModule()
    sim += atmo

    sim += RadiationModule(iradiation=5)

    sim += ColumnStatisticsOutputModule(x=middle[0], y=middle[1])
    sim += VirtualMeasurementOutputModule(x=middle[0], y=middle[1], enabled=True)

    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
    )
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "nammicrophysics", "imicro", 2
    )
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "nammicrophysics", "l_sb", True
    )
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "PHYSICS", "ltimedep", True)
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "NAMNUDGE", "lnudge", True)
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "lanisotrop", True
    )

    sim += CheckSimulationModule(check_interval=120)
    sim.sim_preprocessing_pipeline()
    exit()
    wd = os.getcwd()
    os.chdir(sim.output_path.as_posix())
    subprocess.run("./job.001", check=True)
    os.chdir(wd)
