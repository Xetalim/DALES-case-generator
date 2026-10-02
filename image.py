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

import logging
import os
import subprocess
import pytest
import yaml
import numpy as np
import xarray as xr
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
)
from modular_dales.Atmosphere.atmosphere import TimedAtmosphereProfile
from modular_dales.Configuration import (
    DefaultNamelistModule,
    EasyOutputModule,
    TimeModule,
    SamplingModule,
)
from modular_dales.Configuration.output_modules import CheckSimulationModule
from modular_dales.Emission.emission import EmissionModule, EmissionTracer
import modular_dales.Emission.emission as emission
from modular_dales.Geometry import GridDales
from modular_dales.LBC import Nest_in_Dales, NestingTopology, do_openboundary
from modular_dales.LBC.openbc import Nest_in_AtmosphereProfiles
from modular_dales.logging_wrapper import setup_logging
from modular_dales.modular import (
    dales_simulation,
    TimeDependentScalar,
    TimedependentModule,
)
from modular_dales.Surface import ConstantSurfaceTemperatureModule
from modular_dales.modular.simulation_module import set_nml_section
from modular_dales.vars import *
from modular_dales.vars import register_var
from modular_dales.vars import get_all_vars

setup_logging("logging.yaml")
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# logger.info("Added TimeModule")

# # Execute workflow
# logger.info("\n--- Initializing simulation ---")
# sim.init_output_folder()
# sim.setup_module_links()
# logger.info("--- Configuring modules ---")
# sim.do_config()

# logger.info("--- Checking settings ---")
# sim.check_settings()

# logger.info("--- Preparing calculations ---")
# sim.prepare_all_calculations()
# sim.apply_job_configuration()
# logger.info("--- Writing files ---")
# sim.write_module_files()
# sim.write_simulation_files()

# logger.info("\nBasic simulation completed successfully!")


if __name__ == "__main__":
    # pytest.skip("unsupported case")
    with open("machine_conf.yaml", "r") as file:
        machine_conf = yaml.safe_load(file)
    """Create a basic DALES simulation with minimal configuration.

    This example demonstrates:
    - Creating a simulation with config and machine settings
    - Adding default and grid modules
    - Executing the simulation workflow
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Basic Simulation Setup")
    logger.info("=" * 70)

    # Create minimal configuration with just case name and output directory
    case_name = "0001_image"
    output_directory = None

    supergrid = GridDales(
        itot=64,
        jtot=64,
        kmax=80,
        xsize=320.0,
        ysize=320.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=20,
        # "proj4": "+proj=..."  # optional
    )
    # Machine configuration (would normally come from machine_conf.yaml)
    # Create simulation instance
    sim = dales_simulation(case_name, machine_conf)
    logger.info(f"Created simulation with case: {case_name}")

    # Add modules - this demonstrates the += operator for module registration
    sim += DefaultNamelistModule()
    logger.info("Added DefaultNamelistModule")

    sim += supergrid
    logger.info("Added GridModule")

    co2 = VariableDefinition(
        "other",
        "Carbon Dioxide (CO2)",
        unit="ppm",
        can_be_time_dependent=True,
        must_only_be_time_dependent=False,
        time_dependent_name="other",
    )
    # ALL_VARIABLES = (*ALL_VARIABLES, co2)
    # ATMO_VARS_BY_NAME = {v.name: v for v in ALL_VARIABLES}
    register_var(co2)
    emis = EmissionModule()

    emis += EmissionTracer(
        name="other",
        long_name="Carbon Dioxide (CO2)",
        unit="ppm",
        molar_mass=44.009,
        lemis=True,
    )
    emis += emission.EmissionPointSource(
        tracer_name="other",
        x_idx=32,
        y_idx=8,
        height=10,
        temperature=294.0,
        volume=1.0,
        emission=10.0,
        stack_exit_area=1.0,
    )
    sim += emis
    # time_mod = TimedependentModule(timesteps=[0, 50, 600, 9600])
    # sim += time_mod
    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(variable=ua, shape="lin", params=dict(surf_val=3, ddz=0))
    atmo += AtmosphericProfile(variable=va, shape="lin", params=dict(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(variable=ug, shape="lin", params=dict(surf_val=3, ddz=0))
    atmo += AtmosphericProfile(
        variable=ua_nudge, shape="lin", params=dict(surf_val=3, ddz=0)
    )
    atmo += AtmosphericProfile(
        variable=va_nudge, shape="lin", params=dict(surf_val=3, ddz=0)
    )
    # atmo += AtmosphericProfile(
    #     variable=thetal, shape="lin", params=dict(surf_val=293.15, ddz=1e-2)
    # )
    atmo += InterpolatedProfile(
        variable=thetal,
        z=[0, 400, 410, 1600],
        points=[293.15, 293.15, 298.15, 301.15],
    )
    atmo += InterpolatedProfile(
        variable=thl_nudge,
        z=[0, 400, 410, 1600],
        points=[293.15, 293.15, 298.15, 301.15],
    )
    atmo += AtmosphericProfile(
        variable=qt, shape="lin", params=dict(surf_val=0.0018, ddz=0)
    )
    atmo += AtmosphericProfile(
        variable=qt_nudge, shape="lin", params=dict(surf_val=0.0018, ddz=0)
    )
    atmo += AtmosphericProfile(variable=wa, shape="lin", params=dict(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(
        variable=wa_nudge, shape="lin", params=dict(surf_val=0, ddz=0)
    )
    atmo += InterpolatedProfile(
        variable=tke,
        z=[0, 4000, 5000],
        points=[1, 1e-8, 1e-8],
    )
    atmo += AtmosphericProfile(
        variable=w,
        shape="lin",
        params=dict(surf_val=0.0, ddz=0.0),
    )
    atmo += InterpolatedProfile(
        variable=co2,
        z=[0, 400, 410, 1600],
        points=[0, 0, 0, 0],
    )

    sim += atmo
    sim += TimeModule(
        xday=1,
        xtime=12.0,
        xyear=2023,
        runtime=3600,  # 3700 - 1800,
        startyear=2023,
        startmonth=1,
        startday=1,
    )
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15, z0mav=0.0001, z0hav=0.0001, ps=100000
    )

    sim += EasyOutputModule(
        output_interval=5,
        enable_output=True,
    )

    # External atmosphere module, not registered via sim += as openbc inits it for you.
    atmo_external = AtmosphereModule()
    for prof in atmo.shaped_profiles:
        if prof.variable != co2:
            atmo_external.shaped_profiles.append(prof)
    for prof in atmo.interpolated_profiles:
        if prof.variable != co2:
            atmo_external.interpolated_profiles.append(prof)
    atmo_external += InterpolatedProfile(
        variable=co2,
        z=[0, 400, 410, 1600],
        points=[
            0,
            0,
            TimeDependentScalar(times=np.linspace(0, 9600, 960), values=np.zeros(960)),
            0,
        ],
    )

    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
    )
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "physics", "lcoriol", False)
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "ldelta", False
    )
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "lanisotrop", True
    )
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocx", 0)
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocy", 0)
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "PHYSICS", "ltimedep", False
    )  # we don't want to have time dependent forcing somehow due to LS2D, as we will inject time series via the openboundary module, so we set ltimedep to False to be sure that we don't get any unexpected time dependence from the physics module.
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "NAMNUDGE", "lnudge", False
    )  # we don't want to have nudging, so we explicitly disable
    # if sim.nml.get("namchecksim") is None:
    #     sim.nml["namchecksim"] = {}
    # sim.nml["namchecksim"]["tcheck"] = 60
    sim += CheckSimulationModule(check_interval=60)
    sim.init_output_folder()
    sim.setup_module_links()
    sim.do_config()
    sim.check_settings()
    sim.prepare_all_calculations()
    sim.write_module_files()
    sim.apply_job_configuration()
    sim.write_simulation_files()
    # exit()
    wd = os.getcwd()
    os.chdir(sim.output_path.as_posix())
    subprocess.run("./job.001", check=True)
    os.chdir(wd)
