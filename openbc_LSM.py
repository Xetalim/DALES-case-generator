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
from modular_dales.Geometry import GridDales, AllGeometry, RectangleRealGeometry
from modular_dales.LBC import Nest_in_Dales, NestingTopology, do_openboundary
from modular_dales.LBC.openbc import Nest_in_AtmosphereProfiles
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.logging_wrapper import setup_logging
from modular_dales.modular import (
    dales_simulation,
    TimeDependentScalar,
    TimedependentModule,
)
from modular_dales.Surface import ConstantSurfaceTemperatureModule
import modular_dales.Surface.LSM as LSM
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
    case_name = "019_test"
    output_directory = None

    supergrid = GridDales(
        itot=64,
        jtot=64,
        kmax=80,
        xsize=640.0,
        ysize=640.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=20,
        # "proj4": "+proj=..."  # optional
    )

    # this is the grid that is inside us
    subgrid = GridDales(
        itot=64,
        jtot=64,
        kmax=70,
        xsize=320.0,
        ysize=320.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=160.0,
        y0=160.0,
        alpha=1.0,
        dz0=20,
        # "proj4": "+proj=..."  # optional
    )

    nesting = NestingTopology()
    # this is us
    nesting += supergrid
    nesting.my_idx = nesting.nestings.index(supergrid)

    nesting += subgrid

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
    atmo += AtmosphericProfile(variable=ug, shape="lin", params=dict(surf_val=3, ddz=0))
    atmo += AtmosphericProfile(variable=vg, shape="lin", params=dict(surf_val=3, ddz=0))
    atmo += AtmosphericProfile(variable=va, shape="lin", params=dict(surf_val=3, ddz=0))
    # atmo += AtmosphericProfile(
    #     variable=thetal, shape="lin", params=dict(surf_val=293.15, ddz=1e-2)
    # )
    atmo += InterpolatedProfile(
        variable=thetal,
        z=[0, 400, 410, 1600],
        points=[293.15, 293.15, 298.15, 301.15],
    )
    atmo += AtmosphericProfile(
        variable=qt, shape="lin", params=dict(surf_val=0.0018, ddz=0)
    )
    atmo += AtmosphericProfile(variable=wa, shape="lin", params=dict(surf_val=0, ddz=0))
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
        runtime=3700,
        startyear=2023,
        startmonth=1,
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
    lsm += LSM.UniformSkinTemperature(293.15)
    lsm += LSM.UniformSoilMoisture([0.2, 0.2, 0.2, 0.2])
    lsm += LSM.UniformSoilTemperature([293.15, 293.15, 293.15, 293.15])
    lsm += LSM.LandUseModification(geometry=AllGeometry(), type="aqu")
    # lsm += LSM.LandUseModification(
    #     geometry=RectangleRealGeometry(minx=160, maxx=480, miny=160, maxy=480),
    #     type="grs",
    # )
    sim += lsm

    sim += RadiationModule(
        iradiation=4,  # 0=off, 1=simple, 4=RRTMGP
    )

    # sim += nesting

    sim += EasyOutputModule(
        output_interval=5,
        enable_output=True,
    )
    sim += SamplingModule(
        output_interval=60,
        enable_output=True,
    )

    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocx", 0)
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocy", 0)
    # External atmosphere module, not registered via sim += as openbc inits it for you.
    atmo_external = AtmosphereModule()
    for prof in atmo.shaped_profiles:
        if prof.variable != co2:
            atmo_external.shaped_profiles.append(prof)
    for prof in atmo.interpolated_profiles:
        if prof.variable != co2:
            atmo_external.interpolated_profiles.append(prof)
    timesteps = np.arange(0, 3600 + 30, step=30)
    atmo_external += InterpolatedProfile(
        variable=co2,
        z=[0, 400, 410, 1600],
        points=[
            0,
            0,
            0,
            TimeDependentScalar(
                times=timesteps,
                values=np.zeros_like(timesteps),
            ),
        ],
    )

    # Basic openboundary configuration using the external atmosphere
    openbc = do_openboundary(
        time0="2023-01-01T12:00:00",
        start="2023-01-01T12:00:00",
        end="2023-01-02T12:00:00",
        e12=1,
        tauh=20,
        taum=0,
        dxint=supergrid.xsize,  # / supergrid.itot,
        dyint=supergrid.ysize,  # / supergrid.jtot,
        tracernames=["other"],
        lper=[True, False],
        lsynturb=False,
    )

    openbc += Nest_in_AtmosphereProfiles(
        atmosphere_module=atmo_external,
        variable_mapping={"other": "other"},
        noise_boundaries=["south", "west", "east", "north"],
        noise_variables=["thl"],
        noise_std=1,
        noise_seed=0,
        noise_minzt=0,
        noise_maxzt=400,
        add_to_top_thl=0.5,  # add 0.5 K to the top boundary thl to make sure we don't get a downdraft along the top everywhere
    )
    sim += openbc

    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
    )
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "physics", "lcoriol", False)

    # if sim.nml.get("namchecksim") is None:
    #     sim.nml["namchecksim"] = {}
    # sim.nml["namchecksim"]["tcheck"] = 60
    sim += CheckSimulationModule(check_interval=60)
    sim.init_output_folder()
    sim.setup_module_links()
    sim.do_config()
    sim.check_settings()
    sim.prepare_all_calculations()

    # ref = sim.retrieve_module(do_openboundary)
    # time, zt, xt = xr.broadcast(
    #     ref.boundaries.time, ref.boundaries.zt, ref.boundaries.xt
    # )

    # def triangle_function(x, x0, width):

    #     return np.maximum(0, 1 - np.abs(x - x0) / width)

    # time_prof = triangle_function(np.mod(time, 410), 300, 100)
    # zt_prof = triangle_function(zt, 90, 30)
    # xt_prof = triangle_function(xt, 320, 50)
    # ref.boundaries.othersouth[:, :, :] = time_prof * zt_prof * xt_prof
    sim.write_module_files()
    sim.apply_job_configuration()
    sim.write_simulation_files()

    wd = os.getcwd()
    os.chdir(sim.output_path.as_posix())
    subprocess.run("./job.001", check=True)
    run_result = subprocess.run(
        ["/Users/andrevanginkel/bin/combine.sh", "run_001"], check=False
    )
    print(run_result.stderr)
    os.chdir(wd)
    exit()
    # Machine configuration (would normally come from machine_conf.yaml)
    # Create simulation instance
    case_name = "018_test"
    sim2 = dales_simulation(case_name, machine_conf)
    logger.info(f"Created simulation with case: {case_name}")

    # Add modules - this demonstrates the += operator for module registration
    sim2 += DefaultNamelistModule()
    logger.info("Added DefaultNamelistModule")

    sim2 += subgrid
    logger.info("Added GridModule")

    emis = EmissionModule()

    emis += EmissionTracer(
        name="other",
        long_name="Carbon Dioxide (CO2)",
        unit="ppm",
        molar_mass=44.009,
        lemis=True,
    )
    sim2 += emis
    atmo2 = AtmosphereModule()
    for prof in atmo.shaped_profiles:
        if prof.variable != co2:
            atmo2.shaped_profiles.append(prof)
    for prof in atmo.interpolated_profiles:
        if prof.variable != co2:
            atmo2.interpolated_profiles.append(prof)
    sim2 += atmo2
    # atmo = AtmosphereModule()
    # atmo += AtmosphericProfile(variable=ua, shape="lin", params=dict(surf_val=1, ddz=0))
    # # atmo += AtmosphericProfile(variable=ug, shape="lin", params=dict(surf_val=1, ddz=0))
    # atmo += AtmosphericProfile(variable=va, shape="lin", params=dict(surf_val=0, ddz=0))
    # # atmo += AtmosphericProfile(variable=vg, shape="lin", params=dict(surf_val=0, ddz=0))
    # # atmo += AtmosphericProfile(
    # #     variable=thetal, shape="lin", params=dict(surf_val=293.15, ddz=1e-2)
    # # )
    # atmo += InterpolatedProfile(
    #     variable=thetal,
    #     z=[0, 200, 210, 500],
    #     points=[293.15, 293.15, 298.15, 301.15],
    # )
    # atmo += AtmosphericProfile(
    #     variable=qt, shape="lin", params=dict(surf_val=0.0018, ddz=0)
    # )
    # atmo += AtmosphericProfile(variable=wa, shape="lin", params=dict(surf_val=0, ddz=0))
    # atmo += InterpolatedProfile(
    #     variable=tke,
    #     z=[0, 4000, 5000],
    #     points=[1, 1e-8, 1e-8],
    # )
    # atmo += AtmosphericProfile(
    #     variable=w,
    #     shape="lin",
    #     params=dict(surf_val=0.0, ddz=0.0),
    # )
    # atmo += InterpolatedProfile(
    #     variable=co2,
    #     z=[0, 200, 210, 500],
    #     points=[0, 0, 0, 0],
    # )

    # sim2 += atmo
    sim2 += TimeModule(
        xday=1,
        xtime=12.0,
        xyear=2023,
        runtime=3700,
        startyear=2023,
        startmonth=1,
        startday=1,
    )

    # sim += RadiationModule(
    #     iradiation=4,  # 0=off, 1=simple, 4=RRTMGP
    # )
    sim2 += ConstantSurfaceTemperatureModule(
        thls=293.15, z0mav=0.0001, z0hav=0.0001, ps=100000
    )

    set_nml_section(
        sim2.nml, sim2.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
    )
    set_nml_section(
        sim2.nml, sim2.nml_docs, "user_defined", "physics", "lcoriol", False
    )

    nesting = NestingTopology()
    # this is us
    nesting += supergrid

    nesting += subgrid
    nesting.my_idx = nesting.nestings.index(subgrid)
    sim2 += nesting

    openbc = do_openboundary(
        # tracernames=["tracer1", "tracer2"],
        time0="2023-01-01T12:00:00",
        start="2023-01-01T12:00:00",
        end="2023-01-02T12:00:00",
        e12=0.01,
        dxint=subgrid.xsize,  # / subgrid.itot,
        dyint=subgrid.ysize,  # / subgrid.jtot,
        tauh=0,
        taum=0,
        dxturb=subgrid.xsize / subgrid.itot,
        dyturb=subgrid.ysize / subgrid.jtot,
        linithetero=True,
        tracernames=["other"],
    )
    openbc += Nest_in_Dales(
        inpath=sim.output_path / "input/",
        inpath_coarse=sim.output_path / "input/",
        outpath_coarse=sim.output_path / "run_001/",
        outpath_coarse_old=sim.output_path / "run_001/",
    )
    sim2 += openbc
    sim2 += EasyOutputModule(
        output_interval=5,
        enable_output=True,
    )
    sim2 += SamplingModule(
        output_interval=60,
        enable_output=True,
    )
    sim2 += CheckSimulationModule(check_interval=60)
    # if sim2.nml.get("RUN") is None:
    #     sim2.nml["RUN"] = {}
    # if sim2.nml["RUN"].get("nprocx") is None:
    #     sim2.nml["RUN"]["nprocx"] = 0
    # if sim2.nml["RUN"].get("nprocy") is None:
    #     sim2.nml["RUN"]["nprocy"] = 0
    # if sim2.nml.get("namchecksim") is None:
    #     sim2.nml["namchecksim"] = {}
    # sim2.nml["namchecksim"]["tcheck"] = 60
    # if sim2.nml.get("solver") is None:
    #     sim2.nml["solver"] = {}
    # sim2.nml["solver"]["solver_id"] = 100
    # sim2.nml["solver"]["tolerance"] = 1e-4

    sim2.sim_preprocessing_pipeline()

    wd = os.getcwd()
    os.chdir(sim2.output_path.as_posix())
    subprocess.run("./job.001", check=True)
    run_result = subprocess.run(
        ["/Users/andrevanginkel/bin/combine.sh", "run_001"], check=True
    )
    # print(run_result.stderr)
    os.chdir(wd)
    # txt = sim.save_sim_to_yaml()
    # with open("test_sim.yaml", "w") as f:
    #     f.write(txt)
    # sim2 = dales_simulation.load_sim_from_yaml(txt, machine_conf=machine_conf)
    # print(sim2)

    # yaml.dump(sim2, open("test_sim2_dump.yaml", "w"))
    # yaml.dump(sim, open("test_sim_dump.yaml", "w"))
