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
import pytest
import yaml
import numpy as np
import xarray as xr
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    LS2DAtmosphereModule,
    FromLS2D,
)
from modular_dales.Atmosphere.atmosphere import TimedAtmosphereProfile
from modular_dales.Atmosphere.ls2d_atmosphere import LS2DAtmosphereModule
from modular_dales.Configuration import (
    DefaultNamelistModule,
    EasyOutputModule,
    TimeModule,
    SamplingModule,
)
from modular_dales.Configuration.output_modules import CheckSimulationModule
from modular_dales.Emission.emission import EmissionModule, EmissionTracer
import modular_dales.Emission.emission as emission
from modular_dales.Geometry import GridDales, AllGeometry
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
    case_name = "021_utrecht"
    output_directory = None
    x0, y0 = 133303, 453047  # knooppunt oudenrijn
    x1, y1 = 140146, 459189  # de bilt
    print(x1 - x0)
    print(y1 - y0)
    print(f"Size in km: {(x1 - x0) / 1000} x {(y1 - y0) / 1000}")
    if (x1 - x0) != (y1 - y0):
        print(f"Warning: domain is not square, got {(x1 - x0)} x {(y1 - y0)}")
        print("Adjusting y1 to make it square")
        y1 = y0 + (x1 - x0)

    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=64,
        xsize=x1 - x0,
        ysize=y1 - y0,
        kmax_soil=4,
        xlat=52.09135,
        xlon=5.12258,
        x0=133251.0,  # knooppunt oudenrijn
        y0=453085.0,  # knooppunt oudenrijn
        alpha=1.0,
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

    sim += domain_info
    logger.info("Added GridModule")

    sim += TimeModule(
        xday=223,
        xtime=0.0,
        xyear=2025,
        runtime=600,
        startyear=2025,
        startmonth=8,
        startday=11,
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
    lsm += LSM.LandUseModification(geometry=AllGeometry(), type="grs")
    # lsm += LSM.FromLCZ()
    sim += lsm
    # sim += LSM.SLURBModule(deep_soil_temperature=293.15)

    sim += RadiationModule(
        iradiation=5,  # 0=off, 1=simple, 4=RRTMGP
    )

    # sim += nesting

    sim += EasyOutputModule(
        output_interval=60,
        enable_output=True,
    )
    sim += SamplingModule(
        output_interval=60,
        enable_output=True,
    )

    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocx", 0)
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocy", 0)
    # External atmosphere module, not registered via sim += as openbc inits it for you.
    # sim += TimedependentModule()
    # # Configure LS2D-driven atmosphere; central_lat / central_lon and
    # # case_name will be taken from GridDales and dales_simulation.
    # atmo_ls2d = LS2DAtmosphereModule(
    #     era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
    #         "era5_path",
    #         "/Users/andrevanginkel/Documents/20_Code/28_dales_input/28.01_Dales_LSM_generator/jupyter_tests/era5_data",
    #     ),
    #     start_date=datetime.datetime(2025, 8, 11, hour=0),
    #     end_date=datetime.datetime(2025, 8, 12, hour=23),
    #     write_log=False,
    #     data_source=sim.machine_conf.get("ls2d_conf", {}).get("data_source", "CDS"),
    #     n_av=0,
    #     method="2nd",
    # )
    # sim += atmo_ls2d
    # lsm += FromLS2D()

    # Base AtmosphereModule; LS2DAtmosphereModule will inject LS2D-based
    # initial profiles here so that ``init.<exp_id>.nc`` is written via
    # the standard AtmosphereModule machinery.
    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(variable=ua, shape="lin", params=dict(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(variable=ug, shape="lin", params=dict(surf_val=0, ddz=0))
    atmo += AtmosphericProfile(
        variable=va, shape="lin", params=dict(surf_val=3, ddz=1e-3)
    )
    atmo += AtmosphericProfile(
        variable=vg, shape="lin", params=dict(surf_val=3, ddz=1e-3)
    )
    # atmo += AtmosphericProfile(
    #     variable=thetal, shape="lin", params=dict(surf_val=293.15, ddz=1e-2)
    # )
    atmo += InterpolatedProfile(
        variable=thetal,
        z=[0, 200, 210, 500],
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
    sim += atmo
    # openbc = do_openboundary(
    #     time0="2025-08-11T00:00:00",
    #     start="2025-08-11T00:00:00",
    #     end="2025-08-12T23:10:00",
    #     e12=0.1,
    #     dxint=domain_info.xsize / domain_info.itot,
    #     dyint=domain_info.ysize / domain_info.jtot,
    # )
    # openbc += Nest_in_AtmosphereProfiles(
    #     atmosphere_module=atmo,
    #     noise_boundaries=["south", "west", "east", "north"],
    #     noise_std=0.1,
    #     noise_seed=42,
    #     noise_variables=["thl"],
    # )
    # sim += openbc

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
