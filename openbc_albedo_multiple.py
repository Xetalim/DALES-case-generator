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
    FielddumpModule,
    StatsModule,
    TimestatModule,
    TimeModule,
    SamplingModule,
)
from modular_dales.Configuration.output_modules import CheckSimulationModule
from modular_dales.Emission.emission import EmissionModule, EmissionTracer
import modular_dales.Emission.emission as emission
from modular_dales.Geometry import GridDales
from modular_dales.LBC import Nest_in_Dales, NestingTopology, do_openboundary
from modular_dales.LBC.openbc import Nest_in_AtmosphereProfiles
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.SLuRB.slurb import SLURBModification
from modular_dales.Surface.LSM.SLuRB.slurb import SLURBModification
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

    #   logical :: sgs_surface_shear_ustar = .false. !< switch for using local du/dz and dv/dz from ustar in shear production in SGS TKE equation
    #   logical :: sgs_surface_buoyancy = .false. !< switch for using surface thlflux in buoyancy production in SGS TKE equation
    #   logical :: sgs_surface_shear_virt_velocity = .false. !< switch for using virtual surface velocity in shear production in SGS TKE equation
    base_conf = {
        "sgs_surface_shear_ustar": False,
        "sgs_surface_buoyancy": False,
        "sgs_surface_shear_virt_velocity": False,
        "resolution": 50,  # m
    }
    confs = []
    for shear_option in ["ustar", "dudz"]:
        for buoy_option in [True, False]:
            for resolution in [50, 100, 200]:
                conf = base_conf.copy()
                conf["resolution"] = resolution
                conf["sgs_surface_shear_ustar"] = shear_option == "ustar"
                conf["sgs_surface_shear_virt_velocity"] = shear_option == "virt"
                conf["sgs_surface_shear_dudz"] = shear_option == "dudz"
                conf["sgs_surface_buoyancy"] = buoy_option
                confs.append(conf)
    # confs = []
    # loop over resolution options

    for sg_conf in confs:
        # Create minimal configuration with just case name and output directory
        confname = f"shear_{'ustar' if sg_conf['sgs_surface_shear_ustar'] else 'virt' if sg_conf['sgs_surface_shear_virt_velocity'] else 'dudz'}_buoyancy_{'flux' if sg_conf['sgs_surface_buoyancy'] else 'dthvdz'}_res_{sg_conf['resolution']}m"
        case_name = f"005_utrecht_multiple/{confname}"
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

        hor_size = 1600
        domain_info = GridDales(
            itot=int(hor_size / sg_conf["resolution"]),
            jtot=int(hor_size / sg_conf["resolution"]),
            kmax=128,
            xsize=hor_size,
            ysize=hor_size,
            kmax_soil=4,
            xlat=52.09135,
            xlon=5.12258,
            x0=133251.0,  # knooppunt oudenrijn
            y0=453085.0,  # knooppunt oudenrijn
            alpha=1.013,
            dz0=10,
            proj4="epsg:28992",
        )
        # print(f"{domain_info.zt=}")

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
            xtime=6.0,
            xyear=2025,
            runtime=3600 * 12,  # 12 hours in seconds
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
        lsm += LSM.UniformSkinTemperature(
            293.3
        )  # this will be overriden by LS2D-driven skin temperature
        lsm += LSM.UniformSoilMoisture(
            [0.02, 0.02, 0.02, 0.02]
        )  # this will be overriden by LS2D-driven soil moisture
        lsm += LSM.UniformSoilTemperature(
            [293.3, 293.3, 293.3, 293.3]
        )  # this will be overriden by LS2D-driven soil temperature
        # lsm += LSM.FromLCZ()
        lsm += LSM.LandUseModification("all", type="grs", params={})
        sim += lsm
        # slurb = LSM.SLURBModule(
        #     deep_soil_temperature=293.15, building_indoor_temperature=273.15 + 30
        # )
        # sim += slurb

        sim += RadiationModule(
            iradiation=5,  # 0=off, 1=simple, 4=RRTMGP
        )

        # sim += EasyOutputModule(
        #     output_interval=10,
        #     enable_output=True,
        # )
        sim += FielddumpModule(lfielddump=True, dtav=3600)
        sim += StatsModule(enabled=True, dtav=30, timeav=30)
        sim += TimestatModule(enabled=True, dtav=30)
        # sim += SamplingModule(
        #     output_interval=10,
        #     enable_output=True,
        # )

        set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocx", 0)
        set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocy", 0)
        set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "NAMSLURB", "dtav_slurb", 10
        )
        # # External atmosphere module, not registered via sim += as openbc inits it for you.
        # sim += TimedependentModule(ltimedep=True)
        # # Configure LS2D-driven atmosphere; central_lat / central_lon and
        # # case_name will be taken from GridDales and dales_simulation.
        # atmo_ls2d = LS2DAtmosphereModule(
        #     era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
        #         "era5_path",
        #         "/Users/andrevanginkel/Documents/20_Code/28_dales_input/28.01_Dales_LSM_generator/jupyter_tests/era5_data",
        #     ),
        #     start_date=datetime.datetime(2025, 8, 11, hour=0),
        #     end_date=datetime.datetime(2025, 8, 16, hour=0),
        #     write_log=False,
        #     data_source=sim.machine_conf.get("ls2d_conf", {}).get("data_source", "CDS"),
        #     n_av=0,
        #     method="2nd",
        #     do_nudging=True,
        # )
        # sim += atmo_ls2d
        # lsm += FromLS2D()

        # Base AtmosphereModule; LS2DAtmosphereModule will inject LS2D-based
        # initial profiles here so that ``init.<exp_id>.nc`` is written via
        # the standard AtmosphereModule machinery.
        atmo = AtmosphereModule()
        atmo += AtmosphericProfile(
            variable=ua, shape="lin", params=dict(surf_val=4, ddz=0)
        )
        atmo += AtmosphericProfile(
            variable=va, shape="lin", params=dict(surf_val=4, ddz=0)
        )
        atmo += AtmosphericProfile(
            variable=ug, shape="lin", params=dict(surf_val=4, ddz=0)
        )
        atmo += AtmosphericProfile(
            variable=vg, shape="lin", params=dict(surf_val=4, ddz=0)
        )
        # atmo += AtmosphericProfile(
        #     variable=thetal, shape="lin", params=dict(surf_val=293.15, ddz=1e-2)
        # )
        atmo += InterpolatedProfile(
            variable=thetal,
            z=[0, 400, 410, 3000, 4000, 5000],
            points=[293.15, 293.0, 298.15, 301.15, 301.15, 301.15],
        )

        atmo += AtmosphericProfile(
            variable=qt, shape="lin", params=dict(surf_val=0.000, ddz=0)
        )

        atmo += AtmosphericProfile(
            variable=wa, shape="lin", params=dict(surf_val=0, ddz=0)
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

        sim += atmo

        set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
        )
        set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "physics", "lcoriol", True
        )
        set_nml_section(sim.nml, sim.nml_docs, "user_defined", "surface", "thls", 293.3)
        set_nml_section(
            sim.nml,
            sim.nml_docs,
            "user_defined",
            "namsubgrid",
            "sgs_surface_fix",
            False,
        )
        # set_nml_section(
        #     sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "ldelta", False
        # )
        # set_nml_section(
        #     sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "lanisotrop", True
        # )
        # set_nml_section(
        #     sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "lanisotrop", False
        # )  # anisotropic SGS speeds up the simulation a lot, and we don't really care about the small inconsistencies, so we set to True

        # set_nml_section(
        #     sim.nml, sim.nml_docs, "user_defined", "PHYSICS", "ltimedep", True
        # )  # we don't want to have time dependent forcing somehow due to LS2D, as we will inject time series via the openboundary module, so we set ltimedep to False to be sure that we don't get any unexpected time dependence from the physics module.
        # set_nml_section(
        #     sim.nml, sim.nml_docs, "user_defined", "NAMNUDGE", "lnudge", True
        # )  # we don't want to have nudging, so we explicitly disable
        set_nml_section(
            sim.nml,
            sim.nml_docs,
            "shear",
            "NAMSUBGRID",
            "sgs_surface_shear_ustar",
            sg_conf["sgs_surface_shear_ustar"],
        )
        set_nml_section(
            sim.nml,
            sim.nml_docs,
            "shear",
            "NAMSUBGRID",
            "sgs_surface_shear_virt_velocity",
            sg_conf["sgs_surface_shear_virt_velocity"],
        )
        set_nml_section(
            sim.nml,
            sim.nml_docs,
            "buoyancy",
            "NAMSUBGRID",
            "sgs_surface_buoyancy",
            sg_conf["sgs_surface_buoyancy"],
        )
        set_nml_section(
            sim.nml,
            sim.nml_docs,
            "restart",
            "RUN",
            "trestart",
            -1,
        )
        sim += CheckSimulationModule(check_interval=60)
        sim.init_output_folder()
        sim.setup_module_links()
        sim.do_config()
        sim.check_settings()
        sim.prepare_all_calculations()
        sim.write_module_files()
        sim.apply_job_configuration()
        sim.write_simulation_files()
        wd = os.getcwd()
        os.chdir(sim.output_path.as_posix())
        subprocess.run("./job.001", check=True)
        os.chdir(wd)
