"""Example: DALES case forced by KNMI HARMONIE via HarmonieAtmosphereModule.

Usage:
    1. Set KNMI_ML_GLOB and KNMI_SFC_GLOB to your pre-converted NetCDF files.
    2. Run: python LS2D_knmi.py

This script follows the structure of openbc_knmi.py, but does not use open
boundaries. Instead, it generates DALES init/forcings files through
HarmonieAtmosphereModule.
"""

import logging
from datetime import datetime

import yaml

from modular_dales import dales_simulation
from modular_dales.Atmosphere import (
    AtmosphereModule,
    HarmonieAtmosphereModule,
    FromLS2D,
)
from modular_dales.Atmosphere.atmosphere import InterpolatedProfile
from modular_dales.Configuration import (
    DefaultNamelistModule,
    EasyOutputModule,
    TimeModule,
)
from modular_dales.Configuration.output_modules import CrossSectionOutputModule
from modular_dales.Geometry import GridDales
from modular_dales.IBM.IBM import FromAHN, IBMModule
from modular_dales.Surface import ConstantSurfaceTemperatureModule
from modular_dales.logging_wrapper import setup_logging
from modular_dales.modular.simulation_module import set_nml_section
from modular_dales.modular.time_dependent import TimedependentModule
from modular_dales.vars import ua, va, thetal, qt, wa, tke

setup_logging("logging.yaml")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Glob patterns or single-file paths for pre-converted KNMI HARMONIE NetCDF.
KNMI_ML_GLOB = "/Users/andrevanginkel/Documents/30_Data/32_campus_case/32.01_harm_2026032618/HARM43_V1_P5_2026032618/HA43_N55ML.nc"
KNMI_SFC_GLOB = "/Users/andrevanginkel/Documents/30_Data/32_campus_case/32.01_harm_2026032618/HARM43_V1_P3_2026032618/HA43_N55.nc"

# Domain centre (TU Delft campus)
CENTRE_LAT = 51.9990
CENTRE_LON = 4.3731

# Domain origin in EPSG:28992 (meters)
RD_X0 = 85346  # up to 85663
RD_Y0 = 445928  # up to 446374

# EPSG:28992 proj4 string
PROJ4_RD = (
    "+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 "
    "+k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel "
    "+towgs84=565.4171,50.3319,465.5524,-0.398957,0.343988,-1.87740,4.0725 "
    "+units=m +no_defs"
)

# Domain size
ITOT, JTOT = 512, 512
DX, DY = 1.0, 1.0
# DX, DY = 2.0, 2.0
XSIZE = ITOT * DX
YSIZE = JTOT * DY

# Vertical levels
KMAX = 60
DZ0 = 2.0

# Time settings
START = "2026-03-26T18:00:00"
END = "2026-03-27T00:00:00"
RUNTIME = 21600  # seconds (6 hours)
START_YEAR, START_MONTH, START_DAY = 2026, 3, 26
START_HOUR = 18.0

# Surface defaults
THLS = 290.0
PS = 101325.0

# ============================================================================

if __name__ == "__main__":
    with open("machine_conf.yaml", "r", encoding="utf-8") as f:
        machine_conf = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("DALES forced by KNMI HARMONIE (HarmonieAtmosphereModule)")
    logger.info("=" * 70)

    case_name = "001_tudelft_campus"

    grid = GridDales(
        itot=ITOT,
        jtot=JTOT,
        kmax=KMAX,
        xsize=XSIZE,
        ysize=YSIZE,
        kmax_soil=4,
        xlat=CENTRE_LAT,
        xlon=CENTRE_LON,
        x0=RD_X0,
        y0=RD_Y0,
        alpha=1.025,
        dz0=DZ0,
        proj4=PROJ4_RD,
    )

    sim = dales_simulation(case_name=case_name, machine_conf=machine_conf)

    logger.info("Created simulation: %s", case_name)

    sim += DefaultNamelistModule()
    sim += grid

    # Add Harmonie forcing module before AtmosphereModule so profile
    # injection happens before AtmosphereModule.prepare_calculation().
    sim += HarmonieAtmosphereModule(
        harmonie_ml_glob=KNMI_ML_GLOB,
        harmonie_sfc_glob=KNMI_SFC_GLOB,
        harmonie_start=START,
        harmonie_end=END,
        harmonie_time0=START,
        harmonie_tchunk=1,
        start_date=datetime.fromisoformat(START),
        end_date=datetime.fromisoformat(END),
        central_lat=CENTRE_LAT,
        central_lon=CENTRE_LON,
        do_nudging=True,
        write_nudging_netcdf=True,
        use_wind_as_geostrophic=True,
    )

    # Base atmosphere module receives LS2D-like profiles and timed forcings
    # injected by HarmonieAtmosphereModule.
    atmo = AtmosphereModule()
    # atmo += InterpolatedProfile(variable=ua, z=[0, 4000, 5000], points=[0, 1e-8, 1e-8])
    # atmo += InterpolatedProfile(variable=va, z=[0, 4000, 5000], points=[0, 1e-8, 1e-8])
    # atmo += InterpolatedProfile(variable=qt, z=[0, 4000, 5000], points=[0, 1e-8, 1e-8])
    # atmo += InterpolatedProfile(variable=thl, z=[0, 4000, 5000], points=[0, 1e-8, 1e-8])
    sim += atmo
    sim += TimeModule(
        xday=START_DAY,
        xtime=START_HOUR,
        xyear=START_YEAR,
        runtime=RUNTIME,
        startyear=START_YEAR,
        startmonth=START_MONTH,
        startday=START_DAY,
    )

    sim += ConstantSurfaceTemperatureModule(
        thls=THLS,
        z0mav=0.01,  # aerodynamic roughness length for momentum
        z0hav=0.01,  # aerodynamic roughness length for heat and moisture
        ps=PS,  # surface pressure
    )

    IBM = IBMModule()  # use the IBM
    IBM += FromAHN()  # download height map from Algemene Hoogtekaart Nederland
    sim += IBM

    time = TimedependentModule(ltimedep=True)
    time += FromLS2D()  # Enable LS2D-driven time series injection into atmosphere
    sim += time

    # sim += EasyOutputModule(output_interval=1, enable_output=True)
    # sim += RadfieldModule(enabled=True, dtav=5, timeav=5)
    # sim += TimestatModule(enabled=True, dtav=5)
    # sim += FielddumpModule(lfielddump=True, dtav=5)
    # sim += CrossSectionOutputModule(
    #     cross_enabled=True,
    #     cross_dtav=1,
    #     xy_enabled=True,
    #     xz_enabled=True,
    #     yz_enabled=False,
    #     xy=[1, 2, 3, 4],
    # )

    # set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocx", 0)
    # set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocy", 0)
    # set_nml_section(
    #     sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
    # )
    # set_nml_section(
    #     sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "lanisotrop", True
    # )  # anisotropic SGS speeds up the simulation a lot, and we don't really care about the small inconsistencies, so we set to True

    logger.info("Running preprocessing pipeline...")
    sim.sim_preprocessing_pipeline()
    logger.info("Done! Case written to: %s", sim.output_path)
