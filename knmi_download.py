from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
from pathlib import Path
import subprocess

import pytest
import yaml

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    ConstantSurfaceTemperatureModule,
    DefaultNamelistModule,
    GridDales,
    InterpolatedProfile,
    KNMIHarmonieForecastDownloadModule,
    TimeModule,
    dales_simulation,
    do_openboundary,
    LSMModule,
)
from modular_dales.Atmosphere.ls2d_atmosphere import FromLS2D
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.LBC.openbc import Nest_in_KNMI
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.LSM import (
    FromBofek,
    FromLCZ,
    FromTop10,
    LandUseModification,
)
from modular_dales.Surface.LSM.SLuRB.slurb import SLURBModule
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.logging_wrapper import setup_logging
from modular_dales.vars import qt, tke, thetal, ua, va, wa

setup_logging("logging.yaml")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _has_knmi_credentials(api_key_file: str) -> bool:
    requested = Path(api_key_file).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates = [Path.cwd() / requested, requested, Path.home() / requested]

    return any(path.exists() and path.stat().st_size > 0 for path in candidates)


def _previous_day_forecast_datetime() -> dt.datetime:
    """Return previous-day 00 UTC forecast timestamp.

    Using the previous day avoids depending on the newest cycle availability
    during long CI or local runs.
    """

    now_utc = dt.datetime.now(dt.timezone.utc)
    prev_day = (now_utc - dt.timedelta(days=1)).date()
    return dt.datetime(prev_day.year, prev_day.month, prev_day.day, 0, 0, 0)


"""Build a full simulation that triggers real KNMI download in preprocessing."""
if __name__ == "__main__":
    with open("machine_conf.yaml", "r", encoding="utf-8") as f:
        machine_conf = yaml.safe_load(f)
    os.environ["ECCODES_DEFINITION_PATH"] = (
        "/Users/andrevanginkel/Documents/20_Code/24_dales_source/Harmonie/util/gl/definitions"
    )
    forecast_dt = _previous_day_forecast_datetime()
    forecast_dt = forecast_dt.replace(
        year=2026, month=3, day=26, hour=18, minute=0, second=0, microsecond=0
    )
    forecast_datetime = forecast_dt.strftime("%Y%m%d%H")
    time0_dt = forecast_dt
    start_dt = time0_dt
    end_default = time0_dt + dt.timedelta(hours=12)
    end_dt = end_default

    api_key_file = ".knmiapirc"
    if not _has_knmi_credentials(api_key_file):
        raise ValueError("No KNMI credentials found in .knmiapirc (cwd or home).")

    cdo_path = shutil.which("cdo")
    if cdo_path is None:
        raise ValueError(
            "CDO executable not found in PATH; required for real GRIB->NetCDF conversion."
        )

    sim = dales_simulation("knmi_download_workflow_real", machine_conf)
    sim += DefaultNamelistModule()
    sim += GridDales(
        itot=128,
        jtot=128,
        kmax=96,
        xsize=128 * 200.0,
        ysize=128 * 200.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=136776.567065 - 0.5 * 128 * 200.0,
        y0=455761.357756 - 0.5 * 128 * 200.0,
        alpha=1.02,
        dz0=20.0,
        wkt="""
PROJCS["Amersfoort / RD New",
    GEOGCS["Amersfoort",
        DATUM["Amersfoort",
            SPHEROID["Bessel 1841",6377397.155,299.1528128],
            TOWGS84[565.4171,50.3319,465.5524,1.9342,-1.6677,9.1019,4.0725]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4289"]],
    PROJECTION["Oblique_Stereographic"],
    PARAMETER["latitude_of_origin",52.1561605555556],
    PARAMETER["central_meridian",5.38763888888889],
    PARAMETER["scale_factor",0.9999079],
    PARAMETER["false_easting",155000],
    PARAMETER["false_northing",463000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","28992"]]""",
    )

    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(
        variable=ua, shape="lin", params=dict(surf_val=3.0, ddz=1e-3)
    )
    atmo += AtmosphericProfile(
        variable=va, shape="lin", params=dict(surf_val=0.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=wa, shape="lin", params=dict(surf_val=0.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=thetal, shape="lin", params=dict(surf_val=293.15, ddz=1e-2)
    )
    atmo += AtmosphericProfile(
        variable=qt, shape="lin", params=dict(surf_val=0.01, ddz=0.0)
    )
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
        tskin_laqu=273.15 + 15,
    )
    is_nested = False
    if not is_nested:
        lsm += UniformSkinTemperature(294)
        lsm += UniformSoilTemperature([294, 294, 294, 294])
        lsm += UniformSoilMoisture(
            [
                0.36867549,
                0.25300502,
                0.14997292,
                0.16459982,
            ]
        )
    lsm += LandUseModification(geometry=AllGeometry(), type="grs")
    lsm += FromLCZ(urban_natural_lcz_to_natural_lsm=True)
    # if use_ls2d:
    #     lsm += FromLS2D()
    lsm += FromBofek(spatial_data_path="/Users/andrevanginkel/Downloads/spatial_data")
    lsm += FromTop10(
        urban_natural_lcz_to_natural_lsm=True,
        spatial_data_path="/Users/andrevanginkel/Downloads/spatial_data",
    )
    sim += lsm

    slurb = SLURBModule(
        deep_soil_temperature=273.15 + 15, building_indoor_temperature=273.15 + 21
    )
    sim += slurb

    sim += RadiationModule(iradiation=4)
    sim += TimeModule(
        xtime=float(time0_dt.hour),
        xday=int(time0_dt.day),
        xyear=int(time0_dt.year),
        runtime=3600 * 12,
        startyear=int(time0_dt.year),
        startmonth=int(time0_dt.month),
        startday=int(time0_dt.day),
    )

    cache_base = (
        Path(machine_conf["case_conf"]["BASE_OUTPUT_PATH"]) / "knmi_download_cache"
    )
    sim += KNMIHarmonieForecastDownloadModule(
        forecast_datetime=forecast_datetime,
        download_root=cache_base.as_posix(),
        api_key_file=api_key_file,
        cdo_path=cdo_path,
        check_disk_space=True,
        minimum_free_space_gb=2.0,
        disk_space_safety_factor=2.0,
        delete_grib_files=True,
        skip_cdo=True,
        convert_to_netcdf=False,
    )

    openbc = do_openboundary(
        time0=time0_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        e12=0.1,
        tchunk=1,
    )
    openbc += Nest_in_KNMI()
    sim += openbc

    sim.sim_preprocessing_pipeline()
    # wd = os.getcwd()
    # os.chdir(sim.output_path.as_posix())
    # subprocess.run("./job.001", check=True)
    # os.chdir(wd)
