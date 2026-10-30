"""Open boundaries from the locally cached KNMI HARMONIE 2026032618 cycle."""

import shutil
from pathlib import Path

import pytest

from moda import (
    DefaultNamelistModule,
    GridDales,
    TimeModule,
    dales_simulation,
)
from moda.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    HarmonieAtmosphereModule,
    lin,
)
from moda.LBC import KNMIHarmonieForecastDownloadModule, Nest_in_KNMI
from moda.Surface import ConstantSurfaceTemperatureModule
from moda.vars import qt, thetal, tke, ua, va, w

DATA_PATH = Path("HARM_TEST_DATA")


def extract_zip(path):
    archive = Path(__file__).parent / "openbc_test_input" / "HARM_TEST_DATA.zip"
    import zipfile

    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(Path(path))


def openbc_harmonie_cached_case(machine_conf: dict) -> dales_simulation:
    """Use cached P5/P3 archives through the standard download/convert workflow.

    The 18 UTC forecast supplies one hour of boundaries for a 60-second run.
    Missing local archives skip this machine-specific case, never download them.
    """
    extract_zip(machine_conf["case_conf"]["BASE_OUTPUT_PATH"])
    cdo_path = shutil.which("cdo")
    if cdo_path is None:
        pytest.skip("CDO is required for the cached HARMONIE conversion workflow")

    sim = dales_simulation("openbc_harmonie_cached_2026032618", machine_conf)
    sim.add(DefaultNamelistModule())
    sim.add(
        GridDales(
            itot=16,
            jtot=16,
            kmax=20,
            xsize=1600.0,
            ysize=1600.0,
            kmax_soil=4,
            xlat=52.25,
            xlon=5.45,
            x0=136173.0 - 800.0,
            y0=455912.0 - 800.0,
            proj4="EPSG:28992",
            alpha=1.0,
            dz0=20.0,
        )
    )
    atmosphere = AtmosphereModule()
    for variable, value in (
        (ua, 3.0),
        (va, 0.0),
        (w, 0.0),
        (thetal, 293.15),
        (qt, 0.01),
        (tke, 0.1),
    ):
        atmosphere.add(
            AtmosphericProfile(variable=variable, shape=lin(surf_val=value, ddz=0.0))
        )
    sim.add(atmosphere)
    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15, ps=100000.0, z0mav=0.0001, z0hav=0.0001, albedoav=0.22
        )
    )
    sim.add(
        TimeModule(
            xtime=18.0,
            xyear=2026,
            runtime=60,
            startyear=2026,
            startmonth=3,
            startday=26,
            inferfromdatetime=True,
        )
    )
    sim.add(
        KNMIHarmonieForecastDownloadModule(
            forecast_datetime="2026032618",
            download_root=str(
                Path(machine_conf["case_conf"]["BASE_OUTPUT_PATH"]) / "HARM_TEST_DATA"
            ),
            cdo_path=cdo_path,
            delete_grib_files=False,
            convert_to_netcdf=False,
            check_disk_space=False,
        )
    )
    boundaries = Nest_in_KNMI(
        time0="2026-03-26T18:00:00",
        start="2026-03-26T18:00:00",
        end="2026-03-26T19:00:00",
        e12=0.1,
        tchunk=1,
    )
    sim.add(boundaries)
    return sim


def openbc_harmonie_cached_case_timedep(machine_conf: dict) -> dales_simulation:
    """Use cached P5/P3 archives through the standard download/convert workflow.

    The 18 UTC forecast supplies one hour of boundaries for a 60-second run.
    Missing local archives skip this machine-specific case, never download them.
    """
    extract_zip(machine_conf["case_conf"]["BASE_OUTPUT_PATH"])

    cdo_path = shutil.which("cdo")
    if cdo_path is None:
        pytest.skip("CDO is required for the cached HARMONIE conversion workflow")

    sim = dales_simulation("openbc_harmonie_cached_2026032618_timedep", machine_conf)
    sim.add(DefaultNamelistModule())
    sim.add(
        GridDales(
            itot=16,
            jtot=16,
            kmax=20,
            xsize=1600.0,
            ysize=1600.0,
            kmax_soil=4,
            xlat=52.25,
            xlon=5.45,
            x0=136173.0 - 800.0,
            y0=455912.0 - 800.0,
            proj4="EPSG:28992",
            alpha=1.0,
            dz0=20.0,
        )
    )

    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15, ps=100000.0, z0mav=0.0001, z0hav=0.0001, albedoav=0.22
        )
    )

    sim.add(
        TimeModule(
            xtime=18.0,
            xyear=2026,
            runtime=60,
            startyear=2026,
            startmonth=3,
            startday=26,
            inferfromdatetime=True,
        )
    )
    atmo = AtmosphereModule()
    sim.add(
        HarmonieAtmosphereModule(
            harmonie_ml_glob=str(
                Path(machine_conf["case_conf"]["BASE_OUTPUT_PATH"])
                / "HARM_TEST_DATA"
                / "HARM43_V1_P5_*/*"
            ),
            harmonie_sfc_glob=str(
                Path(machine_conf["case_conf"]["BASE_OUTPUT_PATH"])
                / "HARM_TEST_DATA"
                / "HARM43_V1_P3_*/*"
            ),
            harmonie_start="2026-03-26T18:00:00",
            harmonie_end="2026-03-26T19:00:00",
            harmonie_time0="2026-03-26T18:00:00",
            harmonie_tchunk=1,
            do_nudging=True,
            smooth_initial_uv_to_geostrophic=False,
            infer_adv_tendencies=True,
        )
    )
    atmo.add(AtmosphericProfile(variable=tke, shape=lin(surf_val=1, ddz=0)))
    sim.add(atmo)
    return sim
