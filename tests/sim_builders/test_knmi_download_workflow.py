from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pytest

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
)
from modular_dales.LBC.openbc import Nest_in_KNMI
from modular_dales.vars import qt, tke, thetal, ua, va, wa

from tests.generic_testers import run_simulation_and_check_job


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


def knmi_download_workflow_case(machine_conf: dict) -> dales_simulation:
    """Build a full simulation that triggers real KNMI download in preprocessing."""
    forecast_dt = _previous_day_forecast_datetime()
    forecast_datetime = forecast_dt.strftime("%Y%m%d%H")
    time0_dt = forecast_dt
    start_dt = time0_dt
    end_default = time0_dt + dt.timedelta(hours=1)
    end_dt = end_default
    pytest.skip("Too slow")
    api_key_file = ".knmiapirc"
    if not _has_knmi_credentials(api_key_file):
        pytest.skip("No KNMI credentials found in .knmiapirc (cwd or home).")

    cdo_path = shutil.which("cdo")
    if cdo_path is None:
        pytest.skip(
            "CDO executable not found in PATH; required for real GRIB->NetCDF conversion."
        )

    sim = dales_simulation("knmi_download_workflow_real", machine_conf)
    sim += DefaultNamelistModule()
    sim += GridDales(
        itot=16,
        jtot=16,
        kmax=20,
        xsize=1600.0,
        ysize=1600.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0.0,
        y0=0.0,
        alpha=1.0,
        dz0=20.0,
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

    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=0.0001,
        z0hav=0.0001,
        ps=100000,
        albedoav=0.22,
    )
    sim += TimeModule(
        xtime=float(time0_dt.hour),
        xday=int(time0_dt.day),
        xyear=int(time0_dt.year),
        runtime=60,
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
    return sim


@pytest.mark.slow
@pytest.mark.serial
def test_knmi_download_workflow_simulation_runs(
    machine_conf,
    simulation_report,
) -> None:
    """Run a full simulation that performs real KNMI download+convert preprocessing."""
    pytest.skip(reason="Skipping slow test for now")
    run_simulation_and_check_job(
        knmi_download_workflow_case,
        machine_conf,
        add_report=simulation_report,
    )
