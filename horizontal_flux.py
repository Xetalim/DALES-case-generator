"""Build horizontal-flux experiments for low/high-resolution land-use tiling.

Usage:
    python horizontal_flux.py --start-day <day>
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
import subprocess

import numpy as np
import yaml

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    DefaultNamelistModule,
    GridDales,
    RadiationModule,
    TimeModule,
    dales_simulation,
)
from modular_dales.Atmosphere.ls2d_atmosphere import FromLS2D, LS2DAtmosphereModule
from modular_dales.Configuration.output_modules import (
    CapeModule,
    CrossSectionOutputModule,
    FielddumpModule,
    LSMCrossModule,
    SamplingModule,
    StatsModule,
    RadfieldModule,
    TimestatModule,
    ColumnStatisticsOutputModule,
    VirtualMeasurementOutputModule,
)
from modular_dales.Configuration import NetCDFStatisticsSyncModule
from modular_dales.Geometry.geometry_modification import (
    AllGeometry,
    CheckerboardIdxGeometry,
)
from modular_dales.Surface.LSM.LSM import (
    LSMModule,
    LandUseModification,
)
from modular_dales.Surface.LSM.SLuRB.slurb import (
    SLURBModule,
)
from modular_dales.modular.simulation_module import set_nml_section
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.logging_wrapper import setup_logging
from modular_dales.modular.time_dependent import TimedependentModule
from modular_dales.vars import tke

setup_logging("logging.yaml")
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
x0, y0 = 136173, 455912  # center of the domain in RD coordinates
DX_HIGH = 50
ITOT_HIGH = 128
JTOT_HIGH = 128

_GRID_HIGH = dict(
    itot=ITOT_HIGH,
    jtot=JTOT_HIGH,
    kmax=72,
    xsize=ITOT_HIGH * DX_HIGH,
    ysize=JTOT_HIGH * DX_HIGH,
    kmax_soil=4,
    xlat=52.25,
    xlon=5.45,
    x0=x0 - ((ITOT_HIGH * DX_HIGH) / 2.0),
    y0=y0 - ((JTOT_HIGH * DX_HIGH) / 2.0),
    proj4="EPSG:28992",
    alpha=1.02,
    dz0=20.0,
)

_GRID_LOW = dict(
    itot=ITOT_HIGH // 2,
    jtot=JTOT_HIGH // 2,
    kmax=72,
    xsize=_GRID_HIGH["xsize"],
    ysize=_GRID_HIGH["ysize"],
    kmax_soil=4,
    xlat=52.25,
    xlon=5.45,
    x0=_GRID_HIGH["x0"],
    y0=_GRID_HIGH["y0"],
    proj4="EPSG:28992",
    alpha=1.02,
    dz0=20.0,
)


def _may_start_end(start_day: int) -> tuple[datetime.datetime, datetime.datetime]:
    start = datetime.datetime(2026, 5, start_day, hour=12)
    end = start + datetime.timedelta(hours=1)
    return start, end


def _new_grid(spec: dict) -> GridDales:
    grid = GridDales(**spec)
    zt, zm, dz = make_integer_stretched_grid(grid.kmax, grid.dz0, grid.alpha)
    grid.zt = zt
    grid.zm = zm
    grid.dz = dz
    grid.zsize = float(grid.zm[-1])
    return grid


def make_integer_stretched_grid(kmax, dz0, alpha):
    """
    Construct a DALES-consistent integer stretched grid.

    zt (zf) is integer-valued.
    zm (zh) is derived from zt using the DALES recurrence.
    """

    # Desired cell-center positions
    zt_target = dz0 * (alpha ** np.arange(kmax) - 1) / (
        alpha - 1
    ) + 0.5 * dz0 * alpha ** np.arange(kmax)

    # Integer cell-center heights
    zt = np.rint(zt_target).astype(int)

    # DALES interfaces
    zm = np.zeros(kmax + 1, dtype=int)

    for k in range(kmax):
        zm[k + 1] = 2 * zt[k] - zm[k]

    # Cell thicknesses
    dz = np.diff(zm).astype(int)

    return zt, zm, dz


def _attach_common_physics(
    sim: dales_simulation,
    grid: GridDales,
    start_day: int,
    landuse_experiment: str,
    center_vertical_crosssections: bool = False,
    use_ls2d: bool = False,
    is_nested: bool = False,
) -> None:
    start_ts, _ = _may_start_end(start_day)
    if use_ls2d:
        time = TimedependentModule(ltimedep=True, usesLS2DforTime=True)
        time += FromLS2D()

        sim += time
    if use_ls2d:
        atmo_ls2d = LS2DAtmosphereModule(
            era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
                "era5_path",
                "/tmp/era5_data",
            ),
            start_date=start_ts.replace(hour=0),
            end_date=start_ts.replace(hour=23) + datetime.timedelta(days=1),
            write_log=False,
            data_source=sim.machine_conf.get("ls2d_conf", {}).get("data_source", "CDS"),
            n_av=0,
            method="2nd",
            do_nudging=True,
            smooth_initial_uv_to_geostrophic=True,
        )
        sim += atmo_ls2d
    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(
        variable=tke, shape="lin", params=dict(surf_val=1, ddz=0)
    )
    sim += atmo

    sim += TimeModule(
        xtime=12.0,
        xyear=2026,
        runtime=3600 * 24,
        startyear=2026,
        startmonth=5,
        startday=start_day,
        inferfromdatetime=True,
    )

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

    if landuse_experiment == "lowres_fractional":
        lsm += LandUseModification(
            geometry=AllGeometry(), type="grs", frac=0.5, mode="replace"
        )
        lsm += LandUseModification(
            geometry=AllGeometry(), type="slb", frac=0.5, mode="add"
        )
    elif landuse_experiment == "highres_checkerboard":
        lsm += LandUseModification(
            geometry=AllGeometry(), type="grs", frac=1.0, mode="replace"
        )
        lsm += LandUseModification(
            geometry=CheckerboardIdxGeometry(block=5),
            type="slb",
            frac=1.0,
            mode="replace",
        )
    else:
        raise ValueError(f"Unknown landuse_experiment '{landuse_experiment}'")

    if use_ls2d:
        lsm += FromLS2D()

    sim += lsm

    slurb = SLURBModule(
        deep_soil_temperature=273.15 + 15, building_indoor_temperature=273.15 + 21
    )
    sim += slurb

    sim += RadiationModule(iradiation=4)

    sim += LSMCrossModule(enabled=True, dtav=120)
    sim += CapeModule(enabled=True, dtav=120)
    sim += StatsModule(enabled=True, dtav=60, timeav=60)
    sim += FielddumpModule(dtav=600, lfielddump=True, le12=False)
    sim += RadfieldModule(enabled=True, dtav=120, timeav=120)
    sim += TimestatModule(enabled=True, dtav=60)
    sim += VirtualMeasurementOutputModule(x=x0, y=y0, enabled=True)
    sim += ColumnStatisticsOutputModule(x=x0, y=y0, enabled=True)
    sim += NetCDFStatisticsSyncModule()

    xz_coords = []
    yz_coords = []
    if center_vertical_crosssections:
        xz_coords = [float(grid.y0 + 0.5 * grid.ysize)]
        yz_coords = [float(grid.x0 + 0.5 * grid.xsize)]

    sim += CrossSectionOutputModule(
        cross_enabled=True,
        cross_dtav=10,
        xy_coords=[float(grid.zt[0])],
        xz_coords=xz_coords,
        yz_coords=yz_coords,
        xy_enabled=True,
        xz_enabled=True,
        yz_enabled=True,
    )

    sim += SamplingModule(output_interval=120)

    if sim.nml.get("namchecksim") is None:
        sim.nml["namchecksim"] = {}
    sim.nml["namchecksim"]["tcheck"] = 60
    if sim.nml.get("namnetcdfstats") is None:
        sim.nml["namnetcdfstats"] = {}
    sim.nml["namnetcdfstats"]["deflate"] = 0
    if use_ls2d:
        set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "PHYSICS", "ltimedep", True
        )
        set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "NAMNUDGE", "lnudge", True
        )
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "sgs_surface_fix", True
    )
    set_nml_section(
        sim.nml,
        sim.nml_docs,
        "user_defined",
        "namsubgrid",
        "lD80R",
        False,  # hopefully this addresses stable boundary layer issues
    )

    if sim.nml.get("thermodynamics") is None:
        sim.nml["thermodynamics"] = {}
    sim.nml["thermodynamics"]["lbaseexner"] = True


def _build_simulation(
    machine_conf_dict: dict, start_day: int, stage: str
) -> dales_simulation:
    if stage == "high":
        case_name = "horizontal_flux_highres"
        grid_spec = _GRID_HIGH
        landuse_experiment = "highres_checkerboard"
    elif stage == "low":
        case_name = "horizontal_flux_lowres"
        grid_spec = _GRID_LOW
        landuse_experiment = "lowres_fractional"
    else:
        raise ValueError(f"Unknown stage '{stage}'")

    sim = dales_simulation(case_name, machine_conf_dict)
    sim.init_output_folder()
    sim += DefaultNamelistModule()

    grid = _new_grid(grid_spec)
    sim += grid

    _attach_common_physics(
        sim,
        grid,
        start_day,
        landuse_experiment=landuse_experiment,
        center_vertical_crosssections=True,
        use_ls2d=True,
    )
    return sim


def _run_job_direct(case_dir: Path, stage_name: str) -> None:
    logger.info("Running %s in %s", stage_name, case_dir)
    subprocess.run(["./job.001"], cwd=case_dir, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DALES case input.")
    parser.add_argument("--start-day", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=["low", "high", "all"],
        default="all",
        help="Generation stage: low, high, or all.",
    )
    parser.add_argument(
        "--machine-conf",
        default=str(Path(__file__).resolve().parent / "machine_conf.yaml"),
        help="Path to machine configuration YAML.",
    )
    parser.add_argument(
        "--run-jobs",
        action="store_true",
        help="Run generated job.001 files directly. Default is to only generate.",
    )
    return parser.parse_args()


def _write_handoff(case_root: Path, job_file: Path, next_stage: str | None) -> None:
    payload = {
        "job_file": str(job_file),
        "cwd": str(job_file.parent),
        "next_stage": next_stage,
    }
    handoff_path = case_root / "generator_handoff.json"
    handoff_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    import os

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # dask.config.set(scheduler="single-threaded")
    args = _parse_args()

    from dask.distributed import LocalCluster, Client

    n_cores = os.cpu_count() or 2

    cluster = LocalCluster(
        n_workers=0,
        threads_per_worker=1,
        memory_limit="8GB",
        dashboard_address=":8787",
    )

    cluster.adapt(
        minimum=1,
        maximum=max(1, n_cores - 1),
        wait_count=3,
        interval="1s",
    )

    client = Client(cluster)

    print(f"Dask dashboard: {cluster.dashboard_link}")
    machine_conf_path = Path(args.machine_conf)
    with machine_conf_path.open("r", encoding="utf-8") as handle:
        machine_conf = yaml.safe_load(handle)

    logger.info(
        "Building horizontal flux experiment (stage=%s) for day %s",
        args.stage,
        args.start_day,
    )

    if args.stage in ("low", "all"):
        sim_low = _build_simulation(machine_conf, args.start_day, "low")
        sim_low.sim_preprocessing_pipeline()
        if args.run_jobs:
            _run_job_direct(sim_low.output_path, "job_low_resolution")

    if args.stage in ("high", "all"):
        sim_high = _build_simulation(machine_conf, args.start_day, "high")
        sim_high.sim_preprocessing_pipeline()
        if args.run_jobs:
            _run_job_direct(sim_high.output_path, "job_high_resolution")

    logger.info("Completed horizontal flux setup for stage '%s'", args.stage)
