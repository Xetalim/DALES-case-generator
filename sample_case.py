"""Generate the LS2D-driven outer case and its nested DALES child.

Usage:
    python case_to_generate.py --start-day 1 --stage l1
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import moda
import moda.Atmosphere.Profile
from moda import LBC, Atmosphere, Configuration, Geometry
from moda.Surface import LSM

moda.setup_logging("logging.yaml")
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
x0, y0 = 136173, 455912  # center of the domain in RD coordinates
_GRID64_SUPER = {
    "itot": 64,
    "jtot": 64,
    "kmax": 96,
    "xsize": 6400.0 * 1.5,
    "ysize": 6400.0 * 1.5,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": x0 - ((6400.0 * 1.5) / 2.0),
    "y0": y0 - ((6400.0 * 1.5) / 2.0),
    "proj4": "EPSG:28992",
    "alpha": 1.02,
    "dz0": 20.0,
}

_GRID64_MID = {
    "itot": 64,
    "jtot": 64,
    "kmax": 48,
    "xsize": 1600.0 * 1.5,
    "ysize": 1600.0 * 1.5,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": x0 - ((6400.0 * 1.5) / 2.0) + (2400.0 * 1.5),
    "y0": y0 - ((6400.0 * 1.5) / 2.0) + (2400.0 * 1.5),
    "proj4": "EPSG:28992",
    "alpha": 1,
    "dz0": 10.0,
}


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _extract_nml_overrides(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # Accept either {nml_overrides: {...}} or direct {SECTION: {...}} mappings.
    candidate = raw.get("nml_overrides")
    if not isinstance(candidate, dict):
        candidate = raw

    out: dict[str, dict[str, Any]] = {}
    for section, values in candidate.items():
        if not isinstance(values, dict):
            continue
        out[str(section)] = {str(key): value for key, value in values.items()}
    return out


def _load_nml_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    nml_overrides: dict[str, dict[str, Any]] = {}

    if args.params_file:
        params_file = Path(args.params_file)
        raw = yaml.safe_load(params_file.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("--params-file must contain a YAML/JSON mapping")
        _deep_update(nml_overrides, _extract_nml_overrides(raw))

    if args.params_json:
        raw_json = json.loads(args.params_json)
        if not isinstance(raw_json, dict):
            raise ValueError("--params-json must decode to a JSON object")
        _deep_update(nml_overrides, _extract_nml_overrides(raw_json))

    return nml_overrides


def _may_start_end(start_day: int) -> tuple[datetime.datetime, datetime.datetime]:
    start = datetime.datetime(2026, 5, start_day, hour=12, tzinfo=datetime.UTC)
    end = start + datetime.timedelta(hours=24)
    return start, end


def _new_grid(spec: dict) -> moda.GridDales:
    grid = moda.GridDales(**spec)
    zt, zm, dz = make_integer_stretched_grid(grid.kmax, grid.dz0, grid.alpha)
    grid.zt = zt
    grid.zm = zm
    grid.dz = dz
    grid.zsize = float(grid.zm[-1])
    grid.input_dic = grid.as_dic()
    return grid


def make_integer_stretched_grid(kmax, dz0, alpha):
    """
    Construct a DALES-consistent integer stretched grid.

    zt (zf) is integer-valued.
    zm (zh) is derived from zt using the DALES recurrence.
    """

    # Desired cell-center positions
    if alpha == 1.0:
        zt_target = dz0 * (np.arange(kmax) + 0.5)
    else:
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


def _copy_vertical_grid(
    source_grid: moda.GridDales, target_grid: moda.GridDales
) -> None:
    """Copy exact vertical coordinates from source grid to target grid."""
    if len(source_grid.zt) != target_grid.kmax:
        raise ValueError(
            "Vertical copy requires equal kmax: "
            f"source len(zt)={len(source_grid.zt)} target kmax={target_grid.kmax}"
        )
    if len(source_grid.zm) != target_grid.kmax + 1:
        raise ValueError(
            "Vertical copy requires equal zm length: "
            f"source len(zm)={len(source_grid.zm)} target expected={target_grid.kmax + 1}"
        )

    target_grid.zt = source_grid.zt.copy()
    target_grid.zm = source_grid.zm.copy()
    target_grid.dz = np.diff(target_grid.zm).astype(float)
    target_grid.zsize = float(target_grid.zm[-1])


def _set_child_vertical_grid(
    parent_grid: moda.GridDales, child_grid: moda.GridDales
) -> None:
    child_grid.inherit_vertical_grid(parent_grid)


def _run_job_direct(case_dir: Path, stage_name: str) -> None:
    logger.info("Running %s in %s", stage_name, case_dir)
    subprocess.run(["./job.001"], cwd=case_dir, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DALES case input.")
    parser.add_argument("--start-day", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=["l1", "l2", "all"],
        default="all",
        help="Generation stage: l1 (outer parent), l2 (nested parent), or all.",
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
    parser.add_argument(
        "--params-json",
        default=None,
        help="JSON object with optional namelist overrides.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Path to YAML/JSON file with optional namelist overrides.",
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

    from dask.distributed import Client, LocalCluster

    n_cores = os.cpu_count() or 2
    max_workers = int(
        os.environ.get("DALES_GENERATOR_MAX_WORKERS", str(max(1, n_cores - 1)))
    )
    worker_memory = os.environ.get("DALES_GENERATOR_WORKER_MEMORY", "8GB")

    cluster = LocalCluster(
        n_workers=0,
        threads_per_worker=1,
        memory_limit=worker_memory,
        dashboard_address=":8787",
    )

    cluster.adapt(
        minimum=1,
        maximum=max(1, max_workers),
        wait_count=3,
        interval="1s",
    )

    client = Client(cluster)

    print(f"Dask dashboard: {cluster.dashboard_link}")
    nml_overrides = _load_nml_overrides(args)
    machine_conf_path = Path(args.machine_conf)
    with machine_conf_path.open("r", encoding="utf-8") as handle:
        machine_conf = yaml.safe_load(handle)

    logger.info("Building scaled64 triple-nesting case for day %s", args.start_day)
    case_root = Path(machine_conf["case_conf"]["BASE_OUTPUT_PATH"])
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "requested_nml_overrides.yaml").write_text(
        yaml.safe_dump(nml_overrides, sort_keys=False), encoding="utf-8"
    )

    sim = moda.dales_simulation("openbc_triple64_parent_l1", machine_conf)
    sim.init_output_folder()
    sim.add(moda.DefaultNamelistModule())

    supergrid = _new_grid(_GRID64_SUPER)
    midgrid = _new_grid(_GRID64_MID)
    _set_child_vertical_grid(supergrid, midgrid)

    sim.add(supergrid)

    nesting = LBC.NestingTopology()
    nesting.add(supergrid)
    nesting.add(midgrid)
    nesting.my_idx = nesting.nestings.index(supergrid)
    sim.add(nesting)

    start_ts, end_ts = _may_start_end(start_day)
    spatial_data_path = sim.machine_conf.get("lsm_conf", {}).get(
        "spatial_data_path", "/tmp/spatial_data"
    )

    if use_ls2d:
        atmo_ls2d = Atmosphere.LS2DAtmosphereModule(
            era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
                "era5_path",
                "/tmp/era5_data",
            ),
            case_name=sim.case_name,
            start_date=start_ts,
            end_date=end_ts,
            write_log=False,
            data_source="CDS",
            n_av=0,
            method="2nd",
            do_nudging=True,
            smooth_initial_uv_to_geostrophic=True,
        )
        sim.add(atmo_ls2d)
    atmo = moda.AtmosphereModule()
    atmo.add(
        moda.Atmosphere.Profile.AtmosphericProfile(
            variable=moda.vars.tke, shape=Atmosphere.lin(surf_val=1, ddz=0)
        )
    )
    sim.add(atmo)

    sim.add(
        moda.TimeModule(
            xtime=12.0,
            xyear=2026,
            runtime=(end_ts - start_ts).total_seconds(),
            startyear=2026,
            startmonth=5,
            startday=start_day,
            inferfromdatetime=True,
        )
    )

    lsm = LSM.LSMModule(
        include_slurb=True,
        deep_soil_temperature=273.15 + 15,
        building_indoor_temperature=273.15 + 21,
        land_use_set=LSM.LandUseSet.IFS,
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
        lsm.add(
            [
                LSM.lsm_sources.UniformSkinTemperature(294),
                LSM.lsm_sources.UniformSoilTemperature([294, 294, 294, 294]),
                LSM.lsm_sources.UniformSoilMoisture(
                    [
                        0.36867549,
                        0.25300502,
                        0.14997292,
                        0.16459982,
                    ]
                ),
            ],
            priority=600,
        )
    lsm.add(
        LSM.lsm_sources.LandUseModification(mask=Geometry.AllMask(), type="grs"),
        priority=0,
    )
    lsm.add(
        LSM.lsm_sources.FromLCZ(urban_natural_lcz_to_natural_lsm=True), priority=100
    )
    if use_ls2d:
        lsm.add(LSM.lsm_sources.FromLS2D(), priority=500)
    lsm.add(
        LSM.lsm_sources.FromBofek(spatial_data_path=spatial_data_path), priority=400
    )
    lsm.add(
        LSM.lsm_sources.FromTop10(
            urban_natural_lcz_to_natural_lsm=True,
            spatial_data_path=spatial_data_path,
        ),
        priority=300,
    )
    sim.add(lsm)

    sim.add(moda.Radiation.RRTMGRadiationModule())

    sim.add(Configuration.LSMCrossModule(enabled=True, dtav=60))
    sim.add(Configuration.CapeModule(enabled=True, dtav=120))
    sim.add(Configuration.StatsModule(enabled=True, dtav=5, timeav=5))
    sim.add(Configuration.FielddumpModule(dtav=120, lfielddump=True, le12=False))
    sim.add(Configuration.RadfieldModule(enabled=True, dtav=120, timeav=120))
    sim.add(Configuration.TimestatModule(enabled=True, dtav=60))
    x_meas = [x0]
    y_meas = [y0]
    x_meas.append(float(grid.xt[32 - 1]))
    y_meas.append(float(grid.yt[11 - 1]))
    sim.add(
        Configuration.VirtualMeasurementOutputModule(x=x_meas, y=y_meas, enabled=True)
    )
    sim.add(
        Configuration.ColumnStatisticsOutputModule(x=x_meas, y=y_meas, enabled=True)
    )
    sim.add(Configuration.NetCDFStatisticsSyncModule())

    xz_coords = [float(grid.y0 + 0.5 * grid.ysize)]
    yz_coords = [float(grid.x0 + 0.5 * grid.xsize)]

    sim.add(
        Configuration.CrossSectionOutputModule(
            cross_enabled=True,
            cross_dtav=10,
            xy_coords=[float(grid.zt[0])],
            xz_coords=xz_coords,
            yz_coords=yz_coords,
            xy_enabled=True,
            xz_enabled=True,
            yz_enabled=True,
        )
    )

    sim.add(Configuration.SamplingModule(output_interval=120))
    sim.add(
        Configuration.CheckSimulationModule(
            check_interval=120, stop_on_invalid=False, check_tendencies=True
        )
    )
    # if sim.nml.get("namchecksim") is None:
    #     sim.nml["namchecksim"] = {}
    # sim.nml["namchecksim"]["tcheck"] = 60
    if sim.nml.get("namnetcdfstats") is None:
        sim.nml["namnetcdfstats"] = {}
    sim.nml["namnetcdfstats"]["deflate"] = 0
    if use_ls2d:
        moda.set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "PHYSICS", "ltimedep", True
        )
        moda.set_nml_section(
            sim.nml, sim.nml_docs, "user_defined", "NAMNUDGE", "lnudge", True
        )
    moda.set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namsubgrid", "sgs_surface_fix", False
    )
    moda.set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "timer", "ltimer", True
    )  #     namelist /TIMER/ ltimer, ltimer_print, ltimer_write
    moda.set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "timer", "ltimer_write", True
    )  #     namelist /TIMER/ ltimer, ltimer_print, ltimer_write
    moda.set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lnofill", True
    )
    moda.set_nml_section(
        sim.nml,
        sim.nml_docs,
        "user_defined",
        "namsubgrid",
        "lD80R",
        False,
    )

    if isinstance(nml_overrides, dict):
        for section, values in nml_overrides.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                moda.set_nml_section(
                    sim.nml,
                    sim.nml_docs,
                    "user_defined",
                    str(section),
                    str(key),
                    value,
                )
    # set_nml_section(
    #     sim.nml, sim.nml_docs, "user_defined", "run", "peclet", 0.1 # hopefully this addresses stable boundary layer issues
    # )
    # set_nml_section(
    #     sim.nml, sim.nml_docs, "user_defined", "run", "courant", 0.5 # hopefully this addresses stable boundary layer issues
    # )

    if sim.nml.get("thermodynamics") is None:
        sim.nml["thermodynamics"] = {}
    sim.nml["thermodynamics"]["lbaseexner"] = True

    sim.sim_preprocessing_pipeline()
