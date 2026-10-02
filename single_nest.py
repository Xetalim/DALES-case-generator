"""Run the scaled64 triple-nesting DALES case directly from the project root.

Usage:
    python triple_nest.py
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import yaml

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    CrossSectionOutputModule,
    DefaultNamelistModule,
    FielddumpModule,
    GridDales,
    InterpolatedProfile,
    LSMCrossModule,
    Nest_in_Dales,
    NestingTopology,
    RadfieldModule,
    RadiationModule,
    StatsModule,
    CapeModule,
    TimeModule,
    dales_simulation,
    do_openboundary,
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
)
from modular_dales.Configuration import NetCDFStatisticsSyncModule
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.LSM import (
    FromBofek,
    FromLCZ,
    FromTop10,
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
from modular_dales.vars import *  # noqa: F401,F403
import subprocess

setup_logging("logging.yaml")
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

_GRID64_SUPER = dict(
    itot=64,
    jtot=64,
    kmax=72,
    xsize=6400.0 * 1.5,
    ysize=6400.0 * 1.5,
    kmax_soil=4,
    xlat=52.25,
    xlon=5.45,
    x0=136173.0 - ((6400.0 * 1.5) / 2.0),
    y0=455912.0 - ((6400.0 * 1.5) / 2.0),
    proj4="EPSG:28992",
    alpha=1.02,
    dz0=20.0,
)

_GRID64_MID = dict(
    itot=64,
    jtot=64,
    kmax=48,
    xsize=1600.0 * 1.5,
    ysize=1600.0 * 1.5,
    kmax_soil=4,
    xlat=52.25,
    xlon=5.45,
    x0=136173.0 - ((6400.0 * 1.5) / 2.0) + (2400.0 * 1.5),
    y0=455912.0 - ((6400.0 * 1.5) / 2.0) + (2400.0 * 1.5),
    proj4="EPSG:28992",
    alpha=1,
    dz0=10.0,
)


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


def _copy_vertical_grid(source_grid: GridDales, target_grid: GridDales) -> None:
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


def _set_child_vertical_grid(parent_grid: GridDales, child_grid: GridDales) -> None:
    if len(parent_grid.zt) < child_grid.kmax or len(parent_grid.zm) < (
        child_grid.kmax + 1
    ):
        raise ValueError(
            "Parent vertical grid is too short to provide a child subset: "
            f"parent(zt={len(parent_grid.zt)}, zm={len(parent_grid.zm)}) "
            f"child(kmax={child_grid.kmax})"
        )

    # DALES convention: len(zt)=kmax and len(zm)=kmax+1.
    child_grid.zt = parent_grid.zt[: child_grid.kmax].copy()
    child_grid.zm = parent_grid.zm[: child_grid.kmax + 1].copy()
    child_grid.dz = np.diff(child_grid.zm).astype(float)
    child_grid.zsize = float(child_grid.zm[-1])

    if not np.array_equal(child_grid.zt, parent_grid.zt[: child_grid.kmax]):
        raise ValueError("Child zt is not an exact prefix subset of parent zt")
    if not np.array_equal(child_grid.zm, parent_grid.zm[: child_grid.kmax + 1]):
        raise ValueError("Child zm is not an exact prefix subset of parent zm")


def _attach_common_physics(
    sim: dales_simulation,
    grid: GridDales,
    center_vertical_crosssections: bool = False,
    use_ls2d: bool = False,
    is_nested: bool = False,
) -> None:
    if use_ls2d:
        sim += TimedependentModule(ltimedep=True)
    if use_ls2d:
        atmo_ls2d = LS2DAtmosphereModule(
            era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
                "era5_path",
                "/Users/andrevanginkel/Documents/20_Code/28_dales_input/28.01_Dales_LSM_generator/jupyter_tests/era5_data",
            ),
            start_date=datetime.datetime(2026, 5, 1, hour=0),
            end_date=datetime.datetime(2026, 5, 31, hour=23),
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
        runtime=3600,
        startyear=2026,
        startmonth=5,
        startday=1,
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
    lsm += LandUseModification(geometry=AllGeometry(), type="grs")
    lsm += FromLCZ(urban_natural_lcz_to_natural_lsm=True)
    if use_ls2d:
        lsm += FromLS2D()
    lsm += FromBofek(spatial_data_path="/Users/andrevanginkel/Downloads/spatial_data")
    lsm += FromTop10(
        urban_natural_lcz_to_natural_lsm=True,
        spatial_data_path="/Users/andrevanginkel/Downloads/spatial_data",
    )
    sim += lsm

    slurb = SLURBModule(deep_soil_temperature=293)
    sim += slurb

    sim += RadiationModule(iradiation=4)

    sim += LSMCrossModule(enabled=True, dtav=120)
    sim += CapeModule(enabled=True, dtav=120)
    sim += StatsModule(enabled=True, dtav=60, timeav=60)
    sim += FielddumpModule(dtav=600, lfielddump=True, le12=False)
    sim += RadfieldModule(enabled=True, dtav=120, timeav=120)

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

    if sim.nml.get("thermodynamics") is None:
        sim.nml["thermodynamics"] = {}
    sim.nml["thermodynamics"]["lbaseexner"] = True


def _build_outer_parent_sim_scaled64(machine_conf: dict) -> dales_simulation:
    sim = dales_simulation("openbc_triple64_parent_l1", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID64_SUPER)
    midgrid = _new_grid(_GRID64_MID)
    _set_child_vertical_grid(supergrid, midgrid)

    sim += supergrid

    nesting = NestingTopology()
    nesting += supergrid
    nesting += midgrid
    nesting.my_idx = nesting.nestings.index(supergrid)
    sim += nesting

    _attach_common_physics(
        sim, supergrid, center_vertical_crosssections=True, use_ls2d=True
    )
    return sim


def _build_middle_nested_sim_scaled64(
    machine_conf: dict, parent_sim: dales_simulation
) -> dales_simulation:
    sim = dales_simulation("openbc_triple64_parent_l2", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID64_SUPER)
    if parent_sim.grid is None:
        raise ValueError(
            "parent_sim.grid is missing; cannot align nested vertical grid"
        )
    _copy_vertical_grid(parent_sim.grid, supergrid)

    midgrid = _new_grid(_GRID64_MID)
    _set_child_vertical_grid(supergrid, midgrid)

    sim += midgrid

    nesting = NestingTopology()
    nesting += supergrid
    nesting += midgrid
    nesting.my_idx = nesting.nestings.index(midgrid)
    sim += nesting

    dxturb = midgrid.xsize / midgrid.itot * 4.0
    dyturb = midgrid.ysize / midgrid.jtot * 4.0
    openbc = do_openboundary(
        time0="2026-05-01T12:00:00",
        start="2026-05-01T12:00:00",
        end="2026-05-01T13:00:00",
        e12=0.01,
        dxint=midgrid.xsize / midgrid.itot * 4,
        dyint=midgrid.ysize / midgrid.jtot * 4,
        dxturb=dxturb,
        dyturb=dyturb,
        tauh=0.5,
        taum=0,
        lambda_=dxturb,
        lsynturb=True,
        tchunk=50,
    )
    openbc += Nest_in_Dales(
        inpath=parent_sim.output_path / "input",
        inpath_coarse=parent_sim.output_path / "input",
        outpath_coarse=parent_sim.output_path / "run_001",
        outpath_coarse_old=parent_sim.output_path / "run_001",
    )
    sim += openbc
    _attach_common_physics(sim, midgrid, True)

    sim_lsm = sim.retrieve_module(LSMModule)
    host_surfcross = xr.open_dataset(
        parent_sim.output_path / "run_001" / "surfcross.001.nc"
    )
    sim_lsm += UniformSkinTemperature(
        host_surfcross["tskin"].isel(time=0).mean().item()
    )
    host_surfcross.close()
    host_lsm_inp = xr.open_dataset(parent_sim.output_path / "input" / "lsm.inp_001.nc")
    sim_lsm += UniformSoilTemperature(
        host_lsm_inp["t_soil"].mean(dim=("x", "y")).values.tolist()
    )
    sim_lsm += UniformSoilMoisture(
        host_lsm_inp["theta_soil"].mean(dim=("x", "y")).values.tolist()
    )
    host_lsm_inp.close()

    return sim


def _run_job_direct(case_dir: Path, stage_name: str) -> None:
    logger.info("Running %s in %s", stage_name, case_dir)
    subprocess.run(["./job.001"], cwd=case_dir, check=True)


if __name__ == "__main__":
    import os

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # dask.config.set(scheduler="single-threaded")
    import webbrowser

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

    webbrowser.open(cluster.dashboard_link)

    print(f"Dask dashboard: {cluster.dashboard_link}")
    machine_conf_path = Path(__file__).resolve().parent / "machine_conf.yaml"
    with machine_conf_path.open("r", encoding="utf-8") as handle:
        machine_conf = yaml.safe_load(handle)

    logger.info("Building scaled64 triple-nesting case")
    parent_l1 = _build_outer_parent_sim_scaled64(machine_conf)
    parent_l1.sim_preprocessing_pipeline()
    _run_job_direct(parent_l1.output_path, "job_001_level1")

    parent_l2 = _build_middle_nested_sim_scaled64(machine_conf, parent_l1)
    parent_l2.sim_preprocessing_pipeline()
    _run_job_direct(parent_l2.output_path, "job_001_level2")

    logger.info("Completed scaled64 triple-nesting setup")
    logger.info(
        "Output path: %s",
        _build_inner_nested_sim_scaled64(machine_conf, parent_l2).output_path,
    )
