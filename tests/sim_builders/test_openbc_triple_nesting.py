from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    DefaultNamelistModule,
    GridDales,
    InterpolatedProfile,
    Nest_in_Dales,
    NestingTopology,
    TimeModule,
    dales_simulation,
    do_openboundary,
)
from modular_dales.Configuration.output_modules import (
    CapeModule,
    CrossSectionOutputModule,
    FielddumpModule,
    LSMCrossModule,
    RadfieldModule,
    StatsModule,
)
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.modular.simulation_module import set_nml_section
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM.LSM import (
    FromBofek,
    FromLCZ,
    FromTop10,
    LandUseModification,
    LSMModule,
)
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.Surface.LSM.SLuRB.slurb import (
    SLURBModule,
)
from modular_dales.vars import *
from tests.helpers import run_command_with_report

_GRID_SUPER = {
    "itot": 64,
    "jtot": 64,
    "kmax": 40,
    "xsize": 640.0,
    "ysize": 640.0,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": 136173.0 - (640.0 / 2.0),
    "y0": 455912.0 - (640.0 / 2.0),
    "proj4": "EPSG:28992",
    "alpha": 1.0,
    "dz0": 30.0,
}

_GRID_MID = {
    "itot": 32,
    "jtot": 32,
    "kmax": 36,
    "xsize": 320.0,
    "ysize": 320.0,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": 136173.0 - (640.0 / 2.0) + 160.0,
    "y0": 455912.0 - (640.0 / 2.0) + 160.0,
    "proj4": "EPSG:28992",
    "alpha": 1.0,
    "dz0": 30.0,
}

_GRID_INNER = {
    "itot": 16,
    "jtot": 16,
    "kmax": 32,
    "xsize": 160.0,
    "ysize": 160.0,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": 136173.0 - (640.0 / 2.0) + 240.0,
    "y0": 455912.0 - (640.0 / 2.0) + 240.0,
    "proj4": "EPSG:28992",
    "alpha": 1.0,
    "dz0": 30.0,
}


def _new_grid(spec: dict) -> GridDales:
    return GridDales(**spec)


def _attach_common_physics(
    sim: dales_simulation,
    grid: GridDales,
    center_vertical_crosssections: bool = False,
) -> None:
    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(
        variable=ua, shape="lin", params=dict(surf_val=1.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=va, shape="lin", params=dict(surf_val=1.5, ddz=1e-3)
    )
    atmo += InterpolatedProfile(
        variable=thetal,
        z=[0, 400, 410, 1600],
        points=[293.15, 293.15, 298.15, 301.15],
    )
    atmo += AtmosphericProfile(
        variable=qt, shape="lin", params=dict(surf_val=0.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=wa, shape="lin", params=dict(surf_val=0.0, ddz=0.0)
    )
    atmo += AtmosphericProfile(
        variable=tke, shape="lin", params=dict(surf_val=0.1, ddz=0.0)
    )
    sim += atmo

    sim += TimeModule(
        # xday=180,
        xtime=12.0,
        xyear=2023,
        runtime=60,
        startyear=2023,
        startmonth=7,
        startday=1,
        inferfromdatetime=True,
    )

    # sim += ConstantSurfaceTemperatureModule(
    #     thls=293.15,
    #     z0mav=0.0001,
    #     z0hav=0.0001,
    #     ps=100000.0,
    # )
    lsm = LSMModule(
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
    )
    lsm += UniformSkinTemperature(291)
    lsm += UniformSoilTemperature([288, 288, 288, 288])
    lsm += UniformSoilMoisture(
        [
            0.36867549,
            0.25300502,
            0.14997292,
            0.16459982,
        ]
    )
    lsm += LandUseModification(geometry=AllGeometry(), type="grs")
    lsm += FromLCZ()
    lsm += FromBofek(spatial_data_path="/Users/andrevanginkel/Downloads/spatial_data")
    lsm += FromTop10(spatial_data_path="/Users/andrevanginkel/Downloads/spatial_data")
    sim += lsm

    slurb = SLURBModule(deep_soil_temperature=293)
    # slurb += SLURBModification(
    #     geometry=AllGeometry(),
    #     vars=[SLURBVariableModification(varname="albedo_av", value=10)],
    # )
    sim += slurb

    sim += RadiationModule(
        iradiation=4,
    )

    sim += LSMCrossModule(enabled=True, dtav=10)
    sim += CapeModule(enabled=True, dtav=10)
    sim += StatsModule(enabled=True, dtav=10, timeav=10)
    sim += FielddumpModule(dtav=100, lfielddump=True)
    sim += RadfieldModule(enabled=True, dtav=10)

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
        xz_enabled=False,
        yz_enabled=False,
    )

    if sim.nml.get("namchecksim") is None:
        sim.nml["namchecksim"] = {}
    sim.nml["namchecksim"]["tcheck"] = 60

    if sim.nml.get("thermodynamics") is None:
        sim.nml["thermodynamics"] = {}
    # Keep nested domains thermodynamically consistent with base-state exner.
    sim.nml["thermodynamics"]["lbaseexner"] = True
    # sim.nml["thermodynamics"]["lconstexner"] = False


def _build_outer_parent_sim(machine_conf: dict) -> dales_simulation:
    sim = dales_simulation("openbc_triple_parent_l1", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID_SUPER)
    midgrid = _new_grid(_GRID_MID)

    sim += supergrid

    nesting = NestingTopology()
    nesting += supergrid
    nesting += midgrid
    nesting.my_idx = nesting.nestings.index(supergrid)
    sim += nesting

    _attach_common_physics(sim, supergrid, center_vertical_crosssections=True)
    return sim


def _build_middle_nested_sim(
    machine_conf: dict, parent_sim: dales_simulation
) -> dales_simulation:
    sim = dales_simulation("openbc_triple_parent_l2", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID_SUPER)
    midgrid = _new_grid(_GRID_MID)
    innergrid = _new_grid(_GRID_INNER)

    sim += midgrid

    nesting = NestingTopology()
    nesting += supergrid
    nesting += midgrid
    nesting += innergrid
    nesting.my_idx = nesting.nestings.index(midgrid)
    sim += nesting

    openbc = do_openboundary(
        time0="2023-01-01T12:00:10",
        start="2023-01-01T12:00:00",
        end="2023-01-01T12:01:00",
        e12=0.01,
        dxint=midgrid.xsize / midgrid.itot * 4,
        dyint=midgrid.ysize / midgrid.jtot * 4,
        tauh=20,
        taum=0,
    )
    openbc += Nest_in_Dales(
        inpath_coarse=parent_sim.output_path / "input",
        outpath_coarse=parent_sim.output_path / "run_001",
        outpath_coarse_old=parent_sim.output_path / "run_001",
    )
    sim += openbc

    _attach_common_physics(sim, midgrid, center_vertical_crosssections=True)
    return sim


def _build_inner_nested_sim(
    machine_conf: dict, middle_sim: dales_simulation
) -> dales_simulation:
    sim = dales_simulation("openbc_triple_child_l3", machine_conf)
    sim += DefaultNamelistModule()

    midgrid = _new_grid(_GRID_MID)
    innergrid = _new_grid(_GRID_INNER)

    sim += innergrid

    nesting = NestingTopology()
    nesting += midgrid
    nesting += innergrid
    nesting.my_idx = nesting.nestings.index(innergrid)
    sim += nesting

    openbc = do_openboundary(
        time0="2023-01-01T12:00:10",
        start="2023-01-01T12:00:00",
        end="2023-01-01T12:01:00",
        e12=0.01,
        dxint=innergrid.xsize / innergrid.itot * 4,
        dyint=innergrid.ysize / innergrid.jtot * 4,
        tauh=20,
        taum=0,
    )
    openbc += Nest_in_Dales(
        inpath_coarse=middle_sim.output_path / "input",
        outpath_coarse=middle_sim.output_path / "run_001",
        outpath_coarse_old=middle_sim.output_path / "run_001",
    )
    sim += openbc

    _attach_common_physics(sim, innergrid, center_vertical_crosssections=True)
    return sim


def openbc_triple_nesting_case(machine_conf: dict) -> dales_simulation:
    """Create DALES-in-DALES-in-DALES setup using two sequential Nest_in_Dales stages.

    The first two levels are preprocessed and run here so level 3 can generate
    open boundaries from level 2. The returned simulation is the level-3 run,
    executed by test_simulation_runs via generic test runner.
    """

    # machine_conf.setdefault("job_conf", {})["numcores"] = 1

    parent_l1 = _build_outer_parent_sim(machine_conf)
    parent_l1.sim_preprocessing_pipeline()
    run_command_with_report(
        ["./job.001"],
        stage="job_001_level1",
        case_dir=parent_l1.output_path,
        title="openbc triple nesting level-1 job.001 crash",
    )
    # run_command_with_report(
    #     ["combine.sh", "run_001"],
    #     stage="combine_run_001_level1",
    #     case_dir=parent_l1.output_path,
    #     title="openbc triple nesting level-1 combine crash",
    # )

    parent_l2 = _build_middle_nested_sim(machine_conf, parent_l1)
    parent_l2.sim_preprocessing_pipeline()
    run_command_with_report(
        ["./job.001"],
        stage="job_001_level2",
        case_dir=parent_l2.output_path,
        title="openbc triple nesting level-2 job.001 crash",
    )
    # run_command_with_report(
    #     ["combine.sh", "run_001"],
    #     stage="combine_run_001_level2",
    #     case_dir=parent_l2.output_path,
    #     title="openbc triple nesting level-2 combine crash",
    # )

    return _build_inner_nested_sim(machine_conf, parent_l2)


def openbc_triple_nesting_case_noparallel(machine_conf: dict) -> dales_simulation:
    """Create DALES-in-DALES-in-DALES setup using two sequential Nest_in_Dales stages.

    The first two levels are preprocessed and run here so level 3 can generate
    open boundaries from level 2. The returned simulation is the level-3 run,
    executed by test_simulation_runs via generic test runner.
    """

    # machine_conf.setdefault("job_conf", {})["numcores"] = 1

    parent_l1 = _build_outer_parent_sim(machine_conf)
    set_nml_section(
        parent_l1.nml,
        parent_l1.nml_docs,
        "set_serial",
        "namnetcdfstats",
        "lparallel",
        False,
    )
    parent_l1.sim_preprocessing_pipeline()
    run_command_with_report(
        ["./job.001"],
        stage="job_001_level1",
        case_dir=parent_l1.output_path,
        title="openbc triple nesting level-1 job.001 crash",
    )
    # run_command_with_report(
    #     ["combine.sh", "run_001"],
    #     stage="combine_run_001_level1",
    #     case_dir=parent_l1.output_path,
    #     title="openbc triple nesting level-1 combine crash",
    # )

    parent_l2 = _build_middle_nested_sim(machine_conf, parent_l1)
    set_nml_section(
        parent_l2.nml,
        parent_l2.nml_docs,
        "set_serial",
        "namnetcdfstats",
        "lparallel",
        False,
    )
    parent_l2.sim_preprocessing_pipeline()
    run_command_with_report(
        ["./job.001"],
        stage="job_001_level2",
        case_dir=parent_l2.output_path,
        title="openbc triple nesting level-2 job.001 crash",
    )
    # run_command_with_report(
    #     ["combine.sh", "run_001"],
    #     stage="combine_run_001_level2",
    #     case_dir=parent_l2.output_path,
    #     title="openbc triple nesting level-2 combine crash",
    # )

    return _build_inner_nested_sim(machine_conf, parent_l2)


_GRID64_SUPER = {
    "itot": 64,
    "jtot": 64,
    "kmax": 72,
    "xsize": 6400.0 * 1.5,
    "ysize": 6400.0 * 1.5,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": 136173.0 - ((6400.0 * 1.5) / 2.0),
    "y0": 455912.0 - ((6400.0 * 1.5) / 2.0),
    "proj4": "EPSG:28992",
    "alpha": 1.02,
    "dz0": 10.0,
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
    "x0": 136173.0 - ((6400.0 * 1.5) / 2.0) + (2400.0 * 1.5),
    "y0": 455912.0 - ((6400.0 * 1.5) / 2.0) + (2400.0 * 1.5),
    "proj4": "EPSG:28992",
    "alpha": 1,
    "dz0": 10.0,
}

_GRID64_INNER = {
    "itot": 64,
    "jtot": 64,
    "kmax": 40,
    "xsize": 400.0 * 1.5,
    "ysize": 400.0 * 1.5,
    "kmax_soil": 4,
    "xlat": 52.25,
    "xlon": 5.45,
    "x0": 136173.0 - ((6400.0 * 1.5) / 2.0) + (3000.0 * 1.5),
    "y0": 455912.0 - ((6400.0 * 1.5) / 2.0) + (3000.0 * 1.5),
    "proj4": "EPSG:28992",
    "alpha": 1,
    "dz0": 10.0,
}


def _build_outer_parent_sim_scaled64(machine_conf: dict) -> dales_simulation:
    sim = dales_simulation("openbc_triple64_parent_l1", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID64_SUPER)
    midgrid = _new_grid(_GRID64_MID)
    midgrid.inherit_vertical_grid(supergrid)

    sim += supergrid

    nesting = NestingTopology()
    nesting += supergrid
    nesting += midgrid
    nesting.my_idx = nesting.nestings.index(supergrid)
    sim += nesting

    _attach_common_physics(sim, supergrid, True)
    return sim


def _build_middle_nested_sim_scaled64(
    machine_conf: dict, parent_sim: dales_simulation
) -> dales_simulation:
    sim = dales_simulation("openbc_triple64_parent_l2", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID64_SUPER)
    midgrid = _new_grid(_GRID64_MID)
    midgrid.inherit_vertical_grid(supergrid)
    innergrid = _new_grid(_GRID64_INNER)
    innergrid.inherit_vertical_grid(midgrid)

    sim += midgrid

    nesting = NestingTopology()
    nesting += supergrid
    nesting += midgrid
    nesting += innergrid
    nesting.my_idx = nesting.nestings.index(midgrid)
    sim += nesting
    dxturb = midgrid.xsize / midgrid.itot * 4.0
    dyturb = midgrid.ysize / midgrid.jtot * 4.0
    openbc = do_openboundary(
        time0="2023-01-01T12:00:00",
        start="2023-01-01T12:00:00",
        end="2023-01-01T13:00:00",
        e12=0.01,
        dxint=midgrid.xsize / midgrid.itot,
        dyint=midgrid.ysize / midgrid.jtot,
        dxturb=dxturb,
        dyturb=dyturb,
        tauh=20,
        taum=100,
        lambda_=dxturb,
        lsynturb=False,
    )
    openbc += Nest_in_Dales(
        inpath=parent_sim.output_path / "input",
        inpath_coarse=parent_sim.output_path / "input",
        outpath_coarse=parent_sim.output_path / "run_001",
        outpath_coarse_old=parent_sim.output_path / "run_001",
    )
    sim += openbc

    _attach_common_physics(sim, midgrid, True)
    return sim


def _build_inner_nested_sim_scaled64(
    machine_conf: dict, middle_sim: dales_simulation
) -> dales_simulation:
    sim = dales_simulation("openbc_triple64_child_l3", machine_conf)
    sim += DefaultNamelistModule()

    supergrid = _new_grid(_GRID64_SUPER)
    midgrid = _new_grid(_GRID64_MID)
    midgrid.inherit_vertical_grid(supergrid)
    innergrid = _new_grid(_GRID64_INNER)
    innergrid.inherit_vertical_grid(midgrid)

    sim += innergrid

    nesting = NestingTopology()
    nesting += midgrid
    nesting += innergrid
    nesting.my_idx = nesting.nestings.index(innergrid)
    sim += nesting

    dxturb = innergrid.xsize / innergrid.itot * 4.0
    dyturb = innergrid.ysize / innergrid.jtot * 4.0
    openbc = do_openboundary(
        time0="2023-01-01T12:00:00",
        start="2023-01-01T12:00:00",
        end="2023-01-01T13:00:00",
        e12=0.01,
        dxint=innergrid.xsize / innergrid.itot,
        dyint=innergrid.ysize / innergrid.jtot,
        dxturb=dxturb,
        dyturb=dyturb,
        tauh=20,
        taum=100,
        lambda_=dxturb,
        lsynturb=False,
    )
    openbc += Nest_in_Dales(
        inpath=middle_sim.output_path / "input",
        inpath_coarse=middle_sim.output_path / "input",
        outpath_coarse=middle_sim.output_path / "run_001",
        outpath_coarse_old=middle_sim.output_path / "run_001",
    )
    sim += openbc

    _attach_common_physics(sim, innergrid, True)
    return sim


def _plot_nested_grids_xy(plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    specs = [
        ("L1", _GRID64_SUPER, "tab:blue"),
        ("L2", _GRID64_MID, "tab:orange"),
        ("L3", _GRID64_INNER, "tab:green"),
    ]
    for label, spec, color in specs:
        rect = Rectangle(
            (spec["x0"], spec["y0"]),
            spec["xsize"],
            spec["ysize"],
            fill=False,
            lw=2.0,
            ec=color,
            label=f"{label}: {int(spec['itot'])}x{int(spec['jtot'])}, L={spec['xsize']} m",
        )
        ax.add_patch(rect)
        ax.text(spec["x0"], spec["y0"], label, color=color, fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Triple nesting with fixed 64x64 and xsize/ysize / 4")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "nested_grids_xy.png")
    plt.close(fig)


def _first_match(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def _vertical_cross_nearest_match(
    run_dir: Path,
    pattern: str,
    coord_candidates: tuple[str, ...],
    target: float,
) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None

    best_path = matches[0]
    best_dist = float("inf")

    for path in matches:
        try:
            with xr.open_dataset(path, engine="netcdf4") as ds:
                coord_name = next(
                    (name for name in coord_candidates if name in ds.coords), None
                )
                if coord_name is None:
                    continue
                vals = np.asarray(ds[coord_name].values, dtype=float).reshape(-1)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                dist = abs(float(vals[0]) - float(target))
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
        except Exception:
            continue

    return best_path


def _lowest_crossxy_match(run_dir: Path) -> Path | None:
    """Pick the crossxy file with the lowest vertical coordinate.

    This avoids accidentally plotting a higher slice when multiple XY
    cross-sections are present.
    """
    best_path = None
    best_level = None
    for path in sorted(run_dir.glob("crossxy.*.*.nc")):
        with xr.open_dataset(path, engine="netcdf4") as ds:
            level_value = None
            for coord_name in ("zt", "zm"):
                if coord_name in ds.coords:
                    values = np.asarray(ds[coord_name].values)
                    if values.size > 0:
                        try:
                            level_value = float(np.nanmin(values.astype(float)))
                        except (TypeError, ValueError):
                            level_value = float(values[0])
                        break
            if level_value is None:
                continue
            if best_level is None or level_value < best_level:
                best_level = level_value
                best_path = path
    return best_path


def _resolve_shared_plot_time(level_dirs: dict[str, Path]):
    """Resolve one shared plotting time across L1/L2/L3 cross sections."""
    patterns = ("crossyz.*.*.nc", "crossxz.*.*.nc", "crossxy.*.*.nc")
    per_level_limits_dt = []
    per_level_limits_num: list[float] = []

    resolved_kind = None  # "datetime" or "numeric"

    for level_tag in ("L1", "L2", "L3"):
        run_dir = level_dirs[level_tag] / "run_001"
        maxima_for_level_dt = []
        maxima_for_level_num: list[float] = []

        for pattern in patterns:
            path = _first_match(run_dir, pattern)
            if path is None:
                continue
            with xr.open_dataset(path, engine="netcdf4") as ds:
                if "time" not in ds.coords:
                    continue
                vals = np.asarray(ds["time"].values)
                if vals.size == 0:
                    continue

                if np.issubdtype(vals.dtype, np.datetime64):
                    if resolved_kind is None:
                        resolved_kind = "datetime"
                    if resolved_kind != "datetime":
                        continue
                    maxima_for_level_dt.append(vals.max())
                else:
                    if resolved_kind is None:
                        resolved_kind = "numeric"
                    if resolved_kind != "numeric":
                        continue
                    vals_num = np.asarray(vals, dtype=float)
                    vals_num = vals_num[np.isfinite(vals_num)]
                    if vals_num.size > 0:
                        maxima_for_level_num.append(float(vals_num.max()))

        if resolved_kind == "datetime" and maxima_for_level_dt:
            # Choose time guaranteed to exist across this level's cross outputs.
            per_level_limits_dt.append(min(maxima_for_level_dt))
        elif resolved_kind == "numeric" and maxima_for_level_num:
            # Choose time guaranteed to exist across this level's cross outputs.
            per_level_limits_num.append(min(maxima_for_level_num))

    if resolved_kind == "datetime":
        if not per_level_limits_dt:
            return None
        # Shared time available for all levels.
        return min(per_level_limits_dt)

    if resolved_kind == "numeric":
        if not per_level_limits_num:
            return None
        # Shared time available for all levels.
        return min(per_level_limits_num)

    return None


def _select_time_by_coord(da: xr.DataArray, target_time):
    """Select by time coordinate (never by integer index)."""
    if target_time is None or "time" not in da.dims:
        return da

    coord_vals = np.asarray(da["time"].values)
    if coord_vals.size == 0:
        return da

    if np.issubdtype(coord_vals.dtype, np.datetime64):
        if isinstance(target_time, np.datetime64):
            sel_time = target_time
        else:
            t = float(target_time)
            # If value looks like epoch nanoseconds, interpret accordingly.
            if abs(t) > 1e12:
                sel_time = np.datetime64(int(round(t)), "ns")
            else:
                base = coord_vals[0].astype("datetime64[ns]")
                sel_time = base + np.timedelta64(int(round(t)), "s")
        out = da.sel(time=sel_time, method="nearest")
    else:
        out = da.sel(time=float(target_time), method="nearest")

    if "time" in out.dims:
        out = out.squeeze("time", drop=True)
    return out


def _format_plot_time(target_time) -> str:
    if target_time is None:
        return ""
    if isinstance(target_time, np.datetime64):
        return np.datetime_as_string(target_time, unit="s")
    return f"{float(target_time):.1f}s"


def _select_first_model_level(da: xr.DataArray) -> xr.DataArray:
    """Select the lowest vertical model level by coordinate value.

    This avoids relying on array order, which can differ between files.
    """
    zdim = next((d for d in da.dims if d in ("zt", "zm")), None)
    if zdim is None or zdim not in da.coords:
        return da

    zvals = np.asarray(da[zdim].values)
    if zvals.size == 0:
        return da

    try:
        level_value = zvals[np.nanargmin(zvals.astype(float))]
    except (TypeError, ValueError):
        level_value = zvals[0]

    return da.sel({zdim: level_value}, method="nearest")


def _axis_offset_for_local_coords(
    values: xr.DataArray, origin: float, size: float
) -> float:
    vals = values.values
    vmin = float(vals.min())
    vmax = float(vals.max())
    tol = max(1.0e-6, size * 1.0e-3)
    is_local = (-tol <= vmin) and (vmax <= size + tol)
    return origin if is_local else 0.0


def _overlay_xy_nested_boxes(ax, ds: xr.Dataset, level_specs: dict[str, dict]) -> None:
    from matplotlib.patches import Rectangle

    x_axis = "xt" if "xt" in ds.coords else "xm"
    y_axis = "yt" if "yt" in ds.coords else "ym"

    l1 = level_specs["L1"]
    xoff = _axis_offset_for_local_coords(ds[x_axis], l1["x0"], l1["xsize"])
    yoff = _axis_offset_for_local_coords(ds[y_axis], l1["y0"], l1["ysize"])

    color_map = {"L1": "tab:blue", "L2": "tab:orange", "L3": "tab:green"}
    for label in ("L1", "L2", "L3"):
        spec = level_specs[label]
        rect = Rectangle(
            (spec["x0"] - xoff, spec["y0"] - yoff),
            spec["xsize"],
            spec["ysize"],
            fill=False,
            lw=0.9,
            ec=color_map[label],
            label=f"{label} extent",
        )
        ax.add_patch(rect)


def _overlay_cross_refinement_spans(
    ax, ds: xr.Dataset, horizontal_dim: str, level_specs: dict[str, dict]
) -> None:
    axis_key = "x" if horizontal_dim in ("xt", "xm") else "y"
    size_key = "xsize" if axis_key == "x" else "ysize"
    origin_key = "x0" if axis_key == "x" else "y0"

    l1 = level_specs["L1"]
    hoff = _axis_offset_for_local_coords(
        ds[horizontal_dim], l1[origin_key], l1[size_key]
    )

    color_map = {"L1": "tab:blue", "L2": "tab:orange", "L3": "tab:green"}
    for label in ("L1", "L2", "L3"):
        spec = level_specs[label]
        h0 = spec[origin_key] - hoff
        h1 = h0 + spec[size_key]
        ax.axvline(h0, color=color_map[label], lw=0.8, alpha=0.9)
        ax.axvline(h1, color=color_map[label], lw=0.8, alpha=0.9)
        ax.axvspan(h0, h1, color=color_map[label], alpha=0.04)


def _plot_crosssections_for_level(
    level_dir: Path,
    level_tag: str,
    plot_dir: Path,
    level_specs: dict[str, dict],
    target_time: float | None,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = level_dir / "run_001"
    plotted = 0

    level_spec = level_specs[level_tag]
    center_x = float(level_spec["x0"]) + 0.5 * float(level_spec["xsize"])
    center_y = float(level_spec["y0"]) + 0.5 * float(level_spec["ysize"])

    cross_files = {
        "crossyz": _vertical_cross_nearest_match(
            run_dir, "crossyz.*.*.nc", ("xt", "xm"), center_x
        ),
        "crossxz": _vertical_cross_nearest_match(
            run_dir, "crossxz.*.*.nc", ("yt", "ym"), center_y
        ),
        "crossxy": _lowest_crossxy_match(run_dir),
    }

    var_candidates = {
        "u": ["u"],
        "v": ["v"],
        "w": ["w"],
        "thl": ["thl"],
        "qt": ["qt"],
        "e12": ["e12", "e120"],
    }

    for cross_name, cross_path in cross_files.items():
        if cross_path is None:
            continue

        with xr.open_dataset(cross_path, engine="netcdf4") as ds:
            for var_label, names in var_candidates.items():
                var_name = next((cand for cand in names if cand in ds.variables), None)
                if var_name is None:
                    continue

                da = _select_time_by_coord(ds[var_name], target_time)

                if cross_name == "crossxy":
                    da = _select_first_model_level(da)

                while da.ndim > 2:
                    da = da.isel({da.dims[-1]: 0}, drop=True)

                if da.ndim != 2:
                    continue

                fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=130)
                vertical_dim = next((d for d in da.dims if d in ("zt", "zm")), None)
                horizontal_dim = next(
                    (d for d in da.dims if d in ("xt", "xm", "yt", "ym")), None
                )

                if vertical_dim is not None and horizontal_dim is not None:
                    da.plot(ax=ax, robust=True, x=horizontal_dim, y=vertical_dim)
                else:
                    da.plot(ax=ax, robust=True)

                if cross_name == "crossxy":
                    _overlay_xy_nested_boxes(ax, ds, level_specs)
                elif horizontal_dim is not None:
                    _overlay_cross_refinement_spans(ax, ds, horizontal_dim, level_specs)

                if target_time is None:
                    ax.set_title(f"{level_tag} {cross_name} {var_label}")
                else:
                    ax.set_title(
                        f"{level_tag} {cross_name} {var_label} at t~{_format_plot_time(target_time)}"
                    )
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                fig.savefig(plot_dir / f"{level_tag}_{cross_name}_{var_label}.png")
                plt.close(fig)
                plotted += 1

    return plotted


def _global_axis_for_dim(values: np.ndarray, dim: str, spec: dict) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if dim in ("xt", "xm"):
        off = _axis_offset_for_local_coords(
            xr.DataArray(arr), float(spec["x0"]), float(spec["xsize"])
        )
        return arr + off
    if dim in ("yt", "ym"):
        off = _axis_offset_for_local_coords(
            xr.DataArray(arr), float(spec["y0"]), float(spec["ysize"])
        )
        return arr + off
    return arr


def _extract_plot_axes(da: xr.DataArray, cross_name: str) -> tuple[str, str]:
    if cross_name == "crossxy":
        xdim = "xt" if "xt" in da.dims else "xm"
        ydim = "yt" if "yt" in da.dims else "ym"
        return xdim, ydim

    if cross_name == "crossxz":
        xdim = "xt" if "xt" in da.dims else "xm"
        ydim = "zt" if "zt" in da.dims else "zm"
        return xdim, ydim

    if cross_name == "crossyz":
        xdim = "yt" if "yt" in da.dims else "ym"
        ydim = "zt" if "zt" in da.dims else "zm"
        return xdim, ydim

    # Fallback for unexpected naming
    dims = list(da.dims)
    return dims[-1], dims[0]


def _mask_for_refinement(
    arr: np.ndarray,
    xvals: np.ndarray,
    yvals: np.ndarray,
    cross_name: str,
    level_tag: str,
    level_specs: dict[str, dict],
) -> np.ndarray:
    """Mask coarse values where finer grid values should be shown.

    L1 is masked inside L2, L2 is masked inside L3, L3 remains unmasked.
    """
    out = np.array(arr, copy=True)

    if level_tag == "L3":
        return out

    finer_tag = "L2" if level_tag == "L1" else "L3"
    finer = level_specs[finer_tag]

    if cross_name == "crossxy":
        xx, yy = np.meshgrid(xvals, yvals)
        inside = (
            (xx >= float(finer["x0"]))
            & (xx <= float(finer["x0"]) + float(finer["xsize"]))
            & (yy >= float(finer["y0"]))
            & (yy <= float(finer["y0"]) + float(finer["ysize"]))
        )
        out = np.where(inside, np.nan, out)
        return out

    # crossxz uses horizontal x; crossyz uses horizontal y.
    axis_key = "x" if cross_name == "crossxz" else "y"
    okey = "x0" if axis_key == "x" else "y0"
    skey = "xsize" if axis_key == "x" else "ysize"
    h0 = float(finer[okey])
    h1 = h0 + float(finer[skey])
    inside_1d = (xvals >= h0) & (xvals <= h1)
    if out.ndim == 2:
        out[:, inside_1d] = np.nan
    return out


def _plot_refinement_value_overlays(
    level_dirs: dict[str, Path],
    level_specs: dict[str, dict],
    plot_dir: Path,
    target_time: float | None,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    cross_patterns = {
        "crossyz": "crossyz.*.*.nc",
        "crossxz": "crossxz.*.*.nc",
        "crossxy": "crossxy.*.*.nc",
    }
    var_candidates = {
        "u": ["u"],
        "v": ["v"],
        "w": ["w"],
        "thl": ["thl"],
        "qt": ["qt"],
        "e12": ["e12", "e120"],
    }
    alpha_map = {"L1": 1.00, "L2": 1.00, "L3": 1.00}

    made = 0
    for cross_name, pattern in cross_patterns.items():
        for var_label, candidates in var_candidates.items():
            per_level: dict[str, tuple[xr.DataArray, dict]] = {}
            for level_tag in ("L1", "L2", "L3"):
                level_dir = level_dirs[level_tag]
                if cross_name == "crossxy":
                    path = _lowest_crossxy_match(level_dir / "run_001")
                else:
                    path = _first_match(level_dir / "run_001", pattern)
                if path is None:
                    continue

                with xr.open_dataset(path, engine="netcdf4") as ds:
                    var_name = next((c for c in candidates if c in ds.variables), None)
                    if var_name is None:
                        continue
                    da = _select_time_by_coord(ds[var_name], target_time)

                    if cross_name == "crossxy":
                        da = _select_first_model_level(da)

                    while da.ndim > 2:
                        da = da.isel({da.dims[-1]: 0}, drop=True)
                    if da.ndim != 2:
                        continue
                    per_level[level_tag] = (da.load(), level_specs[level_tag])

            if "L1" not in per_level or len(per_level) < 2:
                continue

            all_vals = []
            for da, _ in per_level.values():
                vals = np.asarray(da.values, dtype=float)
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    all_vals.append(finite)
            if not all_vals:
                continue
            vmin = float(min(np.min(v) for v in all_vals))
            vmax = float(max(np.max(v) for v in all_vals))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=135)
            base_mappable = None

            for level_tag in ("L1", "L2", "L3"):
                if level_tag not in per_level:
                    continue
                da, spec = per_level[level_tag]
                xdim, ydim = _extract_plot_axes(da, cross_name)
                if xdim not in da.dims or ydim not in da.dims:
                    continue

                arr = da.transpose(ydim, xdim).values.astype(float)
                xvals = _global_axis_for_dim(np.asarray(da[xdim].values), xdim, spec)
                yvals = _global_axis_for_dim(np.asarray(da[ydim].values), ydim, spec)

                arr = _mask_for_refinement(
                    arr=arr,
                    xvals=xvals,
                    yvals=yvals,
                    cross_name=cross_name,
                    level_tag=level_tag,
                    level_specs=level_specs,
                )

                m = ax.pcolormesh(
                    xvals,
                    yvals,
                    arr,
                    shading="auto",
                    cmap="viridis",
                    norm=norm,
                    alpha=alpha_map[level_tag],
                    zorder={"L1": 1, "L2": 2, "L3": 3}[level_tag],
                )
                if base_mappable is None:
                    base_mappable = m

            if base_mappable is None:
                plt.close(fig)
                continue

            # Draw explicit extents to make overlap legible.
            if cross_name == "crossxy":
                for tag, color in (
                    ("L1", "tab:blue"),
                    ("L2", "tab:orange"),
                    ("L3", "tab:green"),
                ):
                    spec = level_specs[tag]
                    ax.plot(
                        [
                            spec["x0"],
                            spec["x0"] + spec["xsize"],
                            spec["x0"] + spec["xsize"],
                            spec["x0"],
                            spec["x0"],
                        ],
                        [
                            spec["y0"],
                            spec["y0"],
                            spec["y0"] + spec["ysize"],
                            spec["y0"] + spec["ysize"],
                            spec["y0"],
                        ],
                        color=color,
                        lw=0.9,
                        label=f"{tag} extent",
                    )
                ax.set_xlabel("horizontal [m]")
                ax.set_ylabel("horizontal [m]")
            else:
                axis_key = "x" if cross_name == "crossxz" else "y"
                okey = "x0" if axis_key == "x" else "y0"
                skey = "xsize" if axis_key == "x" else "ysize"
                for tag, color in (
                    ("L1", "tab:blue"),
                    ("L2", "tab:orange"),
                    ("L3", "tab:green"),
                ):
                    spec = level_specs[tag]
                    h0 = float(spec[okey])
                    h1 = h0 + float(spec[skey])
                    ax.axvline(h0, color=color, lw=0.8, alpha=0.9)
                    ax.axvline(h1, color=color, lw=0.8, alpha=0.9)
                    ax.axvspan(h0, h1, color=color, alpha=0.04)
                ax.set_xlabel(f"{axis_key} [m]")
                ax.set_ylabel("z [m]")

            cbar = fig.colorbar(base_mappable, ax=ax)
            cbar.set_label(var_label)
            if target_time is None:
                ax.set_title(
                    f"Overlay {cross_name} {var_label}: L1 base + L2/L3 refinement"
                )
            else:
                ax.set_title(
                    f"Overlay {cross_name} {var_label} at t~{_format_plot_time(target_time)}: L1 base + L2/L3 refinement"
                )
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"overlay_{cross_name}_{var_label}.png")
            plt.close(fig)
            made += 1

    return made


def assert_openbc_triple_nesting_scaled64_plots(output_path: Path) -> None:
    """Generate diagnostic nesting/cross-section plots for the scaled-64 triple test."""

    case_root = output_path.parent
    level_dirs = {
        "L1": case_root / "openbc_triple64_parent_l1",
        "L2": case_root / "openbc_triple64_parent_l2",
        "L3": output_path,
    }
    level_specs = {
        "L1": _GRID64_SUPER,
        "L2": _GRID64_MID,
        "L3": _GRID64_INNER,
    }

    plot_dir = output_path / "pytest_artifacts" / "triple64_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    shared_time = _resolve_shared_plot_time(level_dirs)

    _plot_nested_grids_xy(plot_dir)

    total_plots = 1
    for level_tag, level_dir in level_dirs.items():
        if not level_dir.is_dir():
            continue
        total_plots += _plot_crosssections_for_level(
            level_dir, level_tag, plot_dir, level_specs, shared_time
        )

    total_plots += _plot_refinement_value_overlays(
        level_dirs, level_specs, plot_dir, shared_time
    )

    if total_plots <= 1:
        raise AssertionError(
            f"No cross-section plots were generated. Checked levels: {level_dirs}"
        )


def openbc_triple_nesting_scaled64_case(machine_conf: dict) -> dales_simulation:
    """Triple DALES nesting with fixed 64x64 and xsize/ysize reduced by factor 4 per level."""

    # machine_conf.setdefault("job_conf", {})["numcores"] = 1

    parent_l1 = _build_outer_parent_sim_scaled64(machine_conf)
    parent_l1.sim_preprocessing_pipeline()
    run_command_with_report(
        ["./job.001"],
        stage="job_001_level1",
        case_dir=parent_l1.output_path,
        title="openbc triple64 level-1 job.001 crash",
    )

    parent_l2 = _build_middle_nested_sim_scaled64(machine_conf, parent_l1)
    parent_l2.sim_preprocessing_pipeline()
    run_command_with_report(
        ["./job.001"],
        stage="job_001_level2",
        case_dir=parent_l2.output_path,
        title="openbc triple64 level-2 job.001 crash",
    )

    return _build_inner_nested_sim_scaled64(machine_conf, parent_l2)
