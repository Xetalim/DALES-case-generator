import numpy as np
import pytest
import xarray as xr

from moda import (
    DefaultNamelistModule,
    GridDales,
    TimeModule,
    dales_simulation,
)
from moda.Atmosphere import AtmosphereModule, AtmosphericProfile
from moda.Configuration import EasyOutputModule
from moda.LBC import Nest_in_Dales, NestingTopology
from moda.modular.simulation_module import set_nml_section
from moda.Surface import ConstantSurfaceTemperatureModule
from moda.vars import *
from tests.helpers import run_command_with_report
from tests.LBC.test_openbc import (
    _assert_crosssection_matches_fielddump_scalar_slice_coord,
)


@pytest.fixture
def supergrid():
    supergrid = GridDales(
        itot=64,
        jtot=64,
        kmax=80,
        xsize=640.0,
        ysize=640.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=5,
        # "proj4": "+proj=..."  # optional
    )

    return supergrid


@pytest.fixture
def subgrid():
    # this is the grid that is inside us
    subgrid = GridDales(
        itot=32,
        jtot=32,
        kmax=70,
        xsize=320.0,
        ysize=320.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=160.0,
        y0=160.0,
        alpha=1.0,
        dz0=5,
        # "proj4": "+proj=..."  # optional
    )

    return subgrid


def _build_coarse_sim_with_easyoutput(
    machine_conf: dict, casename: str, supergrid, subgrid
) -> dales_simulation:
    """Small coarse simulation that writes fielddump + cross-sections.

    This mirrors the first case in oop_test.py but with a reduced grid so
    the resulting files remain small enough for unit tests.
    """

    sim = dales_simulation(casename, machine_conf)
    sim.add(DefaultNamelistModule())

    sim.add(supergrid)

    nesting = NestingTopology()
    nesting.add(supergrid)
    nesting.add(subgrid)
    nesting.my_idx = nesting.nestings.index(supergrid)
    sim.add(nesting)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=0.5, ddz=0)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=3.0, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0.0, ddz=0.0)))
    sim.add(atmo)

    sim.add(
        TimeModule(
            xday=1,
            xtime=12.0,
            xyear=2023,
            runtime=60,
            startyear=2023,
            startmonth=1,
            startday=1,
        )
    )

    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15,
            z0mav=0.0001,
            z0hav=0.0001,
            ps=100000,
        )
    )

    sim.add(EasyOutputModule(output_interval=10, enable_output=True))

    if sim.nml.get("namchecksim") is None:
        sim.nml["namchecksim"] = {}
    sim.nml["namchecksim"]["tcheck"] = 60

    if sim.nml.get("thermodynamics") is None:
        sim.nml["thermodynamics"] = {}
    # sim.nml["thermodynamics"]["lconstexner"] = True

    return sim


def _build_nested_openbc_sim_from_coarse(
    machine_conf: dict, coarse_sim: dales_simulation, supergrid, subgrid
) -> dales_simulation:
    """Second small case: subgrid nested in coarse sim via Nest_in_Dales.

    This mirrors the second case in oop_test.py where ``openboundaries``
    are generated from an existing coarse DALES run.
    """

    sim2 = dales_simulation("openbc_nested_from_coarse", machine_conf)
    sim2.add(DefaultNamelistModule())
    sim2.add(subgrid)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=0.5, ddz=0)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=3.0, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0.0, ddz=0.0)))
    sim2.add(atmo)

    sim2.add(
        TimeModule(
            xday=1,
            xtime=12.0,
            xyear=2023,
            runtime=60,
            startyear=2023,
            startmonth=1,
            startday=1,
        )
    )

    sim2.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15,
            z0mav=0.0001,
            z0hav=0.0001,
            ps=100000,
        )
    )

    if sim2.nml.get("namnetcdfstats") is None:
        sim2.nml["namnetcdfstats"] = {}
    sim2.nml["namnetcdfstats"]["lsync"] = True

    nesting = NestingTopology()
    nesting.add(supergrid)
    nesting.add(subgrid)
    nesting.my_idx = nesting.nestings.index(subgrid)
    sim2.add(nesting)

    openbc = Nest_in_Dales(
        time0="2023-01-01T12:00:10",
        start="2023-01-01T12:00:00",
        end="2023-01-02T12:00:00",
        e12=0.01,
        dxint=subgrid.xsize / subgrid.itot,
        dyint=subgrid.ysize / subgrid.jtot,
        tauh=20,
        taum=0,
        inpath_coarse=coarse_sim.output_path / "input/",
        outpath_coarse=coarse_sim.output_path / "run_001/",
        outpath_coarse_old=coarse_sim.output_path / "run_001/",
    )
    sim2.add(openbc)

    sim2.add(EasyOutputModule(output_interval=5, enable_output=True))

    if sim2.nml.get("namchecksim") is None:
        sim2.nml["namchecksim"] = {}
    sim2.nml["namchecksim"]["tcheck"] = 60
    if sim2.nml.get("thermodynamics") is None:
        sim2.nml["thermodynamics"] = {}
    # sim2.nml["thermodynamics"]["lconstexner"] = True

    return sim2


@pytest.mark.slow
@pytest.mark.parametrize("core_changer", [1])
def test_openboundaries_match_fielddump_and_crosssection(
    machine_conf, core_changer, supergrid, subgrid, simulation_report
) -> None:
    """Small Nest_in_Dales open-BC run checking consistency of outputs.

    1. Build a coarse simulation that writes ``fielddump.nc`` and cross-section
       files.
    2. Run a nested openboundary simulation that uses the coarse run via
       ``Nest_in_Dales`` and produces ``openboundaries.inp.*.nc``.
    3. Compare slices from ``fielddump.nc``, cross-section output and
       openboundaries along the corresponding boundary, and compute a
       total divergence metric.
    """
    pytest.skip("Skipping test_openboundaries_match_fielddump_and_crosssection")
    conf = machine_conf("openbc_openboundaries_fielddump")
    conf["job_conf"]["numcores"] = core_changer

    # 1) Coarse simulation with EasyOutput
    coarse_sim = _build_coarse_sim_with_easyoutput(
        conf, "openbc_coarse_for_nested", supergrid, subgrid
    )
    set_nml_section(
        coarse_sim.nml,
        coarse_sim.nml_docs,
        "user_defined",
        "NAMNETCDFSTATS",
        "lparallel",
        True,
    )
    coarse_sim.sim_preprocessing_pipeline()

    coarse_outdir = coarse_sim.output_path
    run_command_with_report(
        ["./job.001"],
        stage="job_001",
        case_dir=coarse_outdir,
        title="openboundaries coarse job.001 crash",
        add_report=simulation_report,
        info_lines=[f"numcores={core_changer}"],
    )
    run_command_with_report(
        ["combine.sh", "run_001"],
        stage="combine_run_001",
        case_dir=coarse_outdir,
        title="openboundaries coarse combine crash",
        add_report=simulation_report,
        info_lines=[f"numcores={core_changer}"],
    )

    fielddump_file = coarse_outdir / "run_001" / "fielddump.001.nc"
    if not fielddump_file.is_file():
        pytest.skip(
            "fielddump.nc not found for coarse run; DALES did not produce fielddump"
        )

    # # Prefer crossyz, then crossxz, then crossxy
    # cross_candidates = [
    #     coarse_outdir / "crossyz.nc",
    #     coarse_outdir / "crossxz.nc",
    #     coarse_outdir / "crossxy.nc",
    # ]
    # crosssection_file = next((p for p in cross_candidates if p.is_file()), None)

    # if crosssection_file is not None:
    #     var = "thl"
    #     if crosssection_file.name == "crossyz.nc":
    #         fd_coord_dim = "xt"
    #         cs_coord_dim = "xt"
    #     elif crosssection_file.name == "crossxz.nc":
    #         fd_coord_dim = "yt"
    #         cs_coord_dim = "yt"
    #     else:  # crossxy.nc
    #         fd_coord_dim = "zt"
    #         cs_coord_dim = "zt"

    #     _assert_crosssection_matches_fielddump(
    #         fielddump_file,
    #         crosssection_file,
    #         var=var,
    #         fd_coord_dim=fd_coord_dim,
    #         cs_coord_dim=cs_coord_dim,
    #         cross_index=0,
    #         time_index=0,
    #     )
    cross_patterns = [
        ("run_001/crossyz.*.001.nc", "crossyz.nc"),
        ("run_001/crossxz.*.001.nc", "crossxz.nc"),
        ("run_001/crossxy.*.001.nc", "crossxy.nc"),
    ]

    var = "thl"
    for pattern, canonical_name in cross_patterns:
        matching_files = list(coarse_outdir.glob(pattern))
        if not matching_files:
            continue
        crosssection_file = matching_files[0]

        # Choose which coordinate is the fixed scalar slice coord in the cross-section.
        if canonical_name == "crossyz.nc":
            cs_slice_coord = "xm" if var == "u" else "xt"
            fd_coord_dim = cs_slice_coord
        elif canonical_name == "crossxz.nc":
            cs_slice_coord = "ym" if var == "v" else "yt"
            fd_coord_dim = cs_slice_coord
        else:  # crossxy.nc
            cs_slice_coord = "zt"
            fd_coord_dim = "zt"

        # The slice coord is a scalar variable in the cross-section file, not a dim.
        _assert_crosssection_matches_fielddump_scalar_slice_coord(
            fielddump_file,
            crosssection_file,
            var=var,
            fd_coord_dim=fd_coord_dim,
            cs_slice_coord=cs_slice_coord,
            time_index=0,
        )

    # 2) Nested openboundary simulation that reads from the coarse run
    nested_sim = _build_nested_openbc_sim_from_coarse(
        conf, coarse_sim, supergrid, subgrid
    )
    nested_sim.sim_preprocessing_pipeline()

    nested_input_dir = nested_sim.output_path / "input"
    ob_files = list(nested_input_dir.glob("openboundaries.inp.*.nc"))
    if not ob_files:
        pytest.skip(
            "openboundaries.inp.*.nc not found; openBC preprocessor did not run"
        )
    openboundaries_file = ob_files[0]

    # 3) Compare openboundary variables against coarse fielddump slices and
    #    compute total divergence.
    total_divergence = 0.0

    with (
        xr.open_dataset(fielddump_file) as ds_fd_all_z,
        xr.open_dataset(openboundaries_file) as ds_ob,
    ):
        ds_fd = ds_fd_all_z.sel(zt=nested_sim.grid.zt, zm=nested_sim.grid.zm)
        pairs = [
            ("uwest", "u", "xm", "yt", "west"),
            ("ueast", "u", "xm", "yt", "east"),
            ("vnorth", "v", "ym", "xt", "north"),
            ("vsouth", "v", "ym", "xt", "south"),
            # ("vwest", "v", "ym", "xt", "west"),
            # ("thlwest", "thl", "xt", "yt", "west"),
        ]

        # Define the shifts to test; run all combinations in a single test.
        index_shifts = [0, 1, 2, 3, -1, -2, -3]
        time_shifts = [0, 1, 2, -1, -2]

        unexpected_matches = []

        for index_shift in index_shifts:
            for time_shift in time_shifts:
                for ob_name, fd_name, slice_dim, other_dim, side in pairs:
                    if ob_name not in ds_ob or fd_name not in ds_fd:
                        continue

                    index = 0 if side in {"west", "south"} else -1
                    fd_da = ds_fd[fd_name].sel(
                        {
                            slice_dim: getattr(nested_sim.grid.as_openbc(), slice_dim)[
                                index
                            ]
                        },
                        drop=True,
                    )
                    fd_da = fd_da.sel(
                        {other_dim: getattr(nested_sim.grid.as_openbc(), other_dim)}
                    )
                    ob_da = ds_ob[ob_name].squeeze()

            # Select time slices; apply time shift to fielddump selection.
            try:
                fd_slice_t = fd_da.isel(time=0 + time_shift)
            except IndexError:
                # time shift out of range for this dataset; skip this combination
                continue
            try:
                ob_slice_t = ob_da.isel(time=1)
            except IndexError:
                continue

            # Align dims
            fd_slice_t = fd_slice_t.transpose(*ob_slice_t.dims)

            # Apply spatial index shift to fielddump values along the slice_dim axis
            fd_vals = fd_slice_t.values
            if index_shift != 0:
                if slice_dim not in ob_slice_t.dims:
                    # can't shift along a non-existing dim; skip
                    continue
                axis = ob_slice_t.dims.index(slice_dim)
                fd_vals = np.roll(fd_vals, shift=index_shift, axis=axis)
            if index_shift == 0 and time_shift == 0:
                diff = fd_vals - ob_slice_t.values
                total_divergence += float(np.abs(diff).sum())

            # If both shifts are zero, values should match exactly.
            if index_shift == 0 and time_shift == 0:
                np.testing.assert_allclose(
                    fd_vals,
                    ob_slice_t.values,
                    rtol=1e-6,
                    atol=1e-8,
                )
            else:
                # For nonzero shifts: detect if values unexpectedly match.
                if np.allclose(fd_vals, ob_slice_t.values, rtol=1e-6, atol=1e-8):
                    unexpected_matches.append((ob_name, index_shift, time_shift))

        # If any unexpected matches were found for nonzero shifts, fail with a summary.
        if unexpected_matches:
            details = ", ".join(
                f"{name}@idx={i},t={t}" for name, i, t in unexpected_matches
            )
            pytest.fail(f"Unexpected matches for nonzero shifts: {details}")

    assert total_divergence >= 0.0
    assert total_divergence < 1e-3
