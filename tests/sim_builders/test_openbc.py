from pathlib import Path
from functools import partial

import numpy as np
import pytest
import xarray as xr

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    ConstantSurfaceTemperatureModule,
    DefaultNamelistModule,
    GridDales,
    TimeModule,
    dales_simulation,
)
from modular_dales.Configuration.output_modules import EasyOutputModule
from modular_dales.LBC import NestingTopology
from modular_dales.LBC.nest_dales_in_dales.get_all_dales_boundaries import (
    _promote_boundary_stagger_dims,
)
from modular_dales.vars import *  # noqa: F401,F403
from modular_dales.modular.simulation_module import set_nml_section
from tests.openbc_test_input.openbc_fixtures import (
    atmo_netcdf_file,
    surface_netcdf_file,
)
from tests.helpers import run_command_with_report


@pytest.mark.skip(reason="openBCModule not fully implemented yet")
def test_openbc(atmo_netcdf_file, surface_netcdf_file, simulation_report) -> None:
    """Smoke test for OpenBC module using generated NetCDF inputs.

    This is preserved from the original test file and remains skipped until
    openBCModule is fully implemented.
    """

    print("Testing openBCModule with atmospheric profile from netCDF file")

    file1 = atmo_netcdf_file(xlen=50, ylen=50, levlen=21, timesteps=2)
    run_command_with_report(
        ["ncview", str(file1)],
        stage="ncview_atmo",
        case_dir=Path.cwd(),
        title="openbc ncview atmosphere crash",
        add_report=simulation_report,
        info_lines=[f"file={file1}"],
    )
    print(file1)
    file2 = surface_netcdf_file(xlen=50, ylen=50, levlen=21, timesteps=2)
    run_command_with_report(
        ["ncview", str(file2)],
        stage="ncview_surface",
        case_dir=Path.cwd(),
        title="openbc ncview surface crash",
        add_report=simulation_report,
        info_lines=[f"file={file2}"],
    )
    print(file2)


def _build_nested_sim_with_easyoutput(
    machine_conf: dict, casename: str
) -> dales_simulation:
    """Create a small nested DALES sim with EasyOutput and cross-sections.

    This mirrors the structure of the example in oop_test, but is reduced in
    size so it is suitable for use in unit tests. The simulation consists of a
    coarse "supergrid" with a smaller "subgrid" nested inside via
    NestingTopology. EasyOutputModule enables both fielddumps and
    cross-sections so that DALES will write ``fielddump.nc`` and
    cross-section NetCDF files (e.g. ``crossxz.nc`` or ``crossyz.nc``) after
    running ``job.001``.
    """

    sim = dales_simulation(casename, machine_conf)

    # Basic namelist + coarse grid
    sim += DefaultNamelistModule()

    supergrid = GridDales(
        itot=16,
        jtot=16,
        kmax=40,
        xsize=640.0,
        ysize=640.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0.0,
        y0=0.0,
        alpha=1.0,
        dz0=10.0,
    )

    # Subgrid fully contained inside the supergrid domain
    subgrid = GridDales(
        itot=64,
        jtot=64,
        kmax=30,
        xsize=320.0,
        ysize=320.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=160.0,
        y0=160.0,
        alpha=1.0,
        dz0=10.0,
    )

    sim += supergrid

    # Simple atmosphere with linear profiles for basic variables
    atmo = AtmosphereModule()
    atmo += AtmosphericProfile(
        variable=ua,
        shape="lin",
        params=dict(surf_val=3.0, ddz=1e-3),
    )
    atmo += AtmosphericProfile(
        variable=va,
        shape="lin",
        params=dict(surf_val=0.0, ddz=0.0),
    )
    atmo += AtmosphericProfile(
        variable=w,
        shape="lin",
        params=dict(surf_val=0.0, ddz=0.0),
    )
    atmo += AtmosphericProfile(
        variable=thetal,
        shape="lin",
        params=dict(surf_val=293.15, ddz=1e-2),
    )
    atmo += AtmosphericProfile(
        variable=qt,
        shape="lin",
        params=dict(surf_val=0.01, ddz=0.0),
    )

    sim += atmo

    # Short runtime so the produced files stay small
    sim += TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)

    # Simple surface module
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=0.0001,
        z0hav=0.0001,
        ps=100000,
        albedoav=0.22,
    )

    # Nesting topology: supergrid hosting subgrid; this also configures
    # cross-section indices in namcrosssection so DALES writes cross* files.
    nesting = NestingTopology()
    nesting += supergrid
    nesting.my_idx = nesting.nestings.index(supergrid)
    nesting += subgrid
    sim += nesting

    # Enable fielddumps and cross-sections via EasyOutputModule
    sim += EasyOutputModule(
        output_interval=1,
        enable_output=True,
    )

    # Minimal additional namelist tweaks to keep parity with other examples
    if sim.nml.get("namchecksim") is None:
        sim.nml["namchecksim"] = {}
    sim.nml["namchecksim"]["tcheck"] = 5

    if sim.nml.get("thermodynamics") is None:
        sim.nml["thermodynamics"] = {}
    sim.nml["thermodynamics"]["lconstexner"] = True

    return sim


def _assert_crosssection_matches_fielddump(
    fielddump_path: Path,
    crosssection_path: Path,
    var: str = "thl",
    fd_coord_dim: str = "xt",
    cs_coord_dim: str = "xt",
    cross_index: int = 0,
    time_index: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Compare a 2D slice between two DALES NetCDF outputs at a grid index.

    Both NetCDF files are expected to contain a variable ``var`` on a grid
    with coordinate ``fd_coord_dim`` (e.g. ``xt`` or ``yt``) in the first
    dataset and ``cs_coord_dim`` in the second. The function selects the
    coordinate value from the first dataset at ``index`` along
    ``fd_coord_dim``, locates the nearest matching coordinate in the second
    dataset along ``cs_coord_dim``, and asserts the resulting 2D slices are
    numerically equal within the provided tolerances.
    """

    with (
        xr.open_mfdataset(
            str(fielddump_path), combine="by_coords", join="outer"
        ) as ds_ref,
        xr.open_mfdataset(
            str(crosssection_path),
            combine="by_coords",
            join="outer",
            preprocess=partial(
                _promote_boundary_stagger_dims,
                boundary={
                    "xt": "west",
                    "xm": "west",
                    "yt": "south",
                    "ym": "south",
                    "zt": "top",
                    "zm": "top",
                }[cs_coord_dim],
            ),
        ) as ds_other,
    ):
        if var not in ds_ref or var not in ds_other:
            raise AssertionError(f"Variable '{var}' not found in both datasets")

        if fd_coord_dim not in ds_ref.coords:
            raise AssertionError(
                f"Coordinate '{fd_coord_dim}' not found in fielddump coords"
            )
        if cs_coord_dim not in ds_other.coords:
            raise AssertionError(
                f"Coordinate '{cs_coord_dim}' not found in crosssection coords"
            )

        coord_vals_ref = ds_ref[fd_coord_dim]
        n_ref = coord_vals_ref.sizes[fd_coord_dim]

        def _get_slices(idx_ref: int):
            """Return aligned slices for a given reference index.

            Uses the reference dataset coordinate at ``idx_ref`` to find the
            nearest matching coordinate in the other dataset and returns the
            corresponding 2D slices at ``time_index``.
            """

            coord_val = coord_vals_ref.isel({fd_coord_dim: idx_ref}).item()
            coord_vals_other = ds_other[cs_coord_dim].values
            other_index = int(np.abs(coord_vals_other - coord_val).argmin())

            field_slice_local = ds_ref[var].isel(
                {"time": time_index, fd_coord_dim: idx_ref}
            )
            cross_slice_local = ds_other[var].isel(
                {"time": time_index, cs_coord_dim: other_index}
            )
            return field_slice_local, cross_slice_local, other_index

        coord_value = ds_other.coords[cs_coord_dim].values[cross_index]
        index = list(ds_ref[fd_coord_dim].values).index(coord_value)

        # First try the requested index directly.
        base_field_slice, base_cross_slice, base_other_index = _get_slices(index)

        if np.allclose(
            base_field_slice.values, base_cross_slice.values, rtol=rtol, atol=atol
        ):
            # Slices match at the requested index: test passes.
            return
        else:
            # If that fails, try small shifts in the reference index to detect
            # a possible off-by-N error between fielddump and cross-section.
            max_shift = 50
            for shift in range(-max_shift, max_shift + 1):
                if shift == 0:
                    continue
                idx_shift = index + shift
                if idx_shift < 0 or idx_shift >= n_ref:
                    continue

                field_slice_s, cross_slice_s, other_idx_s = _get_slices(idx_shift)
                if np.allclose(
                    field_slice_s.values,
                    cross_slice_s.values,
                    rtol=rtol,
                    atol=atol,
                ):
                    # Slices only match with a shifted index: report and fail
                    raise AssertionError(
                        f"{fielddump_path.name}/{crosssection_path.name} slices do not match at index "
                        f"{index} along '{fd_coord_dim}', but do match when the "
                        f"fielddump index is shifted by {shift} (to {idx_shift}) "
                        f"and crosssection index {other_idx_s} is used along "
                        f"'{cs_coord_dim}'. This suggests an index offset of "
                        f"{shift} between fielddump and crosssection coordinates."
                    )
                else:
                    print(
                        shift,
                        np.mean(
                            np.abs(field_slice_s.values - cross_slice_s.values) ** 2
                        ),
                    )
                    # This shift did not help; try the next one.
                    continue

            # No shift produced a match: re-raise the original assertion.
            np.testing.assert_allclose(
                base_field_slice.values,
                base_cross_slice.values,
                rtol=rtol,
                atol=atol,
            )


def _assert_crosssection_matches_fielddump_scalar_slice_coord(
    fielddump_path: Path,
    crosssection_path: Path,
    var: str = "thl",
    fd_coord_dim: str = "xt",
    cs_slice_coord: str = "xt",
    time_index: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Compare a 2D cross-section slice against a fielddump slice at matching grid index.

    For the no-combine file format where the slice coordinate (e.g. ``xt`` in
    a ``crossyz`` file) is a scalar coordinate *variable* on the cross-section
    dataset rather than a dimension of the data variable.  The scalar
    coordinate value is used to locate the nearest matching index in the
    fielddump, and the full 2D cross-section array is compared directly
    (no ``isel`` on the slice coord is needed).
    """

    with (
        xr.open_mfdataset(
            str(fielddump_path), combine="by_coords", join="outer"
        ) as ds_ref,
        xr.open_mfdataset(
            str(crosssection_path), combine="by_coords", join="outer"
        ) as ds_other,
    ):
        if var not in ds_ref or var not in ds_other:
            raise AssertionError(f"Variable '{var}' not found in both datasets")

        if fd_coord_dim not in ds_ref.coords:
            raise AssertionError(
                f"Coordinate '{fd_coord_dim}' not found in fielddump coords"
            )
        if cs_slice_coord not in ds_other.coords:
            raise AssertionError(
                f"Coordinate '{cs_slice_coord}' not found in crosssection coords"
            )

        # Scalar coordinate value encodes the fixed position of this cross-section.
        cs_coord_val = float(ds_other[cs_slice_coord].values)

        # The cross-section data variable has no slice-coord dimension.
        cross_slice = ds_other[var].isel(time=time_index)

        coord_vals_ref = ds_ref[fd_coord_dim].values
        n_ref = len(coord_vals_ref)

        # Find the nearest index in the fielddump.
        index = int(np.abs(coord_vals_ref - cs_coord_val).argmin())
        base_field_slice = ds_ref[var].isel({"time": time_index, fd_coord_dim: index})

        if np.allclose(
            base_field_slice.values, cross_slice.values, rtol=rtol, atol=atol
        ):
            return
        else:
            # Try small shifts to detect an off-by-N error.
            max_shift = 50
            for shift in range(-max_shift, max_shift + 1):
                if shift == 0:
                    continue
                idx_shift = index + shift
                if idx_shift < 0 or idx_shift >= n_ref:
                    continue

                field_slice_s = ds_ref[var].isel(
                    {"time": time_index, fd_coord_dim: idx_shift}
                )
                if np.allclose(
                    field_slice_s.values, cross_slice.values, rtol=rtol, atol=atol
                ):
                    raise AssertionError(
                        f"{fielddump_path.name}/{crosssection_path.name} slices do not match at index "
                        f"{index} (coord={cs_coord_val}) along '{fd_coord_dim}', but do match when the "
                        f"fielddump index is shifted by {shift} (to {idx_shift}). "
                        f"This suggests an index offset of {shift} between fielddump and crosssection coordinates."
                    )
                else:
                    print(
                        shift,
                        np.mean(np.abs(field_slice_s.values - cross_slice.values) ** 2),
                    )
                    continue

            np.testing.assert_allclose(
                base_field_slice.values,
                cross_slice.values,
                rtol=rtol,
                atol=atol,
            )


@pytest.fixture(
    params=[
        pytest.param(1, id="1_cores"),
        pytest.param(2, id="2_cores"),
        pytest.param(4, id="4_cores"),
        pytest.param(8, id="8_cores"),
    ]
)
def core_changer(request):
    return request.param


# @pytest.mark.xfail
def test_crosssection_matches_fielddump_at_grid_index(
    machine_conf, core_changer, simulation_report
) -> None:
    """Ensure cross-section values equal fielddump values at same grid index.

    This is an end-to-end test: it builds a small nested simulation with
    EasyOutputModule, runs ``job.001`` and ``combine.sh``, then compares a
    cross-section NetCDF file against ``fielddump.nc`` at a coordinate index
    taken from the cross-section grid.
    """
    # pytest.skip("")
    conf = machine_conf("openbc_crosssection_fielddump")
    conf["job_conf"]["numcores"] = core_changer
    sim = _build_nested_sim_with_easyoutput(
        conf, casename="openbc_crosssection_fielddump"
    )
    sim.sim_preprocessing_pipeline()
    set_nml_section(
        sim.nml,
        sim.nml_docs,
        "openbc_crosssection_fielddump",
        "namnetcdfstats",
        "lparallel",
        False,
    )
    outdir = sim.output_path

    # Run the DALES job and combine script to generate diagnostic files.
    run_command_with_report(
        ["./job.001"],
        stage="job_001",
        case_dir=outdir,
        title="openbc crosssection fielddump job.001 crash",
        add_report=simulation_report,
        info_lines=[f"numcores={core_changer}"],
    )
    # run_command_with_report(
    #     ["combine.sh", "run_001"],
    #     stage="combine_run_001",
    #     case_dir=outdir,
    #     title="openbc crosssection fielddump combine crash",
    #     add_report=simulation_report,
    #     info_lines=[f"numcores={core_changer}"],
    # )

    fielddump_file = outdir / "run_001" / "fielddump*.nc"
    if not list(fielddump_file.parent.glob(fielddump_file.name)):
        pytest.skip("fielddump.001.nc not found; DALES run did not produce fielddump")

    # Prefer a yz cross-section at fixed x if available, fall back to others.
    candidates = [
        outdir / "run_001" / "crossyz*.nc",
        outdir / "run_001" / "crossxz*.nc",
        outdir / "run_001" / "crossxy*.nc",
    ]
    var = "thl"
    for crosssection_file in candidates:
        if not list(crosssection_file.parent.glob(crosssection_file.name)):
            raise AssertionError(f"No cross-section files match {crosssection_file}")
        # Choose which coordinate dimension is held constant in the cross-section.
        if crosssection_file.name.startswith("crossyz"):
            if var == "u":
                fd_coord_dim = "xm"
                cs_coord_dim = "xm"
            else:
                fd_coord_dim = "xt"
                cs_coord_dim = "xt"
        elif crosssection_file.name.startswith("crossxz"):
            if var == "v":
                fd_coord_dim = "ym"
                cs_coord_dim = "ym"
            else:
                fd_coord_dim = "yt"
                cs_coord_dim = "yt"
        else:  # crossxy.001.nc
            fd_coord_dim = "zt"
            cs_coord_dim = "zt"

        # Use the cross-section grid as reference and compare against fielddump.
        _assert_crosssection_matches_fielddump(
            fielddump_file,
            crosssection_file,
            var=var,
            fd_coord_dim=fd_coord_dim,
            cs_coord_dim=cs_coord_dim,
            cross_index=0,
            time_index=5,
        )


def test_crosssection_matches_fielddump_at_grid_index_no_combine(
    machine_conf, core_changer, simulation_report
) -> None:
    """Ensure cross-section values equal fielddump values at same grid index.

    Variant without combine.sh: assumes files are in outdir/run_001/ with
    naming convention fielddump.001.nc and cross*.crossidx.001.nc etc.

    This is an end-to-end test: it builds a small nested simulation with
    EasyOutputModule, runs ``job.001``, then compares a cross-section NetCDF
    file against ``fielddump.001.nc`` at a coordinate index taken from the
    cross-section grid.
    """

    conf = machine_conf("openbc_crosssection_fielddump_no_combine")
    conf["job_conf"]["numcores"] = core_changer
    sim = _build_nested_sim_with_easyoutput(
        conf, casename="openbc_crosssection_fielddump_no_combine"
    )
    set_nml_section(
        sim.nml,
        sim.nml_docs,
        "openbc_crosssection_fielddump",
        "namnetcdfstats",
        "lparallel",
        True,
    )
    sim.sim_preprocessing_pipeline()

    outdir = sim.output_path
    run_dir = outdir / "run_001"

    # Run the DALES job only; no combine.sh step.
    run_command_with_report(
        ["./job.001"],
        stage="job_001",
        case_dir=outdir,
        title="openbc crosssection fielddump no-combine job.001 crash",
        add_report=simulation_report,
        info_lines=[f"numcores={core_changer}"],
    )

    fielddump_file = run_dir / "fielddump*.nc"
    if not list(run_dir.glob(fielddump_file.name)):
        pytest.skip(
            "fielddump.001.nc not found in run_001/; DALES run did not produce fielddump"
        )

    # Prefer a yz cross-section at fixed x if available, fall back to others.
    # Files are named with pattern: cross{yz|xz|xy}.<index>.001.nc where <index> is from namelist
    cross_patterns = [
        ("crossyz.*.nc", "crossyz.nc"),
        ("crossxz.*.nc", "crossxz.nc"),
        ("crossxy.*.nc", "crossxy.nc"),
    ]

    var = "thl"
    for pattern, canonical_name in cross_patterns:
        matching_files = sorted(run_dir.glob(pattern))
        if not matching_files:
            continue
        section_prefix = ".".join(matching_files[0].name.split(".")[:2])
        crosssection_file = run_dir / f"{section_prefix}.*.nc"

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
            time_index=5,
        )
