from __future__ import annotations
import numpy as np
import netCDF4

from modular_dales import (
    AtmosphereModule,
    AtmosphericProfile,
    ConstantSurfaceTemperatureModule,
    DefaultNamelistModule,
    GridDales,
    Nest_in_AtmosphereProfiles,
    TimeModule,
    dales_simulation,
    do_openboundary,
)
from modular_dales.Atmosphere import TimedAtmosphereProfile
from modular_dales.Configuration.output_modules import EasyOutputModule
from modular_dales.vars import *  # noqa: F401,F403


def _build_openbc_atmo_sim(
    machine_conf: dict,
    casename: str,
    add_timedep: bool = False,
) -> dales_simulation:
    """Internal helper to construct an open-BC AtmosphereProfiles simulation.

    When ``add_timedep`` is True, time-dependent boundary profiles are
    added so that open boundary conditions contain multiple time slices.
    """

    sim = dales_simulation(casename, machine_conf)

    # Basic namelist + grid + surface so the pipeline can run
    sim += DefaultNamelistModule()

    # Minimal grid: small domain to keep files lightweight
    grid = GridDales(
        itot=16,
        jtot=16,
        kmax=30,
        xsize=640.0,
        ysize=640.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=10.0,
    )
    sim += grid

    atmo = AtmosphereModule()

    # Simple linear profiles for u, v, w, thetal, qt, tke
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
    atmo += AtmosphericProfile(
        variable=tke,
        shape="lin",
        params=dict(surf_val=0.1, ddz=0.0),
    )

    sim += atmo

    # Simple surface so job.001 has basic settings
    sim += ConstantSurfaceTemperatureModule(
        thls=293.15,
        z0mav=0.0001,
        z0hav=0.0001,
        ps=100000,
        albedoav=0.22,
    )

    # Configure time so Atmosphere profiles have a reference start
    sim += TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=600)

    boundary_u = AtmosphericProfile(
        variable=ua,
        shape="lin",
        params=dict(surf_val=3.0, ddz=1e-3),
    )
    boundary_v = AtmosphericProfile(
        variable=va,
        shape="lin",
        params=dict(surf_val=0.0, ddz=0),
    )
    boundary_w = AtmosphericProfile(
        variable=w,
        shape="lin",
        params=dict(surf_val=0.0, ddz=0.0),
    )
    boundary_thl = AtmosphericProfile(
        variable=thetal,
        shape="lin",
        params=dict(surf_val=293.15, ddz=1e-2),
    )
    boundary_qt = AtmosphericProfile(
        variable=qt,
        shape="lin",
        params=dict(surf_val=0.01, ddz=0.0),
    )
    boundary_e12 = AtmosphericProfile(
        variable=tke,
        shape="lin",
        params=dict(surf_val=0.1, ddz=0.0),
    )

    if add_timedep:
        boundary_u = [
            TimedAtmosphereProfile(
                time=0.0,
                profile=AtmosphericProfile(
                    variable=ua,
                    shape="lin",
                    params=dict(surf_val=3.0, ddz=1e-3),
                ),
            ),
            TimedAtmosphereProfile(
                time=3600.0,
                profile=AtmosphericProfile(
                    variable=ua,
                    shape="lin",
                    params=dict(surf_val=0.0, ddz=0),
                ),
            ),
        ]
        boundary_v = [
            TimedAtmosphereProfile(
                time=0.0,
                profile=AtmosphericProfile(
                    variable=va,
                    shape="lin",
                    params=dict(surf_val=0, ddz=0),
                ),
            ),
            TimedAtmosphereProfile(
                time=3600.0,
                profile=AtmosphericProfile(
                    variable=va,
                    shape="lin",
                    params=dict(surf_val=3.0, ddz=1e-3),
                ),
            ),
        ]

    openbc = do_openboundary(
        time0="2025-01-01T00:00:00",
        start="2025-01-01T00:00:00",
        end="2025-01-01T00:10:00",
        e12=0.1,
        dxint=grid.xsize / grid.itot,
        dyint=grid.ysize / grid.jtot,
    )

    openbc += Nest_in_AtmosphereProfiles(
        u=boundary_u,
        v=boundary_v,
        w=boundary_w,
        thl=boundary_thl,
        qt=boundary_qt,
        e12=boundary_e12,
        noise_boundaries=["south", "west"],
        noise_std=0.1,
        noise_seed=42,
        noise_variables=["ua", "va"],
    )
    sim += openbc

    sim += EasyOutputModule(
        output_interval=10,
    )

    return sim


def openbc_atmosphere_profiles_case(machine_conf: dict) -> dales_simulation:
    """Static AtmosphereProfiles-based open boundary conditions."""

    return _build_openbc_atmo_sim(machine_conf, "openbc_atmo_profiles")


def openbc_atmosphere_profiles_timedep_case(
    machine_conf: dict,
) -> dales_simulation:
    """Time-dependent AtmosphereProfiles-based open boundary conditions."""

    return _build_openbc_atmo_sim(
        machine_conf,
        "openbc_atmo_profiles_timedep",
        add_timedep=True,
    )


def assert_openbc_atmosphere_profiles_files_written(machine_conf) -> None:
    """Check that basic open boundary and initfields files are written."""

    sim = openbc_atmosphere_profiles_case(
        machine_conf("openbc_atmosphere_profiles_files")
    )
    sim.sim_preprocessing_pipeline()

    input_dir = sim.output_path / "input"
    assert (input_dir / f"openboundaries.inp.{sim.exp_id:03d}.nc").is_file()
    assert (input_dir / f"initfields.inp.{sim.exp_id:03d}.nc").is_file()


def assert_openbc_atmosphere_profiles_timedep_files_written(machine_conf) -> None:
    """Check that open boundaries contain a time-dependent Atmosphere signal."""

    sim = openbc_atmosphere_profiles_timedep_case(
        machine_conf("openbc_atmosphere_profiles_timedep_files")
    )
    sim.sim_preprocessing_pipeline()

    input_dir = sim.output_path / "input"
    ob_files = list(input_dir.glob("openboundaries.inp.*.nc"))
    assert len(ob_files) == 1, f"Expected 1 openboundaries file, found {len(ob_files)}"

    with netCDF4.Dataset(ob_files[0], "r") as ds:
        assert "time" in ds.dimensions
        # We expect two time slices: t=0 and t=60
        assert len(ds.dimensions["time"]) == 2
        assert "uwest" in ds.variables
        u = ds.variables["uwest"][:]
        # time should be the leading dimension
        assert u.shape[0] == 2
        # Profiles at t=0 and t=60 should differ
        assert not np.allclose(u[0, ...], u[1, ...])
