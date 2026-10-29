from __future__ import annotations

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
    InterpolatedProfile,
    lin,
)
from moda.Configuration import EasyOutputModule
from moda.Emission.emission import EmissionModule, EmissionTracer
from moda.Geometry.masks import AllMask
from moda.Radiation.radiation import RadiationModule
from moda.Surface import ConstantSurfaceTemperatureModule
from moda.Surface.LSM import LandUseSet
from moda.Surface.LSM.lsm_module import LSMModule
from moda.Surface.LSM.lsm_sources.lu_modifications import LandUseModification
from moda.Surface.LSM.lsm_sources.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from moda.vars import *
from moda.vars import (
    VariableDefinition,
    get_var_by_name,
    register_var,
)
from tests.helpers import run_command_with_report


def simple_LSM_case(machine_conf: dict) -> dales_simulation:
    """Construct a simple simulation including an LSM configuration."""
    sim = dales_simulation("LSM_case", machine_conf)

    sim.add(DefaultNamelistModule())

    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=20,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=5.0,
    )
    sim.add(domain_info)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0, ddz=0)))
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )

    sim.add(atmo)
    lsm = LSMModule(
        land_use_set=LandUseSet.DEPAC,
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
    )
    lsm.add(UniformSkinTemperature(293), priority=600)
    lsm.add(UniformSoilTemperature([293, 293, 293, 293]), priority=600)
    lsm.add(UniformSoilMoisture([0.3, 0.3, 0.3, 0.3]), priority=600)
    lsm.add(LandUseModification(mask=AllMask(), type="grs"), priority=0)
    sim.add(lsm)

    sim.add(
        RadiationModule(
            iradiation=4,
        )
    )
    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)
    sim.add(time_module)
    return sim


def assert_lsm_files_written(machine_conf) -> None:
    """Test that LSM files are written correctly for simple_LSM_case."""

    sim = simple_LSM_case(machine_conf("lsm_files"))
    sim.sim_preprocessing_pipeline()

    lsm_files = list(sim.output_path.glob("input/lsm.inp_*.nc"))
    assert len(lsm_files) == 1, f"Expected 1 LSM file, found {len(lsm_files)}"
    for lsm_file in lsm_files:
        assert lsm_file.is_file(), f"Expected LSM file {lsm_file} to exist"
        assert lsm_file.stat().st_size > 0, (
            f"Expected LSM file {lsm_file} to be non-empty"
        )


def lsm_emission_ags_co2_case(machine_conf: dict) -> dales_simulation:
    """Construct an LSM + emission case with AGS switch and CO2 open boundaries."""

    sim = dales_simulation("lsm_emission_ags_co2_case", machine_conf)

    sim.add(DefaultNamelistModule())

    if "co2" not in get_var_by_name():
        register_var(
            VariableDefinition(
                name="co2",
                long_name="Carbon Dioxide (CO2)",
                unit="ppm",
                can_be_time_dependent=True,
            )
        )
    co2_var = get_var_by_name()["co2"]

    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=20,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=0,
        y0=0,
        alpha=1.0,
        dz0=5.0,
    )
    sim.add(domain_info)

    # Base state needed by open boundaries.
    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=2.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=w, shape=lin(surf_val=0.0, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=293.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0.004, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=tke, shape=lin(surf_val=0.1, ddz=0.0)))
    atmo.add(AtmosphericProfile(variable=co2_var, shape=lin(surf_val=420.0, ddz=0.0)))
    sim.add(atmo)

    lsm = LSMModule(
        land_use_set=LandUseSet.DEPAC,
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
    )
    lsm.add(UniformSkinTemperature(293), priority=600)
    lsm.add(UniformSoilTemperature([293, 293, 293, 293]), priority=600)
    lsm.add(UniformSoilMoisture([0.3, 0.3, 0.3, 0.3]), priority=600)
    lsm.add(LandUseModification(mask=AllMask(), type="grs"), priority=0)
    sim.add(lsm)

    # AGS switch requested by user.
    if sim.nml.get("namlsm") is None:
        sim.nml["namlsm"] = {}
    sim.nml["namlsm"]["lags"] = True

    # Only tracer input needed: CO2 concentration.
    emis = EmissionModule()
    emis.add(
        EmissionTracer(
            name="co2",
            long_name="Carbon Dioxide (CO2)",
            unit="ppm",
            molar_mass=44.009,
            lemis=True,
        )
    )
    sim.add(emis)

    # Keep surface forcing simple and stable.
    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=293.15,
            z0mav=0.0001,
            z0hav=0.0001,
            ps=100000,
            albedoav=0.22,
        )
    )

    sim.add(TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=600))
    sim.add(RadiationModule(iradiation=4))
    sim.add(EasyOutputModule(output_interval=60))

    return sim


@pytest.mark.slow
def test_lsm_emission_ags_co2_openbc_run(machine_conf, simulation_report) -> None:
    """Run test for AGS+LSM with CO2 tracer/emission.

    The pipeline generates init.001.nc and tracers.001.nc.  backrad.inp.001.nc
    and rrtmg_*.nc are provided via the normal required_files mechanism.
    The generated namoptions.001 is then overwritten with the hand-crafted
    reference namoptions from tests/sim_builders/crossags/.
    """
    CROSSAGS = Path(__file__).parent / "crossags"

    sim = lsm_emission_ags_co2_case(machine_conf("lsm_emission_ags_co2"))
    sim.sim_preprocessing_pipeline()

    input_dir = sim.output_path / "input"
    assert (input_dir / "tracers.001.nc").is_file(), "Expected tracers.001.nc"

    # Overwrite the generated namoptions with the reference one.
    shutil.copy(
        CROSSAGS / "namoptions.001", sim.output_path / "input" / "namoptions.001"
    )

    run_command_with_report(
        ["./job.001"],
        stage="job_001",
        case_dir=sim.output_path,
        title="lsm_emission_ags_co2 job.001 crash",
        add_report=simulation_report,
    )

    run_dir = sim.output_path / "run_001"
    assert run_dir.is_dir(), "Expected run_001 directory after job.001"
    crossAGS_candidates = list(run_dir.glob("crossAGS*.nc"))
    assert len(crossAGS_candidates) == 1, "Expected crossAGS output in run_001"


@pytest.mark.slow
def test_depcross(tmp_path: Path, machine_conf, simulation_report) -> None:
    """Manual crossdep run: copy folder to tmp and run DALES in-place."""

    source_dir = Path(__file__).parent / "crossdep"
    if not source_dir.is_dir():
        pytest.skip(f"Missing crossdep directory: {source_dir}")

    conf = machine_conf("depcross_manual")
    dales_exec = Path(conf["job_conf"]["dales_exec"])
    if not dales_exec.is_file():
        pytest.skip(f"DALES executable not found: {dales_exec}")

    run_dir = tmp_path / "crossdep"
    shutil.copytree(source_dir, run_dir)

    run_command_with_report(
        [dales_exec.as_posix(), "namoptions.009"],
        stage="depcross_manual",
        case_dir=run_dir,
        title="depcross manual crash",
        add_report=simulation_report,
        info_lines=[f"dales_exec={dales_exec}"],
    )

    assert (run_dir / "depcross.009.nc").is_file(), (
        "Expected depcross.009.nc after manual depcross run"
    )
