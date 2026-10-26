from __future__ import annotations

from modular_dales import dales_simulation
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    lin,
)
from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.Configuration.output_modules import (
    CheckSimulationModule,
    EasyOutputModule,
)
from modular_dales.Configuration.run_and_time import TimeModule
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.Geometry.grids import GridDales
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM import LandUseSet
from modular_dales.Surface.LSM.lsm_module import LSMModule
from modular_dales.Surface.LSM.lu_modifications import LandUseModification
from modular_dales.Surface.LSM.modular_temps_moisture import (
    SoilTemperatureMoistureFromHarmonie,
    UniformSkinTemperature,
)
from modular_dales.vars import *


def LSM_with_HARM_soil(machine_conf: dict) -> dales_simulation:
    """Construct a simple simulation including an LSM configuration."""
    sim = dales_simulation("LSM_case", machine_conf)

    sim += DefaultNamelistModule()
    # 293219, 277925
    domain_info = GridDales(
        itot=64,
        jtot=64,
        kmax=60,
        xsize=1280.0,
        ysize=1280.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=215367,
        y0=214087,
        alpha=1.0,
        dz0=5.0,
        proj4="+proj=lcc +lat_1=48.85 +lat_0=48.85 +lon_0=2.35 +k_0=1 +x_0=246999.973166197 +y_0=246999.953483689 +R=6371229 +units=m +no_defs",
    )
    sim += domain_info

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=thetal, shape=lin(surf_val=305.15, ddz=1e-2)))
    atmo.add(AtmosphericProfile(variable=qt, shape=lin(surf_val=0, ddz=0)))
    atmo.add(AtmosphericProfile(variable=wa, shape=lin(surf_val=0, ddz=0)))
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )

    sim += atmo
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
    lsm.add(UniformSkinTemperature(305), priority=600)
    # A uniform soil source uses lsm.add(UniformSoilTemperature(...), priority=600).
    harmonie_soil_levels = [0.01, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5, 8, 12]

    lsm.add(
        SoilTemperatureMoistureFromHarmonie(
            "/Users/andrevanginkel/Documents/20_Code/21_Input_Output_scripts/21.03_paris_NWP/atmo_interpolated/soil_new.nc",
            harmonie_soil_valid_time="2023-08-20T00:00:00",
            harmonie_soil_height_levels=harmonie_soil_levels,
        ),
        priority=600,
    )
    lsm.add(LandUseModification(geometry=AllGeometry(), type="grs"), priority=0)
    sim += lsm

    sim += RadiationModule(
        iradiation=4,
    )
    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)
    sim += time_module
    sim += CheckSimulationModule(check_interval=10)
    sim += EasyOutputModule(output_interval=5)
    return sim


def assert_lsm_HARM_files_written(machine_conf) -> None:
    """Test that LSM files are written correctly for LSM_with_HARM_soil."""

    sim = LSM_with_HARM_soil(machine_conf("lsm_files"))
    sim.sim_preprocessing_pipeline()

    lsm_files = list(sim.output_path.glob("input/lsm.inp_*.nc"))
    assert len(lsm_files) == 1, f"Expected 1 LSM file, found {len(lsm_files)}"
    for lsm_file in lsm_files:
        assert lsm_file.is_file(), f"Expected LSM file {lsm_file} to exist"
        assert lsm_file.stat().st_size > 0, (
            f"Expected LSM file {lsm_file} to be non-empty"
        )
