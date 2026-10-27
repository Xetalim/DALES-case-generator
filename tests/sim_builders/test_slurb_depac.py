from __future__ import annotations

from collections.abc import Callable
from typing import cast
import pytest

import numpy as np
import xarray as xr

from modular_dales import dales_simulation
from modular_dales.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    exp,
    expsinw,
    lin,
    linmlsurf,
)
from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.Configuration.output_modules import EasyOutputModule
from modular_dales.Configuration.run_and_time import TimeModule
from modular_dales.Geometry.geometry_modification import AllGeometry
from modular_dales.Geometry.grids import GridDales
from modular_dales.Radiation.radiation import RadiationModule
from modular_dales.Surface.LSM import (
    LandUseSet,
    LSMModule,
)
from modular_dales.Surface.LSM.lsm_sources import FromLCZ
from modular_dales.Surface.LSM.lu_modifications import LandUseModification
from modular_dales.Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from modular_dales.Surface.LSM.SLuRB.slurb import (
    SLURBModification,
    SLURBModule,
    SLURBVariableModification,
)
from modular_dales.Surface.LSM.translation_tables.landuse_types import (
    get_land_use_definitions,
)
from modular_dales.vars import *


def SLURB_LCZ_case(machine_conf: dict, tileset="IFS") -> dales_simulation:
    """Construct a simulation combining SLURB and LCZ-based LSM configuration."""
    sim = dales_simulation("SLURB_case", machine_conf)

    sim.add(DefaultNamelistModule())

    domain_info = GridDales(
        itot=32,
        jtot=32,
        kmax=40,
        xsize=1600.0,
        ysize=1600.0,
        kmax_soil=4,
        xlat=52.25,
        xlon=5.45,
        x0=85325.531122,
        y0=446137.619111,
        alpha=1.0,
        dz0=10,
        proj4="+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.4171,50.3319,465.5524,1.9342,-1.6677,9.1019,4.0725 +units=m +no_defs +type=crs",
    )
    sim.add(domain_info)

    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=1e-3)))
    atmo.add(AtmosphericProfile(variable=va, shape=lin(surf_val=0, ddz=0)))
    atmo.add(
        AtmosphericProfile(
            variable=thetal,
            shape=linmlsurf(surf_val=293, offset_val=1.25, lapse_rate=1e-3, z_ml=0),
        )
    )
    atmo.add(
        AtmosphericProfile(
            variable=qt, shape=exp(surf_val=0.016, lambda_val=1500, z_ml=0)
        )
    )
    atmo.add(
        AtmosphericProfile(
            variable=wa,
            shape=expsinw(surf_val=4.0e-3, amp=0, H=2500, Hp=5300),
        )
    )
    atmo.add(
        InterpolatedProfile(
            variable=tke,
            z=[0, 4000, 5000],
            points=[1, 1e-8, 1e-8],
        )
    )

    sim.add(atmo)
    lsm = LSMModule(
        ps=100000,
        z0mav=0.0001,
        z0hav=0.0001,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[1.89, 0.72, 0.21, 0.07],
        albedoav=0.22,
        land_use_set=getattr(LandUseSet, tileset),
    )
    lsm.add(UniformSkinTemperature(291), priority=600)
    lsm.add(UniformSoilTemperature([288, 288, 288, 288]), priority=600)
    lsm.add(
        UniformSoilMoisture(
            [
                0.36867549,
                0.25300502,
                0.14997292,
                0.16459982,
            ]
        ),
        priority=600,
    )
    if tileset == "DEPAC":
        lsm.add(LandUseModification(geometry=AllGeometry(), type="grs"), priority=0)
    elif tileset == "IFS":
        lsm.add(LandUseModification(geometry=AllGeometry(), type="sg"), priority=0)
    lsm.add(FromLCZ(), priority=100)
    sim.add(lsm)

    slurb = SLURBModule(deep_soil_temperature=293)
    slurb.add(SLURBModification(
        geometry=AllGeometry(),
        vars=[SLURBVariableModification(varname="albedo_av", value=10)],
    ))
    sim.add(slurb)

    sim.add(RadiationModule(
        iradiation=4,
    ))

    sim.add(EasyOutputModule(output_interval=10))

    time_module = TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60)
    sim.add(time_module)
    return sim


def SLURB_DEPAC_case(machine_conf: dict) -> dales_simulation:
    """Derive DEPAC tile fractions and SLuRB morphology from LCZ data."""
    sim = SLURB_LCZ_case(machine_conf, tileset="DEPAC")
    return sim


def assert_slurb_depac_files_written(machine_conf: Callable[[str], dict]) -> None:
    """Check both output formats and the retained DEPAC/SLuRB tile fractions."""
    sim = SLURB_DEPAC_case(machine_conf("slurb_DEPAC"))
    sim.sim_preprocessing_pipeline()
    lsm = cast(LSMModule, sim.retrieve_module(LSMModule))
    assert any(isinstance(source.source, FromLCZ) for source in lsm.sources)
    writer = lsm.lsm_writer
    assert writer is not None
    assert writer.land_use_set is LandUseSet.DEPAC
    assert writer.LCZ_ds is not None

    slurb_files = list(sim.output_path.glob("input/inslurb.*.nc"))
    lsm_files = list(sim.output_path.glob("input/lsm.inp_*.nc"))
    assert len(slurb_files) == 1, f"Expected 1 SLURB file, found {len(slurb_files)}"
    assert len(lsm_files) == 1, f"Expected 1 LSM file, found {len(lsm_files)}"

    with xr.open_dataset(lsm_files[0]) as dataset:
        assert dataset.sizes["nlu"] == writer.nlu
        assert "cover_sg" not in dataset
        allowed = set(get_land_use_definitions(LandUseSet.DEPAC))
        assert set(writer.land_use) <= allowed
        cover = writer.LCZ_ds["ifs_land_cover"].values
        for name, state in writer.land_use.items():
            np.testing.assert_allclose(
                dataset[f"cover_{name}"], writer.land_use_fields[name].cover
            )
            np.testing.assert_allclose(
                dataset[f"lutype_{name}"], state.definition.ifs_id
            )
        if np.any(cover == 20):
            np.testing.assert_allclose(dataset["cover_urb"], cover == 20)
        np.testing.assert_allclose(dataset["cover_tot"], 1.0)

    with xr.open_dataset(slurb_files[0]) as dataset:
        for name in ("h_bld", "f_bld", "hw_can", "z0_urb"):
            assert name in dataset
            assert np.isfinite(dataset[name].values).all()
