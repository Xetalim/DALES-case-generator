from __future__ import annotations

from unittest.mock import Mock, call

import numpy as np
import pytest
import yaml

from moda import DefaultNamelistModule, TimeModule, dales_simulation
from moda.Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    InterpolatedProfile,
    lin,
)
from moda.Geometry.grids import GridDales
from moda.Geometry.masks import (
    AllMask,
    RectangleIdxMask,
)
from moda.IBM import FromAHN, FromGlobalDEM, IBMModifications
from moda.IBM import ibm as ibm_module
from moda.IBM.ibm import IBMModification, IBMModule
from moda.modular.serialize_deserialize import (
    _deserialize_dataclass,
    asdict_user_set,
)
from moda.Radiation.radiation import RadiationModule
from moda.Surface.LSM import LandUseSet
from moda.Surface.LSM.lsm_module import LSMModule
from moda.Surface.LSM.lsm_sources.lu_modifications import LandUseModification
from moda.Surface.LSM.lsm_sources.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
)
from moda.vars import *


def IBM_case(machine_conf: dict) -> dales_simulation:
    """Construct a simulation including an IBM configuration."""
    sim = dales_simulation("IBM", machine_conf)

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
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=0.5, ddz=1e-3)))
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
    sim.add(TimeModule(xday=2, xtime=12.0, xyear=2023, runtime=60))

    sim.add(
        RadiationModule(
            iradiation=2,
        )
    )
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

    ibm = IBMModule(thlibm=293.15, thlroof=293.15)
    ibm.add(IBMModification(mask=AllMask(), height=0), priority=0)
    ibm.add(
        IBMModification(
            mask=RectangleIdxMask(minx=4, maxx=6, miny=4, maxy=4),
            height=20,
        ),
        priority=100,
    )
    sim.add(ibm)
    return sim


def assert_ibm_files_written(machine_conf) -> None:
    """Test that IBM files are written correctly."""

    sim = IBM_case(machine_conf("ibm_files"))
    sim.sim_preprocessing_pipeline()

    ibm_files = list(sim.output_path.glob("input/ibm.inp_*.nc"))
    assert len(ibm_files) == 1, f"Expected 1 IBM file, found {len(ibm_files)}"
    for ibm_file in ibm_files:
        assert ibm_file.is_file(), f"Expected IBM file {ibm_file} to exist"
        assert ibm_file.stat().st_size > 0, (
            f"Expected IBM file {ibm_file} to be non-empty"
        )


@pytest.fixture
def ibm_sources(machine_conf):
    sim = dales_simulation("ibm_sources", machine_conf("ibm_sources"))
    sim.add(
        GridDales(
            itot=4,
            jtot=4,
            kmax=4,
            xsize=40.0,
            ysize=40.0,
            kmax_soil=4,
            xlat=52.25,
            xlon=5.45,
            dz0=5.0,
        )
    )
    module = IBMModule(thlibm=293.15, thlroof=293.15)
    sim.add(module)
    return module


@pytest.mark.parametrize("priority, expected", [(-10, 10.0), (10, 12.0)])
def test_ibm_modification_priority_relative_to_terrain(
    ibm_sources, monkeypatch, priority, expected
):
    monkeypatch.setattr(
        ibm_module, "get_process_ahn", lambda *args, **kwargs: np.full((4, 4), 10.0)
    )
    ibm_sources.add(
        IBMModification(mask=AllMask(), height=2.0, mode="add"), priority=priority
    )
    ibm_sources.add(FromAHN(), priority=0)

    ibm_sources.prepare_calculation()

    np.testing.assert_array_equal(ibm_sources.ibm_generator.bc_height, expected)


def test_ibm_sources_preserve_equal_priority_order_and_groups(ibm_sources):
    ibm_sources.add(
        [
            IBMModification(mask=AllMask(), height=10.0),
            IBMModifications(
                [
                    IBMModification(mask=AllMask(), height=3.0, mode="add"),
                    IBMModification(mask=AllMask(), height=4.0, mode="subtract"),
                ]
            ),
        ],
        priority=0,
    )
    ibm_sources.prepare_calculation()

    np.testing.assert_array_equal(ibm_sources.ibm_generator.bc_height, 9.0)


@pytest.mark.parametrize("ahn_priority, expected", [(-10, 10.0), (10, 20.0)])
def test_ibm_terrain_order_and_holes(ibm_sources, monkeypatch, ahn_priority, expected):
    ahn = np.full((4, 4), 20.0)
    ahn[0, 0] = 0.0
    monkeypatch.setattr(ibm_module, "get_process_ahn", lambda *args, **kwargs: ahn)
    monkeypatch.setattr(
        ibm_module,
        "get_process_global_dem",
        lambda *args, **kwargs: np.full((4, 4), 10.0),
    )
    ibm_sources.add(FromAHN(), priority=ahn_priority)
    ibm_sources.add(FromGlobalDEM("terrain.tif"), priority=0)

    ibm_sources.prepare_calculation()

    heights = ibm_sources.ibm_generator.bc_height
    assert heights[0, 0] == 10.0
    np.testing.assert_array_equal(heights[1:], expected)


@pytest.mark.parametrize(
    "mode, height, expected", [("replace", 0, 0), ("subtract", 20, 0)]
)
def test_ibm_modifications_can_clear_heights(ibm_sources, mode, height, expected):
    ibm_sources.add(IBMModification(mask=AllMask(), height=10), priority=0)
    ibm_sources.add(
        IBMModification(mask=AllMask(), height=height, mode=mode), priority=100
    )
    ibm_sources.prepare_calculation()
    np.testing.assert_array_equal(ibm_sources.ibm_generator.bc_height, expected)


def test_ibm_rejects_terrain_shape_mismatch(ibm_sources, monkeypatch):
    monkeypatch.setattr(
        ibm_module, "get_process_ahn", lambda *args, **kwargs: np.ones((2, 2))
    )
    ibm_sources.add(FromAHN(), priority=0)
    with pytest.raises(ValueError, match="terrain source shape mismatch"):
        ibm_sources.prepare_calculation()


def test_ibm_without_sources_starts_flat(ibm_sources):
    assert not ibm_sources.has_terrain_source()
    ibm_sources.prepare_calculation()
    np.testing.assert_array_equal(ibm_sources.ibm_generator.bc_height, 0.0)


def test_ibm_sources_roundtrip_through_yaml():
    original = IBMModule(thlibm=293.15, thlroof=293.15)
    original.add(IBMModification(mask=AllMask(), height=2.0, mode="add"), priority=100)
    original.add(
        [
            FromGlobalDEM("terrain.tif", zeroes_buffer=3, subtract_dem_mode=False),
            FromAHN(zeroes_buffer=7, subtract_ahn_mode=False),
        ],
        priority=0,
    )
    original.add(
        IBMModifications([IBMModification(mask=AllMask(), height=0)]), priority=200
    )
    serialized = asdict_user_set(original)
    restored = _deserialize_dataclass(
        IBMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )

    assert restored.sources == original.sources
    assert [entry.priority for entry in restored.sources] == [100, 0, 0, 200]
    assert restored.has_terrain_source()
    assert asdict_user_set(restored) == serialized
    assert not {"ibm_modifications", "from_ahn", "from_global_dem"} & serialized.keys()
    assert not {"ahn_zeroes_buffer", "subtract_ahn_mode"} & serialized.keys()


def test_ibm_terrain_settings_belong_to_each_source(ibm_sources, monkeypatch, tmp_path):
    ahn_loader = Mock(return_value=np.full((4, 4), 10.0))
    dem_loader = Mock(return_value=np.full((4, 4), 20.0))
    monkeypatch.setattr(ibm_module, "get_process_ahn", ahn_loader)
    monkeypatch.setattr(ibm_module, "get_process_global_dem", dem_loader)
    dem_path = tmp_path / "terrain.tif"
    ibm_sources.add(
        [
            FromAHN(zeroes_buffer=2, subtract_ahn_mode=False),
            FromGlobalDEM(
                str(dem_path),
                resampling="nearest",
                zeroes_buffer=8,
                subtract_dem_mode=False,
            ),
            FromAHN(),
            FromGlobalDEM(str(dem_path)),
        ],
        priority=0,
    )

    ibm_sources.prepare_calculation()

    assert ahn_loader.call_args_list == [
        call(ibm_sources.grid, zeroes_buffer=2, subtract_ahn_mode=False),
        call(ibm_sources.grid, zeroes_buffer=5, subtract_ahn_mode=True),
    ]
    assert dem_loader.call_args_list == [
        call(
            ibm_sources.grid,
            dem_path=dem_path,
            zeroes_buffer=8,
            subtract_dem_mode=False,
            resampling_name="nearest",
        ),
        call(
            ibm_sources.grid,
            dem_path=dem_path,
            zeroes_buffer=5,
            subtract_dem_mode=True,
            resampling_name="bilinear",
        ),
    ]


@pytest.mark.parametrize(
    "source, priority, message",
    [(object(), 0, "Unsupported IBM source"), (FromAHN(), "high", "integer")],
)
def test_ibm_rejects_invalid_source_entries(source, priority, message):
    module = IBMModule()
    with pytest.raises(TypeError, match=message):
        module.add(source, priority=priority)
    assert module.sources == []


def test_ibm_validates_sources_before_preparation(ibm_sources):
    ibm_sources.sources.append(FromAHN())
    with pytest.raises(TypeError, match=r"add\(source, priority="):
        ibm_sources.prepare_calculation()


def test_ibm_checks_grouped_modification_modes():
    module = IBMModule(thlibm=293.15, thlroof=293.15)
    module.add(
        IBMModifications([IBMModification(mask=AllMask(), mode="invalid")]), priority=0
    )
    with pytest.raises(ValueError, match="IBMModification.mode"):
        module.check_settings()


def test_ibm_checks_each_dem_source(tmp_path):
    existing = tmp_path / "terrain.tif"
    existing.touch()
    module = IBMModule(thlibm=293.15, thlroof=293.15)
    module.add(
        [FromGlobalDEM(str(existing)), FromGlobalDEM(str(tmp_path / "missing.tif"))],
        priority=0,
    )
    with pytest.raises(FileNotFoundError, match="missing.tif"):
        module.check_settings()
