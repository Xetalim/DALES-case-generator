from pathlib import Path

import pytest
import yaml

from modular_dales.Geometry.geometry_modification import MaskGeometry
from modular_dales.modular.serialize_deserialize import (
    _deserialize_dataclass,
    asdict_user_set,
)
from modular_dales.Surface.LSM import (
    FromBofek,
    FromLCZ,
    FromLS2D,
    FromNetCDF,
    FromTop10,
    LandUseSet,
    LSMModule,
    SoilTemperatureMoistureFromHarmonie,
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
)
from modular_dales.Surface.LSM.lu_modifications import (
    LandUseModification,
    LandUseModifications,
)


@pytest.mark.parametrize(
    "specification",
    [
        UniformSkinTemperature(290.0),
        VaryingSkinTemperature([[290.0, 291.0]]),
        SoilTemperatureMoistureFromHarmonie(
            harmonie_soil_file="soil.nc",
            use_as_tskin=True,
            harmonie_soil_valid_time="2023-01-01T12:00:00",
            harmonie_soil_height_levels=[0.1, 0.2, 0.3, 0.4],
        ),
        UniformSoilTemperature([289.0, 288.0]),
        VaryingSoilTemperature([[[289.0, 288.0]]]),
        UniformSoilMoisture([0.2, 0.3]),
        VaryingSoilMoisture([[[0.2, 0.3]]]),
    ],
)
def test_lsm_thermal_source_roundtrip_through_yaml(specification):
    original = LSMModule(land_use_set=LandUseSet.DEPAC)
    original.add(specification, priority=200)
    serialized = asdict_user_set(original)
    assert "profiles" not in serialized
    assert "land_use_modifications" not in serialized
    assert list(serialized["sources"][0]["_PrioritizedSource"]["source"]) == [
        type(specification).__name__
    ]
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )
    assert len(restored.sources) == 1
    assert restored.sources[0].priority == 200
    assert type(restored.sources[0].source) is type(specification)
    assert asdict_user_set(restored) == serialized


def test_explicit_land_use_set_roundtrips_through_yaml():
    original = LSMModule(land_use_set=LandUseSet.DEPAC)
    serialized = asdict_user_set(original)
    assert type(serialized["land_use_set"]) is str
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )
    assert restored.land_use_set is LandUseSet.DEPAC


@pytest.mark.parametrize(
    "source",
    [
        FromLCZ(urban_natural_lcz_to_10=True),
        FromTop10(fill_north_sea=True),
        FromBofek(bofek_filename="custom.nc"),
        FromNetCDF(path="surface.nc", esa_cache_dir="cache"),
        FromLS2D(),
        LandUseModification(
            geometry=MaskGeometry([[True, False], [False, True]]),
            type="grs",
            frac=0.4,
            mode="add",
        ),
        LandUseModifications(
            [
                LandUseModification(type="grs", frac=0.4),
                LandUseModification(type="urb", frac=0.2, mode="add"),
            ]
        ),
    ],
)
def test_lsm_sources_roundtrip_through_yaml(source):
    original = LSMModule(land_use_set=LandUseSet.IFS)
    original.add(source, priority=100)

    serialized = asdict_user_set(original)
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )

    assert len(restored.sources) == 1
    assert restored.sources[0].priority == 100
    restored_source = restored.sources[0].source
    assert type(restored_source) is type(source)
    assert restored_source == source
    assert asdict_user_set(restored) == serialized
    if isinstance(source, (FromTop10, FromBofek)):
        assert isinstance(restored_source.spatial_data_path, Path)


def test_mixed_lsm_sources_preserve_priorities_and_insertion_order_through_yaml():
    original = LSMModule(land_use_set=LandUseSet.DEPAC)
    original.add(UniformSkinTemperature(295.0), priority=200)
    original.add(FromLCZ(), priority=100)
    original.add(LandUseModification(type="grs", frac=0.4), priority=150)
    original.add(
        [UniformSoilTemperature([281.0] * 4), UniformSoilMoisture([0.2] * 4)],
        priority=50,
    )
    original.add(
        LandUseModifications([LandUseModification(type="urb", frac=0.2)]),
        priority=0,
    )

    serialized = asdict_user_set(original)
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )

    assert "profiles" not in serialized
    assert "land_use_modifications" not in serialized
    assert restored.sources == original.sources
    assert [entry.priority for entry in restored.sources] == [200, 100, 150, 50, 50, 0]
    assert asdict_user_set(restored) == serialized
