from pathlib import Path

import pytest
import yaml

from moda.Geometry.masks import AllMask, ExplicitArrayMask
from moda.modular.serialize_deserialize import (
    _deserialize_dataclass,
    asdict_user_set,
)
from moda.Surface.LSM import LandUseSet, LSMModule
from moda.Surface.LSM.lsm_sources import (
    FromBofek,
    FromLCZ,
    FromLS2D,
    FromNetCDF,
    FromTop10,
    LandUseModification,
    LandUseModifications,
    SlurbModification,
    SlurbModifications,
    SoilTemperatureMoistureFromHarmonie,
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
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
    """Each thermal source type roundtrips through YAML with its priority and type preserved."""
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
    """The selected land-use enum is serialized by value and restored as the same enum member."""
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
        SlurbModification(
            mask=ExplicitArrayMask([[True, False], [False, True]]),
            varname="h_bld",
            value=12.0,
        ),
        SlurbModifications(
            [
                SlurbModification(mask=AllMask(), varname="h_bld", value=12.0),
                SlurbModification(
                    mask=AllMask(),
                    varname="albedo_roof",
                    value=0.2,
                    n_layers=3,
                    dtype="real",
                ),
            ]
        ),
        LandUseModification(
            mask=ExplicitArrayMask([[True, False], [False, True]]),
            type="grs",
            frac=0.4,
            mode="add",
        ),
        LandUseModifications(
            [
                LandUseModification(mask=AllMask(), type="grs", frac=0.4),
                LandUseModification(mask=AllMask(), type="urb", frac=0.2, mode="add"),
            ]
        ),
    ],
)
def test_lsm_sources_roundtrip_through_yaml(source):
    """Supported LSM and SLURB source specifications roundtrip with their fields and types intact."""
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
    """Mixed source groups preserve registration order, priorities, and the typed source list in YAML."""
    original = LSMModule(land_use_set=LandUseSet.DEPAC)
    original.add(UniformSkinTemperature(295.0), priority=200)
    original.add(FromLCZ(), priority=100)
    original.add(
        LandUseModification(mask=AllMask(), type="grs", frac=0.4), priority=150
    )
    original.add(
        SlurbModification(mask=AllMask(), varname="h_bld", value=12.0),
        priority=150,
    )
    original.add(
        [UniformSoilTemperature([281.0] * 4), UniformSoilMoisture([0.2] * 4)],
        priority=50,
    )
    original.add(
        LandUseModifications(
            [LandUseModification(mask=AllMask(), type="urb", frac=0.2)]
        ),
        priority=0,
    )

    serialized = asdict_user_set(original)
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )

    assert "profiles" not in serialized
    assert "land_use_modifications" not in serialized
    assert restored.sources == original.sources
    assert [entry.priority for entry in restored.sources] == [
        200,
        100,
        150,
        150,
        50,
        50,
        0,
    ]
    assert asdict_user_set(restored) == serialized


def test_integrated_slurb_settings_roundtrip_through_yaml():
    """Integrated SLURB settings roundtrip while runtime generator state remains unserialized."""
    original = LSMModule(
        land_use_set=LandUseSet.DEPAC,
        include_slurb=True,
        urban_fraction=0.4,
        urban_roughness_length=0.5,
        building_plan_area_fraction=0.3,
        building_frontal_area_fraction=0.2,
        building_height=12.0,
        window_fraction=0.25,
        street_canyon_aspect_ratio=1.5,
        building_type=2,
        pavement_type=3,
        anisotropic_street_canyons=True,
        street_canyon_orientation=45.0,
        deep_soil_temperature=290.0,
        building_indoor_temperature=295.0,
        shf_external=10.0,
        qsws_external=0.01,
    )
    original.add(FromLCZ(), priority=100)
    original.add(
        SlurbModification(mask=AllMask(), varname="h_bld", value=12.0), priority=200
    )

    serialized = asdict_user_set(original)
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )

    assert restored.include_slurb is True
    assert restored.deep_soil_temperature == 290.0
    assert restored.building_indoor_temperature == 295.0
    assert restored.slb_generator is None
    assert "slb_generator" not in serialized
    assert "slurb_module" not in serialized
    assert asdict_user_set(restored) == serialized
