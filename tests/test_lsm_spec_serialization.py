import pytest
import yaml

from modular_dales.modular.serialize_deserialize import (
    _deserialize_dataclass,
    asdict_user_set,
)
from modular_dales.Surface.LSM import (
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


@pytest.mark.parametrize(
    "field_name, specification",
    [
        ("skin_temperature", UniformSkinTemperature(290.0)),
        ("skin_temperature", VaryingSkinTemperature([[290.0, 291.0]])),
        (
            "skin_temperature",
            SoilTemperatureMoistureFromHarmonie(
                harmonie_soil_file="soil.nc",
                use_as_tskin=True,
                harmonie_soil_valid_time="2023-01-01T12:00:00",
                harmonie_soil_height_levels=[0.1, 0.2, 0.3, 0.4],
            ),
        ),
        ("soil_temperature", UniformSoilTemperature([289.0, 288.0])),
        ("soil_temperature", VaryingSoilTemperature([[[289.0, 288.0]]])),
        (
            "soil_temperature",
            SoilTemperatureMoistureFromHarmonie(
                harmonie_soil_file="soil.nc",
                harmonie_soil_valid_time="2023-01-01T12:00:00",
                harmonie_soil_height_levels=[0.1, 0.2, 0.3, 0.4],
            ),
        ),
        ("soil_moisture", UniformSoilMoisture([0.2, 0.3])),
        ("soil_moisture", VaryingSoilMoisture([[[0.2, 0.3]]])),
        (
            "soil_moisture",
            SoilTemperatureMoistureFromHarmonie(
                harmonie_soil_file="soil.nc",
                harmonie_soil_valid_time="2023-01-01T12:00:00",
                harmonie_soil_height_levels=[0.1, 0.2, 0.3, 0.4],
            ),
        ),
    ],
)
def test_lsm_spec_roundtrip(field_name, specification):
    original = LSMModule(**{field_name: specification})
    serialized = asdict_user_set(original)
    assert list(serialized[field_name]) == [type(specification).__name__]
    restored = _deserialize_dataclass(LSMModule, serialized)
    assert type(getattr(restored, field_name)) is type(specification)
    assert getattr(restored, field_name) == specification


def test_explicit_land_use_set_roundtrips_through_yaml():
    original = LSMModule(land_use_set=LandUseSet.DEPAC)
    serialized = asdict_user_set(original)
    assert type(serialized["land_use_set"]) is str
    restored = _deserialize_dataclass(
        LSMModule, yaml.safe_load(yaml.safe_dump(serialized))
    )
    assert restored.land_use_set is LandUseSet.DEPAC
