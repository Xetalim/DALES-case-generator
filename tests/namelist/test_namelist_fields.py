from dataclasses import FrozenInstanceError, dataclass, field, fields

import pytest

from MoDa.modular.dales_simulation import dales_simulation
from MoDa.modular.namelist import NamelistSpec, get_namelist_spec, nml_field
from MoDa.modular.simulation_module import simulation_module


@dataclass
class NamelistModule(simulation_module):
    value: float | None = nml_field(section="TEST", default=2.0)

    def do_config(self):
        pass

    def prepare_calculation(self):
        pass

    def check_settings(self):
        pass

    def write_files(self):
        pass


def attached(module):
    module._attach_to_sim(dales_simulation("test", {}))
    return module


def test_spec_and_dataclass_options():
    @dataclass
    class Settings:
        values: list = nml_field(
            section="TEST",
            key="array",
            default_factory=list,
            description="Test values",
            units="m",
            metadata={"serialize": False, "forcing_var": "psurf"},
            repr=False,
            kw_only=True,
        )

    definition = fields(Settings)[0]
    spec = definition.metadata["namelist"]
    assert spec == NamelistSpec("TEST", "array", "Test values", "m")
    assert definition.metadata["serialize"] is False
    assert definition.metadata["forcing_var"] == "psurf"
    assert definition.kw_only and not definition.repr
    assert Settings().values is not Settings().values
    with pytest.raises(FrozenInstanceError):
        spec.key = "other"


def test_default_key_and_optional_value():
    module = attached(NamelistModule())
    module.apply_namelist_from_fields()
    assert module.nml == {"test": {"value": 2.0}}
    module = attached(NamelistModule(value=None))
    module.apply_namelist_from_fields()
    assert module.nml == {}


def test_required_and_conflict():
    @dataclass
    class Required(NamelistModule):
        value: float | None = nml_field(
            section="TEST",
            key="value",
            default=None,
            required=True,
            raise_conflict=True,
        )

    module = attached(Required())
    with pytest.raises(ValueError, match="requires 'value'"):
        module.apply_namelist_from_fields()
    module.value = 3.0
    module.apply_namelist_from_fields()
    module.value = 4.0
    with pytest.raises(ValueError, match="Conflict detected"):
        module.apply_namelist_from_fields()


@pytest.mark.parametrize(
    "value,expected", [(3, [3, 3]), ([4, 5], [4, 5]), ({"A:x": 6, "B:y": 7}, [6, 7])]
)
def test_multiple_targets(value, expected):
    @dataclass
    class Multiple(NamelistModule):
        value: object = nml_field(section=["A", "B"], key=["x", "y"], default=None)

    module = attached(Multiple(value=value))
    module.apply_namelist_from_fields()
    assert module.nml == {"a": {"x": expected[0]}, "b": {"y": expected[1]}}


def test_single_target_array_is_not_split():
    module = attached(NamelistModule(value=[1, 2]))
    module.apply_namelist_from_fields()
    assert module.nml == {"test": {"value": [1, 2]}}


@pytest.mark.parametrize(
    "metadata",
    [
        {"nml": "TEST"},
        {"key": "value"},
        {"nml": "TEST", "namelist": NamelistSpec("TEST")},
    ],
)
def test_obsolete_metadata_is_rejected(metadata):
    @dataclass
    class Invalid:
        value: int = field(default=1, metadata=metadata)

    with pytest.raises(TypeError, match="must use nml_field or NamelistSpec"):
        get_namelist_spec(fields(Invalid)[0])


def test_targets_broadcast_and_validate_lengths():
    assert NamelistSpec("A", ("x", "y")).targets("value") == (("A", "x"), ("A", "y"))
    assert NamelistSpec(("A", "B")).targets("value") == (("A", "value"), ("B", "value"))
    with pytest.raises(ValueError, match="equal lengths"):
        NamelistSpec(("A", "B"), ("x",))
