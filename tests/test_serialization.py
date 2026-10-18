from dataclasses import dataclass, field
from pathlib import Path

import pytest

from modular_dales.modular.serialize_deserialize import (
    _deserialize_dataclass,
    _deserialize_value,
    asdict_user_set,
)
from modular_dales.modular.time_dependent_scalars import TimeDependentScalar
from modular_dales.MODULE_REGISTRY import SINGLETON_REGISTRY
from modular_dales.vars import ua


def test_ordinary_dataclass_filters_defaults_and_runtime_fields():
    @dataclass
    class Configuration:
        value: int = 2
        optional: str | None = None
        sim: str = "runtime"
        hidden: int = field(default=3, metadata={"serialize": False})

    assert asdict_user_set(Configuration()) == {}
    assert asdict_user_set(Configuration(value=7)) == {"value": 7}


def test_nested_containers_preserve_registered_values():
    scalar = TimeDependentScalar(times=[0.0, 60.0], values=[2.0, 4.0])
    source = {"nested": ["plain", {"scalar": scalar, "path": Path("input.nc")}]}
    serialized = asdict_user_set(source)
    assert serialized["nested"][1]["path"] == "input.nc"
    restored = _deserialize_value(dict, serialized)
    assert restored["nested"][1]["scalar"] == scalar
    assert asdict_user_set(ua) == "ua"


def test_typed_container_recursively_constructs_dataclasses():
    @dataclass
    class Configuration:
        value: int

    restored = _deserialize_value(
        dict[str, list[Configuration]], {"items": [{"value": 7}]}
    )
    assert restored == {"items": [Configuration(value=7)]}


def test_singleton_marker_roundtrip(monkeypatch):
    @dataclass
    class Configuration:
        pass

    monkeypatch.setitem(SINGLETON_REGISTRY, "Configuration", Configuration)
    assert asdict_user_set(Configuration()) is True
    assert isinstance(_deserialize_value(Configuration | None, True), Configuration)


def test_selected_union_constructor_errors_are_not_swallowed():
    @dataclass
    class Configuration:
        value: int

        def __post_init__(self):
            raise ValueError("Invalid configuration")

    with pytest.raises(ValueError, match="Invalid configuration"):
        _deserialize_value(Configuration | str, {"value": 7})


def test_runtime_fields_are_not_passed_to_constructor():
    @dataclass
    class Configuration:
        value: int
        runtime: int = field(default=0, init=False)

    assert _deserialize_dataclass(
        Configuration, {"value": 7, "runtime": 99}
    ) == Configuration(7)
