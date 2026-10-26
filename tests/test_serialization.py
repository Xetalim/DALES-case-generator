from dataclasses import dataclass, field
from pathlib import Path

import pytest

from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.Atmosphere.ls2d_atmosphere import FromLS2D
from modular_dales.modular.dales_simulation import dales_simulation
from modular_dales.modular.serialize_deserialize import (
    _deserialize_dataclass,
    _deserialize_value,
    asdict_user_set,
)
from modular_dales.Surface.LSM.lsm_sources import LSMSource
from modular_dales.modular.time_dependent_scalars import TimeDependentScalar
from modular_dales.Surface.surface import ConstantFluxesModule
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


def test_time_dependent_surface_flux_roundtrips_from_quoted_annotation():
    original = ConstantFluxesModule(
        wtsurf=TimeDependentScalar(times=[0.0, 60.0], values=[2.0, 4.0])
    )

    serialized = asdict_user_set(original)
    assert "TimeDependentScalar" in serialized["wtsurf"]
    restored = _deserialize_dataclass(ConstantFluxesModule, serialized)
    assert restored.wtsurf == original.wtsurf


def test_typed_container_recursively_constructs_dataclasses():
    @dataclass
    class Configuration:
        value: int

    restored = _deserialize_value(
        dict[str, list[Configuration]], {"items": [{"value": 7}]}
    )
    assert restored == {"items": [Configuration(value=7)]}


def test_empty_registered_union_member_roundtrips_without_special_decorators():
    original = LSMSource(source=FromLS2D(), priority=100)
    serialized = asdict_user_set(original)
    assert serialized["source"] == {"FromLS2D": {}}
    assert _deserialize_value(LSMSource, serialized) == original


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


def test_module_binding_and_runtime_flags_are_not_serialized():
    module = DefaultNamelistModule()
    configured = asdict_user_set(module)
    sim = dales_simulation("test", {})
    sim += module
    module.do_config_done = True
    module.ensure_prepared()
    assert asdict_user_set(module) == configured
    restored = _deserialize_dataclass(DefaultNamelistModule, configured)
    with pytest.raises(RuntimeError, match="detached"):
        _ = restored.sim
    restored = _deserialize_dataclass(DefaultNamelistModule, configured, sim=sim)
    assert restored.sim is sim
    assert not restored.do_config_done
    assert not restored.prepare_calculation_done
    assert sim.modules == [module]
