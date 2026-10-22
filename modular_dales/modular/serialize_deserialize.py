"""Convert simulation dataclasses to YAML-ready values and reconstruct typed inputs."""

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import (
    SINGLETON_REGISTRY,
    SPECIAL_SERIALIZING_REGISTRY,
)
from modular_dales.vars import VariableDefinition, get_var_by_name


def asdict_user_set(value: Any) -> Any:
    """Serialize configuration values, omitting defaults from ordinary dataclasses."""
    if isinstance(value, Enum):
        return asdict_user_set(value.value)
    if isinstance(value, VariableDefinition):
        return value.name
    if type(value).__name__ in SINGLETON_REGISTRY:
        return True
    if is_dataclass(value):
        return _serialize_dataclass(value)
    if isinstance(value, dict):
        return {key: asdict_user_set(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [asdict_user_set(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _serialize_dataclass(instance: Any) -> dict:
    tagged = isinstance(instance, tuple(SPECIAL_SERIALIZING_REGISTRY.values()))
    data = {}
    for definition in fields(instance):
        if definition.name in {"sim", "_sim"}:
            continue
        if not definition.metadata.get("serialize", True):
            continue
        if tagged and not definition.init:
            continue
        value = getattr(instance, definition.name)
        if not tagged and (
            definition.name in {"sim", "exp_id", "grid"}
            or value is None
            or (definition.default is not MISSING and definition.default == value)
        ):
            continue
        data[definition.name] = asdict_user_set(value)
    return {type(instance).__name__: data} if tagged else data


def _registered_dataclass(value: Any) -> type | None:
    """Recognize the existing single-key registry tag format."""
    if not isinstance(value, dict) or len(value) != 1:
        return None
    candidate = SPECIAL_SERIALIZING_REGISTRY.get(next(iter(value)))
    return candidate if candidate is not None and is_dataclass(candidate) else None


def _matches_dataclass(candidate: Any, value: Any) -> bool:
    if not isinstance(value, dict) or not is_dataclass(candidate):
        return False
    return all(
        definition.name in value
        for definition in fields(candidate)
        if definition.init
        and definition.name != "sim"
        and definition.default is MISSING
        and definition.default_factory is MISSING
    )


def _deserialize_union(candidates: tuple, value: Any, *, sim=None) -> Any:
    candidates = tuple(
        candidate for candidate in candidates if candidate is not type(None)
    )
    if len(candidates) == 1:
        return _deserialize_value(candidates[0], value, sim=sim)
    for candidate in candidates:
        if get_origin(candidate) in (list, tuple) and isinstance(value, (list, tuple)):
            return _deserialize_value(candidate, value, sim=sim)
        if _matches_dataclass(candidate, value):
            return _deserialize_dataclass(candidate, value, sim=sim)
    return _deserialize_container(Any, value, sim=sim)


def _deserialize_container(field_type: Any, value: Any, *, sim=None) -> Any:
    arguments = get_args(field_type)
    if isinstance(value, dict):
        item_type = (
            arguments[1] if get_origin(field_type) is dict and arguments else Any
        )
        return {
            key: _deserialize_value(item_type, item, sim=sim)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        item_type = arguments[0] if arguments else Any
        return [_deserialize_value(item_type, item, sim=sim) for item in value]
    return value


def _deserialize_value(field_type: Any, value: Any, *, sim=None) -> Any:
    """Dispatch one value by registry identity, declared type, or container shape."""
    if value is None:
        return None
    if field_type is VariableDefinition:
        return get_var_by_name()[value]
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)
    singleton = SINGLETON_REGISTRY.get(getattr(field_type, "__name__", None))
    if singleton is not None:
        return singleton()
    registered = _registered_dataclass(value)
    if registered is not None:
        return _deserialize_dataclass(registered, next(iter(value.values())), sim=sim)
    if get_origin(field_type) in (Union, UnionType):
        return _deserialize_union(get_args(field_type), value, sim=sim)
    if is_dataclass(field_type) and isinstance(value, dict):
        return _deserialize_dataclass(field_type, value, sim=sim)
    return _deserialize_container(field_type, value, sim=sim)


def _deserialize_dataclass(cls: Any, data: dict, *, sim=None) -> Any:
    """Construct a dataclass using only supplied constructor fields."""
    kwargs = {
        definition.name: _deserialize_value(
            definition.type, data[definition.name], sim=sim
        )
        for definition in fields(cls)
        if definition.init and definition.name in data and definition.name != "sim"
    }
    instance = cls(**kwargs)
    if isinstance(instance, simulation_module) and sim is not None:
        instance._attach_to_sim(sim)
    return instance
