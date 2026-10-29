"""Convert simulation dataclasses to YAML-ready values and reconstruct typed inputs."""

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from moda.modular.simulation_module import simulation_module
from moda.module_registry import module_registry
from moda.vars import VariableDefinition, get_var_by_name


def asdict_user_set(value: Any) -> Any:
    """Serialize configuration values, omitting defaults from ordinary dataclasses."""
    return _serialize_value(value, Any, root=True)


def _serialize_value(value: Any, value_type: Any, *, root: bool = False) -> Any:
    if isinstance(value, Enum):
        return _serialize_value(value.value, Any)
    if isinstance(value, VariableDefinition):
        return value.name
    if is_dataclass(value):
        tagged = not root and module_registry.get(type(value).__name__) is type(value)
        return _serialize_dataclass(value, tagged=tagged)
    if isinstance(value, dict):
        arguments = get_args(value_type)
        item_type = (
            arguments[1]
            if get_origin(value_type) is dict and len(arguments) > 1
            else Any
        )
        return {key: _serialize_value(item, item_type) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        arguments = get_args(value_type)
        item_type = arguments[0] if arguments else Any
        return [_serialize_value(item, item_type) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _serialize_dataclass(instance: Any, *, tagged: bool = False) -> dict[str, Any]:
    data = {}
    for definition in fields(instance):
        if definition.name in {"sim", "_sim"}:
            continue
        if not (definition.metadata.get("serialize", True)):
            continue
        if tagged and not definition.init:
            continue
        if not definition.init:
            continue
        value = getattr(instance, definition.name)
        field_type = definition.type
        if not tagged and (
            definition.name in {"sim", "exp_id", "grid"}
            or value is None
            or (definition.default is not MISSING and definition.default == value)
        ):
            continue
        data[definition.name] = _serialize_value(value, field_type)
    return {type(instance).__name__: data} if tagged else data


def _registered_dataclass(value: Any) -> type | None:
    """Recognize the existing single-key registry tag format."""
    if not isinstance(value, dict) or len(value) != 1:
        return None
    candidate = module_registry.get(next(iter(value)))
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


def _deserialize_union(
    candidates: tuple[Any, ...], value: Any, *, sim: Any = None
) -> Any:
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


def _deserialize_container(field_type: Any, value: Any, *, sim: Any = None) -> Any:
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


def _deserialize_value(field_type: Any, value: Any, *, sim: Any = None) -> Any:
    """Dispatch one value by registry identity, declared type, or container shape."""
    if value is None:
        return None
    if field_type is Path:
        return Path(value)
    if isinstance(field_type, type) and issubclass(field_type, VariableDefinition):
        return get_var_by_name()[value]
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)
    registered = _registered_dataclass(value)
    if registered is not None:
        return _deserialize_dataclass(registered, next(iter(value.values())), sim=sim)
    if get_origin(field_type) in (Union, UnionType):
        return _deserialize_union(get_args(field_type), value, sim=sim)
    if is_dataclass(field_type) and isinstance(value, dict):
        return _deserialize_dataclass(field_type, value, sim=sim)
    return _deserialize_container(field_type, value, sim=sim)


def _deserialize_dataclass(cls: Any, data: dict[str, Any], *, sim: Any = None) -> Any:
    """Construct a dataclass using only supplied constructor fields."""
    kwargs = {
        definition.name: _deserialize_value(
            definition.type,
            data[definition.name],
            sim=sim,
        )
        for definition in fields(cls)
        if definition.init and definition.name in data and definition.name != "sim"
    }
    instance = cls(**kwargs)
    if isinstance(instance, simulation_module) and sim is not None:
        instance._attach_to_sim(sim)
    return instance
