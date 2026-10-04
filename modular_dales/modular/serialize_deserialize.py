import logging
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from modular_dales.MODULE_REGISTRY import (
    MODULE_REGISTRY,
    SINGLETON_REGISTRY,
    SPECIAL_SERIALIZING_REGISTRY,
)
from modular_dales.vars import VariableDefinition, get_var_by_name

logger = logging.getLogger(__name__)


def _deserialize_special_case(value: Any) -> Any:
    """Helper to deserialize special-case dataclass patterns from {ClassName: {...}}.

    Only handles special-case patterns like {TimeDependentScalar: {...}}.
    Does NOT recurse into plain dicts or lists - returns value unchanged if not a special pattern.
    Returns (value, was_changed) tuple to let caller know if deserialization occurred.
    """
    if isinstance(value, dict):
        # Check if this is a special-case pattern: {ClassName: {...}}
        if (
            all(key in SPECIAL_SERIALIZING_REGISTRY for key in value.keys())
            and len(value.keys()) == 1
        ):
            type_name = next(iter(value.keys()))
            cls = MODULE_REGISTRY.get(type_name)
            if cls is not None and is_dataclass(cls):
                return _deserialize_dataclass(cls, value[type_name]), True
        # Not a special case: return unchanged
        return value, False
    elif isinstance(value, list):
        # Check if any list elements are special cases
        result = []
        any_changed = False
        for item in value:
            deserialized, changed = _deserialize_special_case(item)
            result.append(deserialized)
            any_changed = any_changed or changed
        return result, any_changed
    else:
        # Scalar: return as-is
        return value, False


def switch_dataclass_or_not(value):
    if is_dataclass(value):
        return asdict_user_set(value)
    else:
        return value


def asdict_user_set(obj):
    """
    Recursively convert a dataclass to dict, but only include fields
    that the user set (exclude default values or None).
    """
    # Non-dataclass objects: recurse into lists/dicts that contain dataclasses
    if not is_dataclass(obj):
        if isinstance(obj, dict):
            all_dataclass = all(is_dataclass(v) for v in obj.values())
            any_dataclass = any(is_dataclass(v) for v in obj.values())
            if all_dataclass:
                return {k: asdict_user_set(v) for k, v in obj.items()}
            elif any_dataclass:
                return {k: switch_dataclass_or_not(v) for k, v in obj.items()}
            else:
                return obj
        elif isinstance(obj, list):
            all_dataclass = all(is_dataclass(v) for v in obj)
            any_dataclass = any(is_dataclass(v) for v in obj)
            if all_dataclass:
                return [asdict_user_set(v) for v in obj]
            elif any_dataclass:
                return [switch_dataclass_or_not(v) for v in obj]
            else:
                return obj
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj  # base case for primitives, lists, dicts

    # Dataclass objects
    # Special-case: represent the surface helper dataclasses
    # (Uniform/Varying* for soil/skin) with an explicit type tag, so they
    # can be reconstructed generically:
    #
    #   skin_temperature:
    # type: UniformSkinTemperature
    #     skin_temperature: 260
    #
    #   soil_temperature:
    # type: UniformSoilTemperature
    #     soil_temperature: [...]
    #
    #   soil_moisture:
    # type: UniformSoilMoisture
    #     soil_moisture: [...]
    #
    # The dataclass field names remain the same, we only add type so
    # the deserializer knows which helper class to instantiate.
    if isinstance(obj, tuple(list(SPECIAL_SERIALIZING_REGISTRY.values()))):
        data = asdict(obj)
        # data["type"] = type(obj).__name__
        return {type(obj).__name__: data}
    if isinstance(obj, VariableDefinition):
        return obj.name

    if type(obj).__name__ in SINGLETON_REGISTRY:
        return True  # just indicate presence of the singleton, no config needed

    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        # Skip default / None / not user-set fields
        if f.name in ["sim", "exp_id", "grid"]:
            continue
        if value is None:
            continue
        if f.default == value:
            continue
        # if f.metadata.get("singleton", False):
        #     result[f.name] = True
        #     continue
        if not f.metadata.get("serialize", True):
            continue
        # Recursively convert nested dataclasses
        result[f.name] = asdict_user_set(value)

    return result


def _deserialize_value(field_type: Any, value: Any) -> Any:
    """Recursively deserialize a value based on a field's annotated type.

    This is intentionally kept small and focused:

    - If the field type is a dataclass and the value is a ``dict``,
      we recursively build that dataclass.
    - If the field type is a ``List[T]``-like container, we recurse on
      each element with inner type ``T``.
    - If the field type is ``Optional[T]`` / ``Union[T, None]``, we
      ignore the ``None`` part and recurse on ``T`` when a value is
      present.
    - Otherwise we return the value unchanged.
    """
    if value is None:
        return None

    if field_type is VariableDefinition:
        return get_var_by_name()[value]

    field_type_name = getattr(field_type, "__name__", None)
    if field_type_name in SINGLETON_REGISTRY:
        return SINGLETON_REGISTRY[
            field_type_name
        ]()  # call the singleton factory to get the instance

    def is_union(t: object) -> bool:
        origin = get_origin(t)
        return origin is Union or origin is UnionType

    # Generic "type-tagged" nested dataclass handling. When we see a dict
    # of the form
    #   {"type": "ClassName", ...}
    # we look up ClassName (using MODULE_REGISTRY) and reconstruct that
    # dataclass from the remaining keys.
    if (
        isinstance(value, dict)
        and all(key in SPECIAL_SERIALIZING_REGISTRY for key in value.keys())
        and len(value.keys()) == 1
    ):
        type_name = next(iter(value.keys()))
        cls = MODULE_REGISTRY.get(type_name)
        if cls is not None and is_dataclass(cls):
            inner = value[type_name]
            return _deserialize_dataclass(cls, inner)

    origin = get_origin(field_type)
    args = get_args(field_type)

    if is_union(field_type):
        for subtype in field_type.__args__:
            if get_origin(subtype) in (list, tuple) and isinstance(
                value, (list, tuple)
            ):
                return _deserialize_value(subtype, value)
            if is_dataclass(subtype) and isinstance(value, dict):
                try:
                    return _deserialize_dataclass(subtype, value)
                except Exception:
                    continue
    # Direct dataclass field (e.g. field_type is ``SomeDataclass``)
    if is_dataclass(field_type) and isinstance(value, dict):
        return _deserialize_dataclass(field_type, value)

    # List[...] types, including List[Dataclass]. For something annotated as
    #   List[T]
    # we call ``_deserialize_value(T, ...)`` on every element.
    if origin in (list, tuple) and isinstance(value, (list, tuple)) and args:
        elem_type = args[0]
        result = []
        for v in value:
            # First try to detect special cases
            deserialized_special, changed = _deserialize_special_case(v)
            if changed:
                result.append(deserialized_special)
            else:
                # Otherwise use type-aware deserialization
                result.append(_deserialize_value(elem_type, v))
        return result

    if origin is dict and isinstance(value, dict) and len(args) == 2:
        return {key: _deserialize_value(args[1], item) for key, item in value.items()}

    # Optional[T] / Union[...] handling (including PEP 604 "|" unions).
    # At this point we've already handled list/tuple container types, so any
    # remaining type with ``args`` is treated as a Union-like construct.
    if args and origin not in (list, tuple):
        non_none = [a for a in args if a is not type(None)]

        # Simple Optional[T] case (Union[T, None])
        if len(non_none) == 1:
            return _deserialize_value(non_none[0], value)

        # Union of multiple types, e.g.
        #   Union[UniformSkinTemperature, VaryingSkinTemperature, None]
        # or the PEP 604 form
        #   UniformSkinTemperature | VaryingSkinTemperature | None
        # If the union members are dataclasses and the value is a dict,
        # try to construct each dataclass in order and return the first
        # that succeeds.
        if isinstance(value, dict):
            for candidate in non_none:
                if is_dataclass(candidate):
                    try:
                        return _deserialize_dataclass(candidate, value)
                    except Exception:
                        continue

    # Handle plain dict fields - recurse into dict values to deserialize any nested dataclasses
    if field_type is dict and isinstance(value, dict):
        result = {}
        for k, v in value.items():
            deserialized_special, changed = _deserialize_special_case(v)
            if changed:
                result[k] = deserialized_special
            else:
                result[k] = v
        return result

    return value


def _deserialize_dataclass(cls: Any, data: dict) -> Any:
    """Instantiate a dataclass ``cls`` from a plain ``dict``.

    This helper is responsible for *one level* of object construction:

    - It iterates over all dataclass fields of ``cls``.
    - For every field name that is present in ``data``, it looks up the
        annotated field type and hands both type and raw value to
        ``_deserialize_value``.
    - The result of ``_deserialize_value`` is then passed as a keyword
        argument to the dataclass constructor.

    Because ``_deserialize_value`` itself is recursive, this gives you a
    tree-shaped reconstruction: nested dataclasses, lists of dataclasses,
    and Optional/Union-wrapped fields are all handled transparently as
    long as their type annotations are correct.

    Example sketch:

    >>> @dataclass
    ... class Child:
    ...     x: int
    ...
    >>> @dataclass
    ... class Parent:
    ...     child: Child
    ...     children: List[Child]
    ...
    >>> raw = {"child": {"x": 1}, "children": [{"x": 2}, {"x": 3}]}
    >>> _deserialize_dataclass(Parent, raw)
    Parent(child=Child(x=1), children=[Child(x=2), Child(x=3)])
    """
    kwargs = {}

    # Iterate over all dataclass fields and look up corresponding entries in the raw dict
    for f in fields(cls):
        if f.name not in data:
            continue
        raw_value = data[f.name]
        kwargs[f.name] = _deserialize_value(f.type, raw_value)
    # return the dataclass instance constructed with the deserialized field values
    return cls(**kwargs)
