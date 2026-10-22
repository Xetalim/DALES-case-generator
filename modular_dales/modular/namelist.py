"""Typed namelist bindings for simulation module dataclass fields."""

from collections.abc import Mapping
from dataclasses import MISSING, Field, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NamelistSpec:
    """Describe the namelist entries controlled by a dataclass field."""

    section: str | tuple[str, ...]
    key: str | tuple[str, ...] | None = None
    description: str = ""
    units: str | None = None
    required: bool = False
    raise_conflict: bool = False

    def __post_init__(self) -> None:
        for name in ("section", "key"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        if not self.section:
            raise ValueError("A namelist section is required")
        if (
            isinstance(self.section, tuple)
            and isinstance(self.key, tuple)
            and len(self.section) != len(self.key)
        ):
            raise ValueError("Namelist sections and keys must have equal lengths")

    def targets(self, field_name: str) -> tuple[tuple[str, str], ...]:
        """Resolve the default key and broadcast single sections or keys."""
        key = self.key if self.key is not None else field_name
        sections = self.section if isinstance(self.section, tuple) else (self.section,)
        keys = key if isinstance(key, tuple) else (key,) * len(sections)
        if isinstance(self.section, str):
            sections = sections * len(keys)
        return tuple(zip(sections, keys, strict=True))


def nml_field(
    *,
    section: str | tuple[str, ...] | list[str],
    key: str | tuple[str, ...] | list[str] | None = None,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    description: str = "",
    units: str | None = None,
    required: bool = False,
    raise_conflict: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **field_options: Any,
) -> Any:
    """Create a dataclass field with a typed namelist binding.

    Other metadata (such as ``serialize`` and ``forcing_var``) and standard
    dataclass field options are passed through unchanged.
    """
    return field(
        default=default,
        default_factory=default_factory,
        metadata={
            **(metadata or {}),
            "namelist": NamelistSpec(
                section=tuple(section) if isinstance(section, list) else section,
                key=tuple(key) if isinstance(key, list) else key,
                description=description,
                units=units,
                required=required,
                raise_conflict=raise_conflict,
            ),
        },
        **field_options,
    )


def get_namelist_spec(dataclass_field: Field[Any]) -> NamelistSpec | None:
    """Read a typed binding and reject obsolete namelist metadata."""
    metadata = dataclass_field.metadata
    if "nml" in metadata or "key" in metadata:
        raise TypeError(
            f"Field '{dataclass_field.name}' must use nml_field or NamelistSpec"
        )
    spec = metadata.get("namelist")
    if spec is not None:
        if not isinstance(spec, NamelistSpec):
            raise TypeError(f"Field '{dataclass_field.name}' needs a NamelistSpec")
        return spec
    return None
