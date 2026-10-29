"""Shared helpers for moda."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from moda import dales_simulation


@contextmanager
def package_resource_path(*parts: str) -> Iterator[Path]:
    """Yield a filesystem path for a resource bundled with moda."""
    resource = resources.files("moda").joinpath("resources", *parts)
    with resources.as_file(resource) as path:
        yield path


def _machine_conf_value(
    sim: dales_simulation | None, *keys: str, default: Any = None
) -> Any:
    if sim is None:
        return default
    machine_conf = getattr(sim, "machine_conf", None) or {}
    for key in keys:
        if key in machine_conf:
            return machine_conf[key]
    case_conf = (
        machine_conf.get("case_conf", {}) if isinstance(machine_conf, dict) else {}
    )
    for key in keys:
        if key in case_conf:
            return case_conf[key]
    return default
