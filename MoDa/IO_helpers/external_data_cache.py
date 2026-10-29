"""Generic helpers for caching generated and downloaded data."""

from __future__ import annotations

import logging
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING

from moda.helpers import _machine_conf_value

if TYPE_CHECKING:
    from moda.modular.dales_simulation import dales_simulation

logger = logging.getLogger(__name__)

CASE_ATTRIBUTION_DIRNAME = "external_data_attribution"


def cache_root(cache_name: str, sim: dales_simulation | None = None) -> pathlib.Path:
    """Return the cache directory for a named helper cache."""
    machine_root = _machine_conf_value(
        sim,
        "external_data_cache_root",
        "external_data_cache_dir",
        "cache_root",
        "cache_dir",
        default=None,
    )
    if machine_root not in (None, ""):
        root = Path(machine_root).expanduser()
    else:
        base_root = _machine_conf_value(sim, "COG_CACHE", "cog_cache", default=None)
        if base_root in (None, ""):
            base_root = pathlib.Path.cwd() / "COG_CACHE"
        root = Path(base_root).expanduser()
    root = root / cache_name
    root.mkdir(parents=True, exist_ok=True)
    return root
