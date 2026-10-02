"""Typed configuration objects for open-boundary preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpenBoundaryConfig:
    """Typed configuration shared by open-boundary preprocessors."""

    e12: float | None = None
    tracernames: list[str] = field(default_factory=list)
    tchunk: int | None = None
    start: str | None = None
    author: str = "author"
    time0: str | None = None
    end: str | None = None

    HARMONIE_ml_glob: str | None = None
    HARMONIE_sfc_glob: str | None = None
    KNMI_ml_glob: str | None = None
    KNMI_sfc_glob: str | None = None
    use_grib: bool | None = None

    w_from_continuity: bool = False
    knmi_ml_var_map: dict[str, str] | None = None
    target_resolution: float | None = None

    filter: dict[str, Any] | None = None
    synturb: dict[str, Any] | None = None
    exnr_file: str | None = None

    lsynturb: bool = False
    outpath_coarse: str | None = None
    outpath_coarse_old: str | None = None
    inpath_coarse: str | None = None
    inpath: str | None = None

    extras: dict[str, Any] = field(default_factory=dict, repr=False)
