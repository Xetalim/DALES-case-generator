"""Typed configuration objects for open-boundary preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OpenBoundaryConfig:
    """Typed configuration shared by open-boundary preprocessors."""

    e12: Optional[float] = None
    tracernames: list[str] = field(default_factory=list)
    tchunk: Optional[int] = None
    start: Optional[str] = None
    author: str = "author"
    time0: Optional[str] = None
    end: Optional[str] = None

    HARMONIE_ml_glob: Optional[str] = None
    HARMONIE_sfc_glob: Optional[str] = None
    KNMI_ml_glob: Optional[str] = None
    KNMI_sfc_glob: Optional[str] = None
    use_grib: Optional[bool] = None

    w_from_continuity: bool = False
    knmi_ml_var_map: Optional[dict[str, str]] = None
    target_resolution: Optional[float] = None

    filter: Optional[dict[str, Any]] = None
    synturb: Optional[dict[str, Any]] = None
    exnr_file: Optional[str] = None

    lsynturb: bool = False
    outpath_coarse: Optional[str] = None
    outpath_coarse_old: Optional[str] = None
    inpath_coarse: Optional[str] = None
    inpath: Optional[str] = None

    extras: dict[str, Any] = field(default_factory=dict, repr=False)
