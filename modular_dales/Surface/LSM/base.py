"""Shared base class for LSM surface modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from modular_dales.IO_helpers.dales_external_data import resolve_van_genuchten_path
from modular_dales.Surface.surface import SurfaceModule

logger = logging.getLogger(__name__)


@dataclass
class BaseLSMModule(SurfaceModule):
    """Base class that owns required LSM namelist options."""

    sim: simulation_module | None = field(default=None, repr=False)
    isurf: int = field(
        default=11,
        init=False,
        metadata={"nml": "NAMSURFACE", "key": "isurf", "required": True},
    )
    ps: float | None = field(
        default=None,
        metadata={
            "nml": "NAMSURFACE",
            "key": "ps",
            "required": True,
            "serialize": True,
            "forcing_var": "psurf",
        },
    )
    z0mav: float | None = field(
        default=None,
        metadata={
            "nml": "NAMSURFACE",
            "key": "z0mav",
            "required": True,
            "serialize": True,
        },
    )
    z0hav: float | None = field(
        default=None,
        metadata={
            "nml": "NAMSURFACE",
            "key": "z0hav",
            "required": True,
            "serialize": True,
        },
    )
    albedoav: float | None = field(
        default=None,
        metadata={"nml": "NAMSURFACE", "key": "albedoav", "serialize": True},
    )
    iinterp_t: int | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM",
            "key": "iinterp_t",
            "required": True,
            "serialize": True,
        },
    )
    iinterp_theta: int | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM",
            "key": "iinterp_theta",
            "required": True,
            "serialize": True,
        },
    )
    kmax_soil: int | None = field(
        default=4,
        metadata={
            "nml": "DOMAIN",
            "key": "kmax_soil",
            "required": True,
            "serialize": True,
        },
    )
    dz_soil: list[float] | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM",
            "key": "dz_soil",
            "required": True,
            "serialize": True,
        },
    )
    nlu: int = field(
        default=0,
        metadata={"nml": "NAMLSM", "key": "nlu", "serialize": False},
        init=False,
        repr=False,
    )

    def __post_init__(self):
        super().__init__(self.sim)

    def do_config(self):
        for param_name in ("iinterp_t", "iinterp_theta"):
            value = getattr(self, param_name)
            if value is None:
                raise ValueError(
                    f"{self.module_name} requires '{param_name}' parameter"
                )
            if not isinstance(value, int):
                raise ValueError(f"{param_name} must be an integer")
            if not 1 <= value <= 4:
                raise ValueError(
                    f"{param_name} must be an integer between 1 and 4 (1=arithmetic mean, 2=geometric mean, 3=harmonic mean, 4=max)"
                )

        if self.dz_soil is not None and self.kmax_soil is not None:
            if len(self.dz_soil) != int(self.kmax_soil):
                raise ValueError(
                    f"{self.module_name}: dz_soil length ({len(self.dz_soil)}) must match kmax_soil ({self.kmax_soil})."
                )

        logger.info("%s: configured base NAMSURFACE/NAMLSM options", self.module_name)

    def check_settings(self):
        van_genuchten = resolve_van_genuchten_path(self.sim)
        self.sim.required_files["van_genuchten_parameters.nc"] = (
            van_genuchten.as_posix()
        )

    def prepare_calculation(self):
        return None

    def write_files(self):
        return None
