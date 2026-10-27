"""Shared base class for LSM surface modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from MoDa.IO_helpers.dales_external_data import resolve_van_genuchten_path
from MoDa.modular.namelist import nml_field
from MoDa.Surface.surface import SurfaceModule

logger = logging.getLogger(__name__)


@dataclass
class BaseLSMModule(SurfaceModule):
    """Base class that owns required LSM namelist options."""

    isurf: int = nml_field(
        default=11,
        init=False,
        section="NAMSURFACE",
        key="isurf",
        required=True,
    )
    ps: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="ps",
        required=True,
        metadata={"serialize": True, "forcing_var": "psurf"},
    )
    z0mav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0mav",
        required=True,
        metadata={"serialize": True},
    )
    z0hav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="z0hav",
        required=True,
        metadata={"serialize": True},
    )
    albedoav: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="albedoav",
        metadata={"serialize": True},
    )
    iinterp_t: int | None = nml_field(
        default=None,
        section="NAMLSM",
        key="iinterp_t",
        required=True,
        metadata={"serialize": True},
    )
    iinterp_theta: int | None = nml_field(
        default=None,
        section="NAMLSM",
        key="iinterp_theta",
        required=True,
        metadata={"serialize": True},
    )
    kmax_soil: int = nml_field(
        default=4,
        section="DOMAIN",
        key="kmax_soil",
        required=True,
        metadata={"serialize": True},
    )
    dz_soil: list[float] = nml_field(
        default_factory=lambda: [1.89, 0.72, 0.21, 0.07],
        section="NAMLSM",
        key="dz_soil",
        required=True,
        metadata={"serialize": True},
    )
    nlu: int = nml_field(
        default=0,
        section="NAMLSM",
        key="nlu",
        metadata={"serialize": False},
        init=False,
        repr=False,
    )

    def do_config(self) -> None:
        for param_name in ("iinterp_t", "iinterp_theta"):
            value = getattr(self, param_name)
            if value is None:
                raise ValueError(
                    f"{self.module_name} requires '{param_name}' parameter"
                )
            if not isinstance(value, int):
                raise TypeError(f"{param_name} must be an integer")
            if not 1 <= value <= 4:
                raise ValueError(
                    f"{param_name} must be an integer between 1 and 4 (1=arithmetic mean, 2=geometric mean, 3=harmonic mean, 4=max)"
                )

        if (self.dz_soil is not None and self.kmax_soil is not None) and (
            len(self.dz_soil) != int(self.kmax_soil)
        ):
            raise ValueError(
                f"{self.module_name}: dz_soil length ({len(self.dz_soil)}) must match kmax_soil ({self.kmax_soil})."
            )

        logger.info("%s: configured base NAMSURFACE/NAMLSM options", self.module_name)

    def check_settings(self) -> None:
        van_genuchten = resolve_van_genuchten_path(self.sim)
        self.sim.required_files["van_genuchten_parameters.nc"] = (
            van_genuchten.as_posix()
        )
