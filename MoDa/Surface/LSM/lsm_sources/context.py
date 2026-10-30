"""Runtime state shared by LSM sources without coupling their implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moda.Surface.LSM.lsm_module import LSMModule
    from moda.Surface.LSM.lsm_output import LSM_output_dales


@dataclass
class LSMSourceContext:
    module: LSMModule
    writer: LSM_output_dales

    def set_aquatic_skin_temperature(self, temperature: float | None = None) -> None:
        temperature = self.module.tskin_laqu if temperature is None else temperature
        if temperature is not None:
            self.writer.set_skin_temperature_laqu(temperature)
