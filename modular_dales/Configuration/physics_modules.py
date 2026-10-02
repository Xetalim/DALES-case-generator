from dataclasses import dataclass, field
from typing import Optional

from modular_dales.modular.dales_simulation import dales_simulation
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module


@register_module
@dataclass
class BulkMicrophysicsSettingsModule(simulation_module):
    """Configure bulk microphysics and required tracer count."""

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    imicro: int = field(
        default=2,
        metadata={"nml": "NAMMICROPHYSICS", "key": "imicro", "required": True},
    )
    l_sb: bool = field(
        default=True,
        metadata={"nml": "NAMMICROPHYSICS", "key": "l_sb", "required": True},
    )
    nsv: int = field(
        default=2,
        metadata={"nml": "RUN", "key": "nsv", "required": True},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "BulkMicrophysicsSettingsModule"

    def do_config(self):
        return None

    def prepare_calculation(self):
        return None

    def check_settings(self):
        return None

    def write_files(self):
        return None


@register_module
@dataclass
class TracerSettingsModule(simulation_module):
    """Configure passive/active tracer count in RUN."""

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    nsv: int = field(
        default=0,
        metadata={"nml": "RUN", "key": "nsv", "required": True},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "TracerSettingsModule"

    def do_config(self):
        return None

    def prepare_calculation(self):
        return None

    def check_settings(self):
        return None

    def write_files(self):
        return None
