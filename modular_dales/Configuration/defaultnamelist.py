"""Default configuration modules and base surface module."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from modular_dales.modular.dales_simulation import dales_simulation
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class DefaultNamelistModule(simulation_module):
    """Module that configures default RUN section.

    This module sets up default values for the RUN section only.
    Domain configuration is handled by GridModule.
    Other configuration modules handle their respective sections.

    Args:
        sim: Parent dales_simulation instance
    """

    sim: Optional["dales_simulation"] = field(default=None, repr=False)
    iexpnr: int = field(
        default=1,
        init=True,
        repr=False,
        metadata={"nml": "RUN", "key": "iexpnr", "serialize": True, "required": True},
    )
    iinput: int = field(
        default=2,
        init=True,
        repr=False,
        metadata={"nml": "RUN", "key": "iinput", "serialize": True, "required": True},
    )
    lwarmstart: bool = field(
        default=False,
        init=True,
        repr=False,
        metadata={
            "nml": "RUN",
            "key": "lwarmstart",
            "serialize": True,
            "required": True,
        },
    )
    trestart: int = field(
        default=3600,
        init=True,
        repr=False,
        metadata={
            "nml": "RUN",
            "key": "trestart",
            "serialize": True,
            "required": True,
        },
    )
    startfile: str = field(
        default="initd_latest_x000y000.001",
        init=True,
        repr=False,
        metadata={
            "nml": "RUN",
            "key": "startfile",
            "serialize": True,
            "required": True,
        },
    )
    ladaptive: bool = field(
        default=True,
        init=True,
        repr=False,
        metadata={
            "nml": "RUN",
            "key": "ladaptive",
            "serialize": True,
            "required": True,
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "DefaultNamelistModule"

    def do_config(self):
        """Configure RUN section defaults."""
        # RUN defaults
        self.apply_namelist_from_fields()

    def check_settings(self):
        """Check defaults validity."""
        return

    def prepare_calculation(self):
        """No calculation work needed."""
        return

    def write_files(self):
        """No files to write."""
        return
