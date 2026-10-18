"""Default configuration modules and base surface module."""

import logging
from dataclasses import dataclass

from modular_dales.modular.namelist import nml_field
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

    iexpnr: int = nml_field(
        default=1,
        init=True,
        repr=False,
        section="RUN",
        key="iexpnr",
        required=True,
        metadata={"serialize": True},
    )
    iinput: int = nml_field(
        default=2,
        init=True,
        repr=False,
        section="RUN",
        key="iinput",
        required=True,
        metadata={"serialize": True},
    )
    lwarmstart: bool = nml_field(
        default=False,
        init=True,
        repr=False,
        section="RUN",
        key="lwarmstart",
        required=True,
        metadata={"serialize": True},
    )
    trestart: int = nml_field(
        default=3600,
        init=True,
        repr=False,
        section="RUN",
        key="trestart",
        required=True,
        metadata={"serialize": True},
    )
    startfile: str = nml_field(
        default="initd_latest_x000y000.001",
        init=True,
        repr=False,
        section="RUN",
        key="startfile",
        required=True,
        metadata={"serialize": True},
    )
    ladaptive: bool = nml_field(
        default=True,
        init=True,
        repr=False,
        section="RUN",
        key="ladaptive",
        required=True,
        metadata={"serialize": True},
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
