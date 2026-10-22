from dataclasses import dataclass, field

from modular_dales.modular.namelist import nml_field
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module


@register_module
@dataclass
class BulkMicrophysicsSettingsModule(simulation_module):
    """Configure bulk microphysics and required tracer count."""

    imicro: int = nml_field(
        default=2,
        section="NAMMICROPHYSICS",
        key="imicro",
        required=True,
    )
    l_sb: bool = nml_field(
        default=True,
        section="NAMMICROPHYSICS",
        key="l_sb",
        required=True,
    )
    nsv: int = nml_field(
        default=2,
        section="RUN",
        key="nsv",
        required=True,
    )

    module_name: str = field(
        default="BulkMicrophysicsSettingsModule",
        init=False,
        metadata={"serialize": False},
    )


@register_module
@dataclass
class TracerSettingsModule(simulation_module):
    """Configure passive/active tracer count in RUN."""

    nsv: int = nml_field(
        default=0,
        section="RUN",
        key="nsv",
        required=True,
    )

    module_name: str = field(
        default="TracerSettingsModule", init=False, metadata={"serialize": False}
    )
