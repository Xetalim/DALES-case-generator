"""Standalone homogeneous LSM module."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from moda.Core.namelist import nml_field
from moda.module_registry import register_module

from .base import BaseLSMModule


@register_module
@dataclass
class LSMHomogeneousModule(BaseLSMModule):
    """Standalone homogeneous LSM configuration module."""

    c_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="c_low",
        description="Tile fraction for low vegetation.",
        metadata={"serialize": True},
    )
    c_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="c_high",
        description="Tile fraction for high vegetation.",
        metadata={"serialize": True},
    )
    c_bare: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="c_bare",
        description="Tile fraction for bare soil.",
        metadata={"serialize": True},
    )
    c_water: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="c_water",
        description="Tile fraction for open water.",
        metadata={"serialize": True},
    )
    c_asph: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="c_asph",
        description="Tile fraction for asphalt/impervious surface.",
        metadata={"serialize": True},
    )
    z0m_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0m_low",
        description="Momentum roughness length for low vegetation in m.",
        metadata={"serialize": True},
    )
    z0m_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0m_high",
        description="Momentum roughness length for high vegetation in m.",
        metadata={"serialize": True},
    )
    z0m_bare: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0m_bare",
        description="Momentum roughness length for bare soil in m.",
        metadata={"serialize": True},
    )
    z0m_water: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0m_water",
        description="Momentum roughness length for water in m.",
        metadata={"serialize": True},
    )
    z0m_asph: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0m_asph",
        description="Momentum roughness length for asphalt in m.",
        metadata={"serialize": True},
    )
    z0h_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0h_low",
        description="Scalar roughness length for low vegetation in m.",
        metadata={"serialize": True},
    )
    z0h_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0h_high",
        description="Scalar roughness length for high vegetation in m.",
        metadata={"serialize": True},
    )
    z0h_bare: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0h_bare",
        description="Scalar roughness length for bare soil in m.",
        metadata={"serialize": True},
    )
    z0h_water: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0h_water",
        description="Scalar roughness length for water in m.",
        metadata={"serialize": True},
    )
    z0h_asph: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="z0h_asph",
        description="Scalar roughness length for asphalt in m.",
        metadata={"serialize": True},
    )
    lambda_s_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_s_low",
        description="Stable transfer coefficient for low vegetation.",
        metadata={"serialize": True},
    )
    lambda_s_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_s_high",
        description="Stable transfer coefficient for high vegetation.",
        metadata={"serialize": True},
    )
    lambda_s_bare: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_s_bare",
        description="Stable transfer coefficient for bare soil.",
        metadata={"serialize": True},
    )
    lambda_s_asph: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_s_asph",
        description="Stable transfer coefficient for asphalt.",
        metadata={"serialize": True},
    )
    lambda_us_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_us_low",
        description="Unstable transfer coefficient for low vegetation.",
        metadata={"serialize": True},
    )
    lambda_us_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_us_high",
        description="Unstable transfer coefficient for high vegetation.",
        metadata={"serialize": True},
    )
    lambda_us_bare: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_us_bare",
        description="Unstable transfer coefficient for bare soil.",
        metadata={"serialize": True},
    )
    lambda_us_asph: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lambda_us_asph",
        description="Unstable transfer coefficient for asphalt.",
        metadata={"serialize": True},
    )
    lai_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lai_low",
        description="Leaf area index for low vegetation.",
        metadata={"serialize": True},
    )
    lai_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="lai_high",
        description="Leaf area index for high vegetation.",
        metadata={"serialize": True},
    )
    rs_min_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="rs_min_low",
        description="Minimum stomatal resistance for low vegetation in s/m.",
        metadata={"serialize": True},
    )
    rs_min_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="rs_min_high",
        description="Minimum stomatal resistance for high vegetation in s/m.",
        metadata={"serialize": True},
    )
    rs_min_bare: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="rs_min_bare",
        description="Minimum surface resistance for bare soil in s/m.",
        metadata={"serialize": True},
    )
    rs_min_asph: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="rs_min_asph",
        description="Minimum surface resistance for asphalt in s/m.",
        metadata={"serialize": True},
    )
    t_soil_p: list[float] | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="t_soil_p",
        description="Initial homogeneous soil temperature profile in K.",
        metadata={"serialize": True},
    )
    theta_soil_p: list[float] | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="theta_soil_p",
        description="Initial homogeneous soil moisture profile in m3/m3.",
        metadata={"serialize": True},
    )
    soil_index_p: list[int] | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="soil_index_p",
        description="Initial homogeneous soil type index profile.",
        metadata={"serialize": True},
    )
    ar_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="ar_low",
        description="Root profile parameter a for low vegetation.",
        metadata={"serialize": True},
    )
    br_low: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="br_low",
        description="Root profile parameter b for low vegetation.",
        metadata={"serialize": True},
    )
    ar_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="ar_high",
        description="Root profile parameter a for high vegetation.",
        metadata={"serialize": True},
    )
    br_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="br_high",
        description="Root profile parameter b for high vegetation.",
        metadata={"serialize": True},
    )
    gD_high: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="gD_high",
        description="Vapour pressure deficit response coefficient for high vegetation.",
        metadata={"serialize": True},
    )
    tskin_water: float | None = nml_field(
        default=None,
        section="NAMLSM_HOMOGENEOUS",
        key="tskin_water",
        description="Skin temperature assigned to water tile in K.",
        metadata={"serialize": True},
    )
    thls: float | None = nml_field(
        default=None,
        section="NAMSURFACE",
        key="thls",
        required=True,
        description="Surface potential temperature in K.",
        metadata={"serialize": True, "forcing_var": "thls"},
    )

    lheterogeneous: bool = nml_field(
        default=False,
        section="NAMLSM",
        key="lheterogeneous",
        required=True,
        description="Use homogeneous LSM mode.",
        metadata={"serialize": True},
    )

    module_name: str = field(
        default="LSMHomogeneousModule", init=False, metadata={"serialize": False}
    )

    nlu: int = nml_field(
        default=6,
        init=False,
        repr=False,
        section="NAMLSM",
        key="nlu",
        metadata={"serialize": False},
    )

    def do_config(self) -> None:
        # Homogeneous LSM always uses the six fixed land-use tiles.
        self.nlu = 6

        # In homogeneous mode, every serialized setting must be explicitly provided.
        missing_fields = [
            f.name
            for f in fields(self)
            if f.init
            and f.metadata.get("serialize", False)
            and getattr(self, f.name) is None
        ]
        if missing_fields:
            raise ValueError(
                f"{self.__class__.__name__} requires all fields to be filled. Missing: {', '.join(missing_fields)}"
            )

        super().do_config()

    def prepare_calculation(self) -> None:
        for name in ("t_soil_p", "theta_soil_p", "soil_index_p"):
            values = getattr(self, name)
            if values is not None and len(values) != int(self.kmax_soil):
                raise ValueError(
                    f"{self.__class__.__name__}.{name} length ({len(values)}) must match kmax_soil ({self.kmax_soil})."
                )
