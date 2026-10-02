"""Standalone homogeneous LSM module."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from modular_dales.MODULE_REGISTRY import register_module

from .base import BaseLSMModule


@register_module
@dataclass
class LSMHomogeneousModule(BaseLSMModule):
    """Standalone homogeneous LSM configuration module."""

    c_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "c_low",
            "serialize": True,
            "doc": "Tile fraction for low vegetation.",
        },
    )
    c_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "c_high",
            "serialize": True,
            "doc": "Tile fraction for high vegetation.",
        },
    )
    c_bare: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "c_bare",
            "serialize": True,
            "doc": "Tile fraction for bare soil.",
        },
    )
    c_water: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "c_water",
            "serialize": True,
            "doc": "Tile fraction for open water.",
        },
    )
    c_asph: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "c_asph",
            "serialize": True,
            "doc": "Tile fraction for asphalt/impervious surface.",
        },
    )
    z0m_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0m_low",
            "serialize": True,
            "doc": "Momentum roughness length for low vegetation in m.",
        },
    )
    z0m_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0m_high",
            "serialize": True,
            "doc": "Momentum roughness length for high vegetation in m.",
        },
    )
    z0m_bare: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0m_bare",
            "serialize": True,
            "doc": "Momentum roughness length for bare soil in m.",
        },
    )
    z0m_water: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0m_water",
            "serialize": True,
            "doc": "Momentum roughness length for water in m.",
        },
    )
    z0m_asph: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0m_asph",
            "serialize": True,
            "doc": "Momentum roughness length for asphalt in m.",
        },
    )
    z0h_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0h_low",
            "serialize": True,
            "doc": "Scalar roughness length for low vegetation in m.",
        },
    )
    z0h_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0h_high",
            "serialize": True,
            "doc": "Scalar roughness length for high vegetation in m.",
        },
    )
    z0h_bare: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0h_bare",
            "serialize": True,
            "doc": "Scalar roughness length for bare soil in m.",
        },
    )
    z0h_water: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0h_water",
            "serialize": True,
            "doc": "Scalar roughness length for water in m.",
        },
    )
    z0h_asph: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "z0h_asph",
            "serialize": True,
            "doc": "Scalar roughness length for asphalt in m.",
        },
    )
    lambda_s_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_s_low",
            "serialize": True,
            "doc": "Stable transfer coefficient for low vegetation.",
        },
    )
    lambda_s_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_s_high",
            "serialize": True,
            "doc": "Stable transfer coefficient for high vegetation.",
        },
    )
    lambda_s_bare: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_s_bare",
            "serialize": True,
            "doc": "Stable transfer coefficient for bare soil.",
        },
    )
    lambda_s_asph: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_s_asph",
            "serialize": True,
            "doc": "Stable transfer coefficient for asphalt.",
        },
    )
    lambda_us_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_us_low",
            "serialize": True,
            "doc": "Unstable transfer coefficient for low vegetation.",
        },
    )
    lambda_us_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_us_high",
            "serialize": True,
            "doc": "Unstable transfer coefficient for high vegetation.",
        },
    )
    lambda_us_bare: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_us_bare",
            "serialize": True,
            "doc": "Unstable transfer coefficient for bare soil.",
        },
    )
    lambda_us_asph: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lambda_us_asph",
            "serialize": True,
            "doc": "Unstable transfer coefficient for asphalt.",
        },
    )
    lai_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lai_low",
            "serialize": True,
            "doc": "Leaf area index for low vegetation.",
        },
    )
    lai_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "lai_high",
            "serialize": True,
            "doc": "Leaf area index for high vegetation.",
        },
    )
    rs_min_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "rs_min_low",
            "serialize": True,
            "doc": "Minimum stomatal resistance for low vegetation in s/m.",
        },
    )
    rs_min_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "rs_min_high",
            "serialize": True,
            "doc": "Minimum stomatal resistance for high vegetation in s/m.",
        },
    )
    rs_min_bare: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "rs_min_bare",
            "serialize": True,
            "doc": "Minimum surface resistance for bare soil in s/m.",
        },
    )
    rs_min_asph: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "rs_min_asph",
            "serialize": True,
            "doc": "Minimum surface resistance for asphalt in s/m.",
        },
    )
    t_soil_p: list[float] | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "t_soil_p",
            "serialize": True,
            "doc": "Initial homogeneous soil temperature profile in K.",
        },
    )
    theta_soil_p: list[float] | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "theta_soil_p",
            "serialize": True,
            "doc": "Initial homogeneous soil moisture profile in m3/m3.",
        },
    )
    soil_index_p: list[int] | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "soil_index_p",
            "serialize": True,
            "doc": "Initial homogeneous soil type index profile.",
        },
    )
    ar_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "ar_low",
            "serialize": True,
            "doc": "Root profile parameter a for low vegetation.",
        },
    )
    br_low: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "br_low",
            "serialize": True,
            "doc": "Root profile parameter b for low vegetation.",
        },
    )
    ar_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "ar_high",
            "serialize": True,
            "doc": "Root profile parameter a for high vegetation.",
        },
    )
    br_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "br_high",
            "serialize": True,
            "doc": "Root profile parameter b for high vegetation.",
        },
    )
    gD_high: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "gD_high",
            "serialize": True,
            "doc": "Vapour pressure deficit response coefficient for high vegetation.",
        },
    )
    tskin_water: float | None = field(
        default=None,
        metadata={
            "nml": "NAMLSM_HOMOGENEOUS",
            "key": "tskin_water",
            "serialize": True,
            "doc": "Skin temperature assigned to water tile in K.",
        },
    )
    thls: float | None = field(
        default=None,
        metadata={
            "nml": "NAMSURFACE",
            "key": "thls",
            "required": True,
            "serialize": True,
            "forcing_var": "thls",
            "doc": "Surface potential temperature in K.",
        },
    )

    lheterogeneous: bool = field(
        default=False,
        metadata={
            "nml": "NAMLSM",
            "key": "lheterogeneous",
            "required": True,
            "serialize": True,
            "doc": "Use homogeneous LSM mode.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "LSMHomogeneousModule"
        self.nlu = 6

    def do_config(self):
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

    def prepare_calculation(self):
        for name in ("t_soil_p", "theta_soil_p", "soil_index_p"):
            values = getattr(self, name)
            if values is not None and len(values) != int(self.kmax_soil):
                raise ValueError(
                    f"{self.__class__.__name__}.{name} length ({len(values)}) must match kmax_soil ({self.kmax_soil})."
                )
