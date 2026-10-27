import warnings
from dataclasses import dataclass, field
from typing import Literal, Self

from modular_dales.Geometry.geometry_modification import GeometricModification
from modular_dales.module_registry import register_module
from modular_dales.Surface.LSM.lsm_data import LandUseName


@register_module
@dataclass
class LandUseModification(GeometricModification):
    """Single land use modification."""

    type: LandUseName = "grs"
    """Land use type (grs, urb, fbd, etc.)"""
    frac: float = 1.0
    """Optional fractional land-use coverage for selected cells."""
    mode: Literal["replace", "add"] = field(default="replace")
    """How to apply fraction on selected cells: replace or add."""


@register_module
@dataclass
class LandUseModifications:
    """Collection of land use modifications."""

    modifications: list[LandUseModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def add(self, modification: LandUseModification) -> Self:
        """Add a modification."""
        self.modifications.append(modification)
        return self

    def __add__(self, modification: LandUseModification) -> Self:
        return self.add(modification)

    def __iadd__(self, modification: LandUseModification) -> Self:
        """In-place addition."""
        warnings.warn(
            "Using '+=' to add land-use modifications is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(modification)
