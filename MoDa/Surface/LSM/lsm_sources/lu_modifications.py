import warnings
from dataclasses import dataclass, field
from typing import Literal, Self

from moda.Geometry.ModifierClass import GeometricModification
from moda.module_registry import register_module
from moda.Surface.LSM.lsm_data import LandUseName
from .context import LSMSourceContext


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

    def apply_source(self, context: LSMSourceContext) -> None:
        context.writer.apply_modification(self)
        context.writer.init_lutypes_ifs()
        context.writer.recalculate_remaining_cover()


@register_module
@dataclass
class LandUseModifications:
    """Collection of land use modifications."""

    modifications: list[LandUseModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def apply_source(self, context: LSMSourceContext) -> None:
        for modification in self.modifications:
            context.writer.apply_modification(modification)
        context.writer.init_lutypes_ifs()
        context.writer.recalculate_remaining_cover()

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
