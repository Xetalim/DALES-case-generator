"""Serializable IBM terrain and geometry sources."""

import warnings
from dataclasses import dataclass, field
from typing import Self

from moda.Geometry.ModifierClass import GeometricModification
from moda.module_registry import register_module


@register_module
@dataclass
class FromAHN:
    """AHN (Actueel Hoogtebestand Nederland) approach for IBM.
    Add this to IBMModule to generate IBM geometry from AHN data."""

    zeroes_buffer: int = field(
        default=5,
        metadata={
            "serialize": True,
            "doc": "Padding in cells for zero-elevation holes.",
        },
    )
    subtract_ahn_mode: bool = field(
        default=True,
        metadata={"serialize": True, "doc": "Subtract the AHN terrain baseline."},
    )


@register_module
@dataclass
class FromGlobalDEM:
    """Use a user-provided global DEM raster as IBM terrain input.

    The DEM must be readable by rasterio, for example a GeoTIFF or a cloud
    optimized GeoTIFF. It is reprojected onto the DALES grid before the same
    post-processing used for AHN is applied. Sources are applied in explicit
    priority order; positive terrain cells override earlier heights.
    """

    dem_path: str
    resampling: str = field(default="bilinear", metadata={"serialize": True})
    zeroes_buffer: int = field(
        default=5,
        metadata={
            "serialize": True,
            "doc": "Padding in cells for zero-elevation holes.",
        },
    )
    subtract_dem_mode: bool = field(
        default=True,
        metadata={"serialize": True, "doc": "Subtract the DEM terrain baseline."},
    )


@register_module
@dataclass
class IBMModification(GeometricModification):
    """Geometric height source with replace, add, or subtract mode."""

    height: float = 0.0
    mode: str = "replace"


@register_module
@dataclass
class IBMModifications:
    """Collection of geometric height sources applied in insertion order."""

    modifications: list[IBMModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def add(self, modification: IBMModification) -> Self:
        """Add a modification."""
        self.modifications.append(modification)
        return self

    def __add__(self, modification: IBMModification) -> Self:
        return self.add(modification)

    def __iadd__(self, modification: IBMModification) -> Self:
        warnings.warn(
            "Using '+=' to add IBM modifications is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(modification)


IBMSourceSpec = FromAHN | FromGlobalDEM | IBMModification | IBMModifications


@register_module
@dataclass
class _PrioritizedIBMSource:
    """Serializable source entry; higher priorities apply later."""

    source: IBMSourceSpec = field(metadata={"serialize": True})
    priority: int = field(metadata={"serialize": True})

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject unsupported sources and non-integer priorities."""
        if not isinstance(self.source, IBMSourceSpec):
            raise TypeError(f"Unsupported IBM source type: {type(self.source)}")
        if not isinstance(self.priority, int):
            raise TypeError("Source priority must be an integer")
