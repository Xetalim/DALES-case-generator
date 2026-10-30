"""Serializable IBM terrain and geometry sources."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from moda.Geometry.ModifierClass import GeometricModification
from moda.modular.sources import Source
from moda.module_registry import register_module

from .download_ahn import get_process_ahn
from .global_dem import get_process_global_dem

if TYPE_CHECKING:
    from .ibm import IBMCreatorClass


def _overlay_terrain(context: IBMCreatorClass, terrain: np.ndarray) -> None:
    values = np.asarray(terrain, dtype=float)
    heights = context.bc_height
    if values.shape != heights.shape:
        raise ValueError(
            f"IBM terrain source shape mismatch: {values.shape}, expected {heights.shape}"
        )
    heights[:] = np.where(values > 0.0, values, heights)


@register_module
@dataclass
class FromAHN:
    """AHN (Actueel Hoogtebestand Nederland) approach for IBM.
    Add this to IBMModule to generate IBM geometry from AHN data."""

    has_terrain_source = True

    def apply_source(self, context: IBMCreatorClass) -> None:
        _overlay_terrain(context, np.asarray(get_process_ahn(context.grid)))


@register_module
@dataclass
class FromGlobalDEM:
    """Use a user-provided global DEM raster as IBM terrain input.

    The DEM must be readable by rasterio, for example a GeoTIFF or a cloud
    optimized GeoTIFF. It is reprojected onto the DALES grid. Sources are applied
    in explicit priority order; positive terrain cells override earlier heights.
    """

    dem_path: str
    resampling: str = field(default="bilinear", metadata={"serialize": True})
    has_terrain_source = True

    def check_settings(self) -> None:
        dem_path = Path(self.dem_path).expanduser()
        if not dem_path.exists():
            raise FileNotFoundError(f"Global DEM file not found: {dem_path}")

    def apply_source(self, context: IBMCreatorClass) -> None:
        terrain = get_process_global_dem(
            context.grid,
            dem_path=Path(self.dem_path).expanduser(),
            resampling_name=self.resampling,
        )
        _overlay_terrain(context, np.asarray(terrain))


@register_module
@dataclass
class IBMPostProcessing:
    """Normalize and buffer one source or a combined group before overlaying it.

    Wrapped sources apply in insertion order to an isolated height field.
    Zero or negative results leave previously applied layers untouched.
    """

    sources: IBMSourceSpec | list[IBMSourceSpec] = field(metadata={"serialize": True})
    zeroes_buffer: int = field(
        default=5,
        metadata={"serialize": True, "doc": "Number of edge cells set to zero."},
    )
    subtract_dem_mode: bool = field(
        default=True,
        metadata={"serialize": True, "doc": "Subtract the DEM terrain baseline."},
    )

    def _sources(self) -> list[IBMSourceSpec]:
        return self.sources if isinstance(self.sources, list) else [self.sources]

    @property
    def has_terrain_source(self) -> bool:
        return any(
            getattr(source, "has_terrain_source", False) for source in self._sources()
        )

    def check_settings(self) -> None:
        if not isinstance(self.zeroes_buffer, int) or self.zeroes_buffer < 0:
            raise ValueError("zeroes_buffer must be a non-negative integer")
        for source in self._sources():
            if not isinstance(source, Source):
                raise TypeError(f"Unsupported IBM source type: {type(source)}")
            check = getattr(source, "check_settings", None)
            if check is not None:
                check()

    def apply_source(self, context: IBMCreatorClass) -> None:
        from .ibm import IBMCreatorClass

        self.check_settings()
        layer = IBMCreatorClass(context.grid)
        for source in self._sources():
            source.apply_source(layer)
        values = layer.bc_height
        values[:] -= values.min()
        if self.subtract_dem_mode:
            finite = values[np.isfinite(values)]
            if finite.size:
                mode = np.argmax(np.bincount(np.round(finite).astype(int)))
                values[:] = np.floor(np.maximum(values - mode, 0.0))
        if self.zeroes_buffer:
            width = self.zeroes_buffer
            values[:width, :] = 0
            values[-width:, :] = 0
            values[:, :width] = 0
            values[:, -width:] = 0
        values[:] -= values.min()
        _overlay_terrain(context, values)


@register_module
@dataclass
class IBMModification(GeometricModification):
    """Geometric height source with replace, add, or subtract mode."""

    height: float = 0.0
    mode: str = "replace"

    def check_settings(self) -> None:
        if str(self.mode).strip().lower() not in {"replace", "add", "subtract"}:
            raise ValueError(
                "IBMModification.mode must be one of: replace, add, subtract"
            )

    def apply_source(self, context: IBMCreatorClass) -> None:
        context.apply_modification(self)


@register_module
@dataclass
class IBMModifications:
    """Collection of geometric height sources applied in insertion order."""

    modifications: list[IBMModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def check_settings(self) -> None:
        for modification in self.modifications:
            modification.check_settings()

    def apply_source(self, context: IBMCreatorClass) -> None:
        for modification in self.modifications:
            modification.apply_source(context)

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


IBMSourceSpec = Source["IBMCreatorClass"]


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
        if not isinstance(self.source, Source):
            raise TypeError(f"Unsupported IBM source type: {type(self.source)}")
        if not isinstance(self.priority, int):
            raise TypeError("Source priority must be an integer")
