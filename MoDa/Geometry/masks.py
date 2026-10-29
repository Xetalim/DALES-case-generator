import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from moda.module_registry import register_module

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


class GeometryModifier(Protocol):
    """Horizontal meshes required to evaluate geometry masks."""

    meshx: npt.NDArray[np.floating[Any]]
    meshy: npt.NDArray[np.floating[Any]]
    idxmesh: npt.NDArray[np.integer[Any]]
    idymesh: npt.NDArray[np.integer[Any]]


@register_module
@dataclass
class AllMask:
    """Select the full 2D horizontal domain."""

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        return np.ones_like(modifier.meshx, dtype=bool)


@register_module
@dataclass
class CircleRealMask:
    """Circular mask in real-space coordinates (meters)."""

    x0: float
    y0: float
    size: float

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        mask = (modifier.meshx - self.x0) ** 2 + (
            modifier.meshy - self.y0
        ) ** 2 <= self.size**2
        return mask


@register_module
@dataclass
class RectangleRealMask:
    """Rectangular mask in real-space coordinates (meters)."""

    minx: float
    maxx: float
    miny: float
    maxy: float

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        return (
            (modifier.meshx >= self.minx)
            & (modifier.meshx <= self.maxx)
            & (modifier.meshy >= self.miny)
            & (modifier.meshy <= self.maxy)
        )


@register_module
@dataclass
class RectangleIdxMask:
    """Rectangular mask in index-space coordinates."""

    minx: int
    maxx: int
    miny: int
    maxy: int

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        return (
            (modifier.idxmesh >= self.minx)
            & (modifier.idxmesh <= self.maxx)
            & (modifier.idymesh >= self.miny)
            & (modifier.idymesh <= self.maxy)
        )


@register_module
@dataclass
class CircleIdxMask:
    """Circular mask in index-space coordinates."""

    idx0: int
    idy0: int
    size: float

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        return (modifier.idxmesh - self.idx0) ** 2 + (
            modifier.idymesh - self.idy0
        ) ** 2 <= self.size**2


@register_module
@dataclass
class ExplicitArrayMask:
    """Explicit boolean mask geometry.

    The provided mask must have shape ``(jtot, itot)`` after conversion to a
    numpy array and will be interpreted as a boolean selection.
    """

    mask: Any

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        mask_arr = np.asarray(self.mask)
        if mask_arr.shape != modifier.meshx.shape:
            raise ValueError(
                "ExplicitArrayMask shape mismatch: "
                f"expected {modifier.meshx.shape}, got {mask_arr.shape}"
            )
        return mask_arr.astype(bool)


@register_module
@dataclass
class FuncMask:
    """Explicit boolean function geometry.

    The provided function must return a boolean array with shape ``(jtot, itot)``
    after conversion to a numpy array and will be interpreted as a boolean selection.
    """

    func: Callable[[GeometryModifier], np.ndarray]

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        mask_arr = np.asarray(self.func(modifier))
        if mask_arr.shape != modifier.meshx.shape:
            raise ValueError(
                "FuncMask shape mismatch: "
                f"expected {modifier.meshx.shape}, got {mask_arr.shape}"
            )
        return mask_arr.astype(bool)


@register_module
@dataclass
class CheckerboardIdxMask:
    """Checkerboard mask in index-space coordinates.

    Cells are grouped in square blocks of size ``block`` (in grid cells).
    ``phase_x`` and ``phase_y`` shift the checkerboard pattern in index space.
    """

    block: int = 1
    phase_x: int = 0
    phase_y: int = 0
    select_even: bool = True

    def to_mask(self, modifier: GeometryModifier) -> npt.NDArray[np.bool_]:
        if self.block <= 0:
            raise ValueError(f"CheckerboardIdxMask.block must be > 0, got {self.block}")

        parity: npt.NDArray[np.bool_] = (
            ((modifier.idxmesh - self.phase_x) // self.block)
            + ((modifier.idymesh - self.phase_y) // self.block)
        ) % 2 == 0
        if self.select_even:
            return parity
        return ~parity


MaskSpec = (
    AllMask
    | CircleRealMask
    | RectangleRealMask
    | RectangleIdxMask
    | CircleIdxMask
    | ExplicitArrayMask
    | FuncMask
    | CheckerboardIdxMask
)
