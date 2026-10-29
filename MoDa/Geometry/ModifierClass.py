from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from moda.Geometry.grids import GridDales
from moda.Geometry.masks import MaskSpec


@dataclass
class GeometricModification:
    """Base class for modifications that target a typed horizontal geometry."""

    mask: MaskSpec


ModificationT = TypeVar("ModificationT", bound=GeometricModification)


class ModifierClass(Generic[ModificationT]):
    """
    A class for generating and applying geometric modifications to DALES grid data.

    This class provides utilities to create various geometric shapes (circles, rectangles)
    in both real-space and index-space coordinates on a DALES grid, and to apply
    modifications based on these geometries.

    Attributes:
        x (ndarray): X-coordinates of the grid.
        y (ndarray): Y-coordinates of the grid.
        meshx (ndarray): 2D mesh grid of X-coordinates.
        meshy (ndarray): 2D mesh grid of Y-coordinates.
        idxmesh (ndarray): 2D mesh grid of X-indices.
        idymesh (ndarray): 2D mesh grid of Y-indices.

    Args:
        grid (GridDales): A GridDales object containing grid coordinate information.
    """

    x: npt.NDArray[np.floating[Any]]
    y: npt.NDArray[np.floating[Any]]
    meshx: npt.NDArray[np.floating[Any]]
    meshy: npt.NDArray[np.floating[Any]]
    idxmesh: npt.NDArray[np.integer[Any]]
    idymesh: npt.NDArray[np.integer[Any]]
    grid: GridDales

    def __init__(self, grid: GridDales):
        self.x = grid.xt
        self.y = grid.yt
        self.meshx, self.meshy = np.meshgrid(self.x, self.y)
        self.idxmesh, self.idymesh = np.meshgrid(
            np.arange(len(self.x)), np.arange(len(self.y))
        )
        self.grid = grid

    def apply_modification(self, modification: ModificationT) -> None:
        """Apply a typed modification object to this modifier."""

        if not isinstance(modification, GeometricModification):
            raise TypeError(
                "Modification must inherit GeometricModification, got "
                f"{type(modification)}"
            )
        geometry_obj = modification.mask
        if not hasattr(geometry_obj, "to_mask") or not callable(geometry_obj.to_mask):
            raise TypeError(
                f"geometry must implement to_mask(modifier), got {type(geometry_obj)}"
            )
        geometry = geometry_obj.to_mask(self)
        self.do_modification(geometry, modification)

    def do_modification(
        self, geometry: npt.NDArray[np.bool_], modification: ModificationT
    ) -> None:
        raise NotImplementedError(
            "Can't call do_modification of superclass, call the subclass instead"
        )
