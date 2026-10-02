"""Immersed Boundary Method (IBM) module for building and geometry representation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import netCDF4
import numpy as np

from modular_dales.Geometry import (
    GridDales,
    ModifierClass,
)
from modular_dales.Geometry.geometry_modification import GeometricModification
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module, register_singleton

from .download_AHN import get_process_ahn
from .global_dem import get_process_global_dem

logger = logging.getLogger(__name__)


@register_singleton
@register_module
@dataclass
class FromAHN:
    """AHN (Actueel Hoogtebestand Nederland) approach for IBM.
    Add this to IBMModule to generate IBM geometry from AHN data."""


@register_module
@dataclass
class FromGlobalDEM:
    """Use a user-provided global DEM raster as IBM terrain input.

    The DEM must be readable by rasterio, for example a GeoTIFF or a cloud
    optimized GeoTIFF. It is reprojected onto the DALES grid before the same
    post-processing used for AHN is applied. When combined with ``FromAHN``,
    the global DEM is used as a base field and AHN refines cells where it has
    positive resolved terrain.
    """

    dem_path: str
    resampling: str = field(default="bilinear", metadata={"serialize": True})


@dataclass(frozen=True)
class _TerrainLayer:
    """Resolved terrain field plus metadata used during IBM preparation."""

    name: str
    values: np.ndarray


@register_module
@dataclass
class IBMModification(GeometricModification):
    """Single IBM modification. Set Geometry type and parameters to define the modification."""

    height: float = 0.0
    """Building height"""
    mode: str = field(default="replace")
    """How the modification affects terrain: replace, add, or subtract."""


class IBMCreatorClass(ModifierClass):
    """
    A class for creating and managing IBM (Immersed Boundary Method) modifications to DALES grid geometry.

    This class handles the creation and output of boundary condition heights for immersed boundary
    method simulations. It maintains a 2D array of boundary condition heights that can be modified
    based on geometry specifications and exported to NetCDF format.

    Attributes:
        bc_height (np.ndarray): A 2D array storing boundary condition heights, initialized with zeros
                                matching the grid mesh dimensions.
    """

    def __init__(self, grid: GridDales):
        super().__init__(grid)
        self.bc_height = np.zeros_like(self.meshx)

    def do_modification(self, geometry, modification):
        height = float(modification.height)
        mode = str(modification.mode).strip().lower()

        if mode == "replace":
            self.bc_height[geometry] = height
            return

        if mode == "add":
            self.bc_height[geometry] = self.bc_height[geometry] + height
            return

        if mode == "subtract":
            self.bc_height[geometry] = np.maximum(
                self.bc_height[geometry] - height,
                0.0,
            )
            return

        raise ValueError(
            f"Unsupported IBM modification mode '{mode}'. Expected one of: replace, add, subtract"
        )

    def output_nc(self, filename):
        with netCDF4.Dataset(filename, "w") as nc:
            nc.createDimension("x", len(self.x))
            nc.createDimension("y", len(self.y))

            var_x = nc.createVariable("x", float, "x")
            var_y = nc.createVariable("y", float, "y")

            var_x[:] = self.x[:]
            var_y[:] = self.y[:]

            dims = ["y", "x"]
            bc_height_var = nc.createVariable("bc_height", float, dims)
            bc_height_var[:, :] = self.bc_height[:, :]
            self.grid.set_cf_grid_mapping(nc, "crs", ["bc_height"])


@register_module
@dataclass
class IBMModule(simulation_module):
    """IBM Module for DALES simulation.

    This module handles Immersed Boundary Method (IBM) configuration and setup for DALES
    (Dutch Atmospheric Large-Eddy Simulation) simulations. It manages IBM modifications,
    AHN (Actueel Hoogtebestand Nederland) terrain data integration, and generates IBM
    geometry files.

    Attributes:
        sim (Optional[simulation_module]): Reference to parent simulation module.
        ibm_modifications (List[IBMModification]): List of IBM geometry modifications to apply.
        ibm_generator (Any): IBM geometry creator instance, initialized during preparation.
        from_ahn (Optional[FromAHN]): AHN terrain data configuration object.
        from_global_dem (Optional[FromGlobalDEM]): Global DEM terrain source.
        ahn_zeroes_buffer (int): Buffer size for AHN zero elevation handling. Default: 5.
        subtract_ahn_mode (bool): Whether to subtract AHN elevation data. Default: True.
        apply_ibm (bool): Enable/disable IBM application. Default: True.
        ibas_prf (int): Fixed base-state profile selector. Always 2.
        iadv_mom (int): Fixed momentum advection scheme selector. Always 2.

    Methods:
        __post_init__: Initialize module and set module name.
        __add__: Add IBM modification or AHN configuration to module.
        __iadd__: In-place addition operator for fluent API.
        prepare_calculation: Set up IBM geometry, integrate AHN data if configured, and apply modifications.
        check_settings: --
        write_files: Export IBM geometry to NetCDF file.
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    ibm_modifications: list[IBMModification] = field(
        default_factory=list, init=True, repr=False, metadata={"serialize": True}
    )
    ibm_generator: Any = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    from_ahn: FromAHN | None = field(
        default=None,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    from_global_dem: FromGlobalDEM | None = field(
        default=None,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    ahn_zeroes_buffer: int = field(
        default=5,
        metadata={
            "serialize": True,
            "doc": "Padding in cells used when filling zero-elevation holes in terrain input.",
        },
        init=True,
        repr=False,
    )
    subtract_ahn_mode: bool = field(
        default=True,
        metadata={
            "serialize": True,
            "doc": "When true, subtract terrain baseline before writing IBM heights.",
        },
        init=True,
        repr=False,
    )
    apply_ibm: bool = field(
        default=True,
        metadata={
            "nml": "IBM",
            "key": "lapply_ibm",
            "serialize": True,
            "required": True,
            "doc": "Enable immersed boundary method in IBM:lapply_ibm.",
        },
        init=True,
        repr=False,
    )
    lwallheat: bool | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "lwallheat",
            "serialize": True,
            "doc": "Enable lateral wall heat flux treatment for buildings.",
        },
        init=True,
        repr=False,
    )
    thlwall: float | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "thlwall",
            "serialize": True,
            "doc": "Potential temperature at building side walls in K.",
        },
        init=True,
        repr=False,
    )
    thlibm: float | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "thlibm",
            "serialize": True,
            "required": True,
            "doc": "Interior obstacle potential temperature in K.",
        },
        init=True,
        repr=False,
    )
    thlroof: float | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "thlroof",
            "serialize": True,
            "required": True,
            "doc": "Potential temperature at obstacle roofs in K.",
        },
        init=True,
        repr=False,
    )
    qtibm: float | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "qtibm",
            "serialize": True,
            "doc": "Interior obstacle specific humidity in kg/kg.",
        },
        init=True,
        repr=False,
    )
    lpoislast: bool | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "lpoislast",
            "serialize": True,
            "doc": "If true, apply Poisson solver after IBM velocity correction.",
        },
        init=True,
        repr=False,
    )
    z0m_wall: float | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "z0m_wall",
            "serialize": True,
            "doc": "Wall roughness length for momentum in m.",
        },
        init=True,
        repr=False,
    )
    z0h_wall: float | None = field(
        default=None,
        metadata={
            "nml": "IBM",
            "key": "z0h_wall",
            "serialize": True,
            "doc": "Wall roughness length for heat/scalars in m.",
        },
        init=True,
        repr=False,
    )
    ibas_prf: int = field(
        default=2,
        metadata={
            "nml": "DYNAMICS",
            "key": "IBAS_PRF",
            "serialize": True,
            "required": True,
            "doc": "Base-state profile selector, forced to IBM-compatible choice by DALES.",
        },
        init=False,
        repr=False,
    )
    iadv_mom: int = field(
        default=2,
        metadata={
            "nml": "DYNAMICS",
            "key": "IADV_MOM",
            "serialize": True,
            "required": True,
            "doc": "Momentum advection scheme; IBM is typically used with second-order centered scheme.",
        },
        init=False,
        repr=False,
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "IBMModule"

    def __add__(self, obj) -> "IBMModule":
        """Add IBM modification to module.

        Args:
            modification: IBMModification to add

        Returns:
            self for chaining
        """
        if isinstance(obj, FromAHN):
            self.from_ahn = obj
        elif isinstance(obj, FromGlobalDEM):
            self.from_global_dem = obj
        elif isinstance(obj, IBMModification):
            self.ibm_modifications.append(obj)
        else:
            raise TypeError(
                f"Cannot add object of type {type(obj)} to IBMModule. Expected FromAHN, FromGlobalDEM or IBMModification."
            )
        return self

    def __iadd__(self, obj) -> "IBMModule":
        """In-place addition."""
        return self.__add__(obj)

    def has_terrain_source(self) -> bool:
        """Return whether any terrain source was configured."""

        return self.from_ahn is not None or self.from_global_dem is not None

    def _terrain_layers(self) -> list[_TerrainLayer]:
        """Load configured terrain sources in merge order.

        The global DEM is treated as the coarse base terrain, while AHN acts as
        a higher-quality refinement that overrides strictly positive cells.
        This preserves the previous AHN-only behavior and keeps the combined
        path deterministic.
        """

        layers: list[_TerrainLayer] = []

        if self.from_global_dem is not None:
            dem_path = Path(self.from_global_dem.dem_path).expanduser()
            terrain = get_process_global_dem(
                self.grid,
                dem_path=dem_path,
                zeroes_buffer=self.ahn_zeroes_buffer,
                subtract_dem_mode=self.subtract_ahn_mode,
                resampling_name=self.from_global_dem.resampling,
            )
            layers.append(
                _TerrainLayer(
                    name=f"global_dem:{dem_path}",
                    values=np.asarray(terrain, dtype=float),
                )
            )

        if self.from_ahn is not None:
            terrain = get_process_ahn(
                self.grid,
                zeroes_buffer=self.ahn_zeroes_buffer,
                subtract_ahn_mode=self.subtract_ahn_mode,
            )
            layers.append(
                _TerrainLayer(name="ahn", values=np.asarray(terrain, dtype=float))
            )

        return layers

    def _merge_terrain_layers(self, layers: list[_TerrainLayer]) -> np.ndarray | None:
        """Merge loaded terrain layers into a single IBM height field."""

        if not layers:
            return None

        merged = np.array(layers[0].values, copy=True, dtype=float)

        for layer in layers[1:]:
            if layer.values.shape != merged.shape:
                raise ValueError(
                    "IBM terrain source shape mismatch: "
                    f"{layer.name} has shape {layer.values.shape}, expected {merged.shape}"
                )
            merged = np.where(layer.values > 0.0, layer.values, merged)

        return merged

    def _apply_terrain(self) -> None:
        """Populate the IBM terrain field from configured terrain sources."""

        terrain_layers = self._terrain_layers()
        terrain = self._merge_terrain_layers(terrain_layers)
        if terrain is None:
            return

        self.ibm_generator.bc_height[:, :] = terrain[:, :]
        logger.info(
            "IBMModule: applied terrain sources %s",
            ", ".join(layer.name for layer in terrain_layers),
        )

    def prepare_calculation(self):
        """Set up IBM geometry and update namelist."""

        self.ibm_generator = IBMCreatorClass(self.grid)
        self._apply_terrain()

        # Apply IBM modifications
        for modification in self.ibm_modifications:
            self.ibm_generator.apply_modification(modification)

        logger.info("IBMModule: IBM configured and namelist updated")

    def check_settings(self):
        """Check IBM settings validity."""

        if self.grid is None:
            raise ValueError("IBMModule requires a GridDales grid to be configured")

        if self.from_global_dem is not None:
            dem_path = Path(self.from_global_dem.dem_path).expanduser()
            if not dem_path.exists():
                raise FileNotFoundError(f"Global DEM file not found: {dem_path}")

        for modification in self.ibm_modifications:
            mode = str(modification.mode).strip().lower()
            if mode not in {"replace", "add", "subtract"}:
                raise ValueError(
                    "IBMModification.mode must be one of: replace, add, subtract"
                )

        if self.thlibm is None:
            raise ValueError("IBMModule.thlibm is mandatory and must be provided")

        if self.thlroof is None:
            raise ValueError("IBMModule.thlroof is mandatory and must be provided")

        if self.lwallheat is True and self.thlwall is None:
            raise ValueError(
                "IBMModule.thlwall is mandatory when IBMModule.lwallheat is True"
            )

        if self.ibas_prf != 2:
            raise ValueError("IBMModule.ibas_prf is fixed and must be 2")

        if self.iadv_mom != 2:
            raise ValueError("IBMModule.iadv_mom is fixed and must be 2")

    def write_files(self):
        """Write IBM files."""
        if self.ibm_generator is not None:
            self.ibm_generator.output_nc(
                self.output_path / "input" / f"ibm.inp_{self.exp_id:03d}.nc"
            )


@register_module
@dataclass
class IBMModifications:
    """Collection of IBM modifications."""

    modifications: list[IBMModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def __add__(self, modification: IBMModification) -> "IBMModifications":
        """Add a modification."""
        self.modifications.append(modification)
        return self

    def __iadd__(self, modification: IBMModification) -> "IBMModifications":
        """In-place addition."""
        return self.__add__(modification)
