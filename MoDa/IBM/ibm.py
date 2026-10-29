"""Immersed Boundary Method (IBM) module for building and geometry representation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from moda.Core.namelist import nml_field
from moda.Geometry import (
    GridDales,
)
from moda.Geometry.ModifierClass import ModifierClass
from moda.IBM.ibm_data_sources import (
    FromAHN,
    FromGlobalDEM,
    IBMModification,
    IBMModifications,
    IBMSourceSpec,
    _PrioritizedIBMSource,
)
from moda.modular.simulation_module import simulation_module
from moda.module_registry import register_module

from .download_ahn import get_process_ahn
from .global_dem import get_process_global_dem

logger = logging.getLogger(__name__)


class IBMCreatorClass(ModifierClass[IBMModification]):
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

    def do_modification(
        self, geometry: np.ndarray, modification: IBMModification
    ) -> None:
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

    def output_nc(self, filename: str | Path) -> None:
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
            self.grid.set_cf_grid_mapping_netcdf(nc, "crs", ["bc_height"])


@register_module
@dataclass
class IBMModule(simulation_module):
    """Build IBM heights from terrain and geometric sources in priority order.

    Add sources with ``add(source, priority=...)``. Higher priorities apply
    later; equal priorities preserve insertion order. Terrain overlays replace
    positive cells, while geometric sources can replace, add or subtract heights.
    """

    sources: list[_PrioritizedIBMSource] = field(
        default_factory=list, metadata={"serialize": True}
    )
    ibm_generator: IBMCreatorClass | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    apply_ibm: bool = nml_field(
        default=True,
        section="IBM",
        key="lapply_ibm",
        required=True,
        description="Enable immersed boundary method in IBM:lapply_ibm.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    lwallheat: bool | None = nml_field(
        default=None,
        section="IBM",
        key="lwallheat",
        description="Enable lateral wall heat flux treatment for buildings.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    thlwall: float | None = nml_field(
        default=None,
        section="IBM",
        key="thlwall",
        description="Potential temperature at building side walls in K.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    thlibm: float | None = nml_field(
        default=None,
        section="IBM",
        key="thlibm",
        required=True,
        description="Interior obstacle potential temperature in K.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    thlroof: float | None = nml_field(
        default=None,
        section="IBM",
        key="thlroof",
        required=True,
        description="Potential temperature at obstacle roofs in K.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    qtibm: float | None = nml_field(
        default=None,
        section="IBM",
        key="qtibm",
        description="Interior obstacle specific humidity in kg/kg.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    lpoislast: bool | None = nml_field(
        default=None,
        section="IBM",
        key="lpoislast",
        description="If true, apply Poisson solver after IBM velocity correction.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    z0m_wall: float | None = nml_field(
        default=None,
        section="IBM",
        key="z0m_wall",
        description="Wall roughness length for momentum in m.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    z0h_wall: float | None = nml_field(
        default=None,
        section="IBM",
        key="z0h_wall",
        description="Wall roughness length for heat/scalars in m.",
        metadata={"serialize": True},
        init=True,
        repr=False,
    )
    ibas_prf: int = nml_field(
        default=2,
        section="DYNAMICS",
        key="IBAS_PRF",
        required=True,
        description="Base-state profile selector, forced to IBM-compatible choice by DALES.",
        metadata={"serialize": True},
        init=False,
        repr=False,
    )
    iadv_mom: int = nml_field(
        default=2,
        section="DYNAMICS",
        key="IADV_MOM",
        required=True,
        description="Momentum advection scheme; IBM is typically used with second-order centered scheme.",
        metadata={"serialize": True},
        init=False,
        repr=False,
    )

    module_name: str = field(
        default="IBMModule", init=False, metadata={"serialize": False}
    )

    def add(
        self,
        obj: IBMSourceSpec | list[IBMSourceSpec],
        *,
        priority: int,
    ) -> None:
        """Add sources at an explicit priority; higher priorities apply later."""
        providers = obj if isinstance(obj, list) else [obj]
        entries = [_PrioritizedIBMSource(source, priority) for source in providers]
        self.sources.extend(entries)

    def _validate_sources(self) -> None:
        """Require validated, explicitly prioritized source entries."""
        for entry in self.sources:
            if not isinstance(entry, _PrioritizedIBMSource):
                raise TypeError(
                    "Use IBMModule.add(source, priority=...) to add sources"
                )
            entry.validate()

    def has_terrain_source(self) -> bool:
        """Return whether any terrain source was configured."""

        return any(
            isinstance(entry.source, (FromAHN, FromGlobalDEM)) for entry in self.sources
        )

    def _apply_terrain_source(self, source: FromAHN | FromGlobalDEM) -> None:
        """Overlay positive terrain heights onto the current field."""
        if self.ibm_generator is None:
            raise RuntimeError("IBM generator has not been initialized")
        if isinstance(source, FromGlobalDEM):
            dem_path = Path(source.dem_path).expanduser()
            terrain = get_process_global_dem(
                self.grid,
                dem_path=dem_path,
                zeroes_buffer=source.zeroes_buffer,
                subtract_dem_mode=source.subtract_dem_mode,
                resampling_name=source.resampling,
            )
        else:
            terrain = get_process_ahn(
                self.grid,
                zeroes_buffer=source.zeroes_buffer,
                subtract_ahn_mode=source.subtract_ahn_mode,
            )
        values = np.asarray(terrain, dtype=float)
        heights = self.ibm_generator.bc_height
        if values.shape != heights.shape:
            raise ValueError(
                "IBM terrain source shape mismatch: "
                f"{type(source).__name__} has shape {values.shape}, "
                f"expected {heights.shape}"
            )
        heights[:] = np.where(values > 0.0, values, heights)

    def _apply_sources(self) -> None:
        """Apply terrain and geometry from lowest to highest priority."""
        if self.ibm_generator is None:
            raise RuntimeError("IBM generator has not been initialized")
        self._validate_sources()
        for entry in sorted(self.sources, key=lambda source: source.priority):
            source = entry.source
            if isinstance(source, (FromAHN, FromGlobalDEM)):
                self._apply_terrain_source(source)
            else:
                modifications = (
                    source.modifications
                    if isinstance(source, IBMModifications)
                    else [source]
                )
                for modification in modifications:
                    self.ibm_generator.apply_modification(modification)

    def prepare_calculation(self) -> None:
        """Set up IBM geometry and update namelist."""

        self.ibm_generator = IBMCreatorClass(self.grid)
        self._apply_sources()

        logger.info("IBMModule: IBM configured and namelist updated")

    def check_settings(self) -> None:
        """Check IBM settings validity."""

        self._validate_sources()
        for entry in self.sources:
            source = entry.source
            if isinstance(source, FromGlobalDEM):
                dem_path = Path(source.dem_path).expanduser()
                if not dem_path.exists():
                    raise FileNotFoundError(f"Global DEM file not found: {dem_path}")
            modifications = (
                source.modifications
                if isinstance(source, IBMModifications)
                else [source]
                if isinstance(source, IBMModification)
                else []
            )
            for modification in modifications:
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

    def write_files(self) -> None:
        """Write IBM files."""
        if self.ibm_generator is not None:
            self.ibm_generator.output_nc(
                self.output_path / "input" / f"ibm.inp_{self.exp_id:03d}.nc"
            )
