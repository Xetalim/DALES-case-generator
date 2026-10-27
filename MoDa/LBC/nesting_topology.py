import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Self

from MoDa.Geometry import GridDales, GridDalesOpenBC
from MoDa.LBC.nesting_idx import NestingIndices
from MoDa.LBC.openbc import OpenBoundaryModule
from MoDa.modular.namelist import nml_field
from MoDa.modular.simulation_module import simulation_module
from MoDa.module_registry import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class NestingTopology(simulation_module):
    """Nesting topology module to determine grid relationships for nested simulations.

    Args:
        sim: Parent simulation instance
    """

    subnest_subgrid: Optional["GridDalesOpenBC"] = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    subnest_supergrid: Optional["GridDalesOpenBC"] = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    supernest_subgrid: Optional["GridDalesOpenBC"] = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    supernest_supergrid: Optional["GridDales"] = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    indices: dict[str, int] | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    nestings: list["GridDales"] = field(
        default_factory=list, repr=False, metadata={"serialize": True}, init=True
    )
    my_idx: int | None = field(
        default=None, repr=False, metadata={"serialize": True}, init=True
    )
    openbc_module: Optional["OpenBoundaryModule"] = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    lcross: bool | None = nml_field(
        default=True,
        section="namcrosssection",
        key="lcross",
        required=True,
        init=False,
    )
    lstat: bool | None = nml_field(
        default=True,
        section="namgenstat",
        key="lstat",
        required=True,
        init=False,
    )

    module_name: str = field(
        default="nesting_topology", init=False, metadata={"serialize": False}
    )

    def add(self, other: GridDales) -> Self:
        """Combine nesting topologies if needed."""
        if isinstance(other, GridDales):
            logger.info("Adding GridDales to nesting_topology: %s", other)
            self.nestings.append(other)
        else:
            raise TypeError("Can only add GridDales instances to nesting_topology")
        return self

    def __add__(self, other: GridDales) -> Self:
        return self.add(other)

    def __iadd__(self, other: GridDales) -> Self:
        """In-place addition for combining nesting topologies."""
        warnings.warn(
            "Using '+=' to add grids to NestingTopology is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(other)

    def __repr__(self) -> str:
        parts = ""
        for idx, nesting in enumerate(self.nestings):
            parts += f"\n  Nesting {idx + 1}: itot={nesting.itot}, jtot={nesting.jtot}, kmax={nesting.kmax}, xsize={nesting.xsize}, ysize={nesting.ysize}, x0={nesting.x0}, y0={nesting.y0}"
            if idx == len(self.nestings) - 1:
                parts += " (bottom-level nest):"
            if idx == 0:
                parts += " (top-level nest):"
        if not parts:
            parts = "No nestings defined"
        return f"nesting_topology with {len(self.nestings)} nestings:{parts}"

    def prepare_calculation(self) -> None:
        """Return nesting info for a specific index if available."""

        if self.my_idx is None or not 0 <= self.my_idx < len(self.nestings):
            raise ValueError("NestingTopology.my_idx must identify a grid in nestings")
        if len(self.nestings) > self.my_idx + 1:
            # another grid is nested inside us, we need to set up the indices for the cross sections in the namelist
            subnest_subgrid = self.nestings[self.my_idx + 1].as_openbc()
            subnest_supergrid = self.grid

            child_indices = NestingIndices.from_grids(
                subnest_supergrid, subnest_subgrid
            )

            existing_height = list(
                self.nml.get("namcrosssection", {}).get("crossheight", [])
            )
            existing_plane = list(
                self.nml.get("namcrosssection", {}).get("crossplane", [])
            )
            existing_ortho = list(
                self.nml.get("namcrosssection", {}).get("crossortho", [])
            )

            merged_height = list(
                dict.fromkeys(existing_height + [child_indices.iz_top + 1])
            )
            merged_plane = list(
                dict.fromkeys(
                    existing_plane
                    + [child_indices.iy_south + 1, child_indices.iy_north + 1]
                )
            )
            merged_ortho = list(
                dict.fromkeys(
                    existing_ortho
                    + [child_indices.ix_west + 1, child_indices.ix_east + 1]
                )
            )

            self.set_nml_section("namcrosssection", "crossheight", merged_height)
            self.set_nml_section("namcrosssection", "crossplane", merged_plane)
            self.set_nml_section("namcrosssection", "crossortho", merged_ortho)
        if self.my_idx > 0:
            # we are nested inside another grid, we need to set up the indices for the interpolation of the boundary conditions from the supergrid
            supernest_subgrid = self.grid.as_openbc()
            supernest_supergrid = self.nestings[self.my_idx - 1]

            if self.module_exists(OpenBoundaryModule):
                self.openbc_module = self.retrieve_module(OpenBoundaryModule)
            else:
                raise ValueError(
                    "nesting_topology requires a OpenBoundaryModule module to set indices for boundary condition interpolation."
                )
            self.openbc_module.indices = NestingIndices.from_grids(
                supernest_supergrid, supernest_subgrid
            )
