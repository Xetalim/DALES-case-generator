import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from modular_dales.Geometry import GridDales, GridDalesOpenBC
from modular_dales.LBC.nesting_idx import NestingIndices
from modular_dales.LBC.openbc import do_openboundary
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class NestingTopology(simulation_module):
    """Nesting topology module to determine grid relationships for nested simulations.

    Args:
        sim: Parent simulation instance
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)
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
    indices: dict | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    nestings: list["GridDales"] = field(
        default_factory=list, repr=False, metadata={"serialize": True}, init=True
    )
    my_idx: int | None = field(
        default=None, repr=False, metadata={"serialize": True}, init=True
    )
    openbc_module: Optional["do_openboundary"] = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    lcross: bool | None = field(
        default=True,
        metadata={
            "nml": "namcrosssection",
            "key": "lcross",
            "required": True,
        },
        init=False,
    )
    lstat: bool | None = field(
        default=True,
        metadata={
            "nml": "namgenstat",
            "key": "lstat",
            "required": True,
        },
        init=False,
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "nesting_topology"

    def __add__(self, other):
        """Combine nesting topologies if needed."""
        if isinstance(other, GridDales):
            logger.info("Adding GridDales to Nesting_Topology: %s", other)
            self.nestings.append(other)
        else:
            raise TypeError("Can only add GridDales instances to Nesting_Topology")
        return self

    def __iadd__(self, other):
        """In-place addition for combining nesting topologies."""
        return self.__add__(other)

    def __repr__(self):
        parts = ""
        for idx, nesting in enumerate(self.nestings):
            parts += f"\n  Nesting {idx + 1}: itot={nesting.itot}, jtot={nesting.jtot}, kmax={nesting.kmax}, xsize={nesting.xsize}, ysize={nesting.ysize}, x0={nesting.x0}, y0={nesting.y0}"
            if idx == len(self.nestings) - 1:
                parts += " (bottom-level nest):"
            if idx == 0:
                parts += " (top-level nest):"
        if not parts:
            parts = "No nestings defined"
        return f"Nesting_Topology with {len(self.nestings)} nestings:{parts}"

    def prepare_calculation(self):
        """Return nesting info for a specific index if available."""

        if len(self.nestings) > self.my_idx + 1:
            # another grid is nested inside us, we need to set up the indices for the cross sections in the namelist
            subnest_subgrid = self.nestings[self.my_idx + 1].as_openbc()
            subnest_supergrid = self.grid

            ixwest, ixeast = (
                list(subnest_supergrid.xm).index(np.min(subnest_subgrid.xm)),
                list(subnest_supergrid.xm).index(np.max(subnest_subgrid.xm)),
            )

            iysouth, iynorth = (
                list(subnest_supergrid.ym).index(np.min(subnest_subgrid.ym)),
                list(subnest_supergrid.ym).index(np.max(subnest_subgrid.ym)),
            )
            iztop = list(subnest_supergrid.zt).index(np.max(subnest_subgrid.zt))

            existing_height = list(
                self.nml.get("namcrosssection", {}).get("crossheight", [])
            )
            existing_plane = list(
                self.nml.get("namcrosssection", {}).get("crossplane", [])
            )
            existing_ortho = list(
                self.nml.get("namcrosssection", {}).get("crossortho", [])
            )

            merged_height = list(dict.fromkeys(existing_height + [iztop + 2]))
            merged_plane = list(
                dict.fromkeys(existing_plane + [iysouth + 1, iynorth + 1])
            )
            merged_ortho = list(
                dict.fromkeys(existing_ortho + [ixwest + 1, ixeast + 1])
            )

            self.set_nml_section("namcrosssection", "crossheight", merged_height)
            self.set_nml_section("namcrosssection", "crossplane", merged_plane)
            self.set_nml_section("namcrosssection", "crossortho", merged_ortho)
        if self.my_idx > 0:
            # we are nested inside another grid, we need to set up the indices for the interpolation of the boundary conditions from the supergrid
            supernest_subgrid = self.grid.as_openbc()
            supernest_supergrid = self.nestings[self.my_idx - 1]

            ixwest, ixeast = (
                list(supernest_supergrid.xm).index(np.min(supernest_subgrid.xm)),
                list(supernest_supergrid.xm).index(np.max(supernest_subgrid.xm)),
            )

            iysouth, iynorth = (
                list(supernest_supergrid.ym).index(np.min(supernest_subgrid.ym)),
                list(supernest_supergrid.ym).index(np.max(supernest_subgrid.ym)),
            )

            iztop = list(supernest_supergrid.zt).index(np.max(supernest_subgrid.zt))

            if self.module_exists(do_openboundary):
                self.openbc_module = self.retrieve_module(do_openboundary)
            else:
                raise ValueError(
                    "Nesting_Topology requires a do_openboundary module to set indices for boundary condition interpolation."
                )
            self.openbc_module.indices = NestingIndices(
                supergrid=supernest_supergrid,
                ix_west=ixwest,
                ix_east=ixeast,
                iy_south=iysouth,
                iy_north=iynorth,
                subgrid_x0=supernest_subgrid.x0,
                subgrid_y0=supernest_subgrid.y0,
                supergrid_x0=supernest_supergrid.x0,
                supergrid_y0=supernest_supergrid.y0,
            )

    def write_files(self):
        pass
