from dataclasses import dataclass
from typing import Optional

import numpy as np

from modular_dales.Geometry import GridDales


@dataclass
class NestingIndices:
    ix_west: int
    ix_east: int
    iy_south: int
    iy_north: int
    supergrid_x0: float
    supergrid_y0: float
    subgrid_x0: float
    subgrid_y0: float
    iz_top: int
    supergrid: Optional["GridDales"] = None

    @classmethod
    def from_grids(cls, parent: GridDales, child: GridDales) -> "NestingIndices":
        """Resolve aligned child boundaries against the parent's output coordinates."""
        for label, grid in (("parent", parent), ("child", child)):
            for name, expected in (("zt", grid.kmax), ("zm", grid.kmax + 1)):
                values = np.asarray(getattr(grid, name), dtype=float)
                if values.shape != (expected,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
                    raise ValueError(
                        f"Nesting {label} grid.{name} must contain {expected} finite, increasing levels; got shape {values.shape}"
                    )

        def aligned(values, target: float, label: str) -> int:
            matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1e-8))
            if matches.size != 1:
                raise ValueError(
                    f"Child {label} boundary at {target} m must align with exactly one parent grid coordinate"
                )
            return int(matches[0])

        parent_bc = parent.as_openbc()
        child_bc = child.as_openbc()
        ix_west = aligned(parent_bc.xm, child_bc.xm[0], "west")
        ix_east = aligned(parent_bc.xm, child_bc.xm[-1], "east")
        iy_south = aligned(parent_bc.ym, child_bc.ym[0], "south")
        iy_north = aligned(parent_bc.ym, child_bc.ym[-1], "north")
        iz_top = aligned(parent.zm, child.zm[-1], "top")
        if ix_east >= len(parent.xt) or iy_north >= len(parent.yt):
            raise ValueError("Child east/north boundaries must lie inside the parent domain to select output-cell coordinates")
        if iz_top >= len(parent.zt):
            raise ValueError("Child top must lie below the parent top to select a cross-section with both zt and zm coordinates")
        return cls(
            ix_west=ix_west,
            ix_east=ix_east,
            iy_south=iy_south,
            iy_north=iy_north,
            iz_top=iz_top,
            supergrid_x0=parent.x0,
            supergrid_y0=parent.y0,
            subgrid_x0=child.x0,
            subgrid_y0=child.y0,
            supergrid=parent,
        )
