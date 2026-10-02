import logging

from modular_dales.Geometry.GridDales import GridDales

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


class openbc_counts_idx:
    def __init__(self, grid: GridDales, fortran_indexing=True):
        # i1, j1, k1, i2, j2, imax, jmax, kmax, nxturb=0, nyturb=0, nzturb=0
        nxturb = 0
        nyturb = 0
        nzturb = 0
        # boundary(west)%nx1turb = nyturb; boundary(west)%nx2turb = nzturb
        boundaries = {
            "west": {
                "nx1": grid.jmax,
                "nx2": grid.kmax,
                "nx1u": grid.jmax,
                "nx2u": grid.kmax,
                "nx1v": grid.j1,
                "nx2v": grid.kmax,
                "nx1w": grid.jmax,
                "nx2w": grid.k1,
                "nx1turb": nyturb,
                "nx2turb": nzturb,
            },
            # boundary(east)%nx1turb = nyturb; boundary(2)%nx2turb = nzturb
            "east": {
                "nx1": grid.jmax,
                "nx2": grid.kmax,
                "nx1u": grid.jmax,
                "nx2u": grid.kmax,
                "nx1v": grid.j1,
                "nx2v": grid.kmax,
                "nx1w": grid.jmax,
                "nx2w": grid.k1,
                "nx1turb": nyturb,
                "nx2turb": nzturb,
            },
            # boundary(south)%nx1turb = nxturb; boundary(3)%nx2turb = nzturb
            "south": {
                "nx1": grid.imax,
                "nx2": grid.kmax,
                "nx1u": grid.i1,
                "nx2u": grid.kmax,
                "nx1v": grid.imax,
                "nx2v": grid.kmax,
                "nx1w": grid.imax,
                "nx2w": grid.k1,
                "nx1turb": nxturb,
                "nx2turb": nzturb,
            },
            # boundary(north)%nx1turb = nxturb; boundary(4)%nx2turb = nzturb
            "north": {
                "nx1": grid.imax,
                "nx2": grid.kmax,
                "nx1u": grid.i1,
                "nx2u": grid.kmax,
                "nx1v": grid.imax,
                "nx2v": grid.kmax,
                "nx1w": grid.imax,
                "nx2w": grid.k1,
                "nx1turb": nxturb,
                "nx2turb": nzturb,
            },
            # boundary(top)%nx1turb = nxturb; boundary(5)%nx2turb = nyturb
            "top": {
                "nx1": grid.imax,
                "nx2": grid.jmax,
                "nx1u": grid.i1,
                "nx2u": grid.jmax,
                "nx1v": grid.imax,
                "nx2v": grid.j1,
                "nx1w": grid.imax,
                "nx2w": grid.jmax,
                "nx1turb": nxturb,
                "nx2turb": nyturb,
            },
        }
        # u0(2:i2,2:j1,1:kmax)
        # v0(2:i1,2:j2,1:kmax)
        # w0(2:i1,2:j1,1:k1)
        # thl0(2:i1,2:j1,1:kmax)
        # qt0(2:i1,2:j1,1:kmax)
        # e120(2:i1,2:j1,1:kmax)
        # sv0(2:i1,2:j1,1:kmax,tracer_prop(n)%trac_idx)
        openbc_initfields_idx = {
            "u0": {"x": (2, grid.i2), "y": (2, grid.j1), "z": (1, grid.kmax)},
            "v0": {"x": (2, grid.i1), "y": (2, grid.j2), "z": (1, grid.kmax)},
            "w0": {"x": (2, grid.i1), "y": (2, grid.j1), "z": (1, grid.k1)},
            "thl0": {"x": (2, grid.i1), "y": (2, grid.j1), "z": (1, grid.kmax)},
            "qt0": {"x": (2, grid.i1), "y": (2, grid.j1), "z": (1, grid.kmax)},
            "e120": {"x": (2, grid.i1), "y": (2, grid.j1), "z": (1, grid.kmax)},
            "sv0": {"x": (2, grid.i1), "y": (2, grid.j1), "z": (1, grid.kmax)},
        }

        var_initfield_counts = {
            var: {
                dim: (
                    openbc_initfields_idx[var][dim][1]
                    - openbc_initfields_idx[var][dim][0]
                    + 1
                )
                for dim in ["x", "y", "z"]
            }
            for var in openbc_initfields_idx
        }

        var_boundary_idx = {
            boundary: {
                "u0": (boundaries[boundary]["nx1u"], boundaries[boundary]["nx2u"]),
                "v0": (boundaries[boundary]["nx1v"], boundaries[boundary]["nx2v"]),
                "w0": (boundaries[boundary]["nx1w"], boundaries[boundary]["nx2w"]),
                "thl0": (boundaries[boundary]["nx1"], boundaries[boundary]["nx2"]),
                "qt0": (boundaries[boundary]["nx1"], boundaries[boundary]["nx2"]),
                "e120": (boundaries[boundary]["nx1"], boundaries[boundary]["nx2"]),
                "sv0": (boundaries[boundary]["nx1"], boundaries[boundary]["nx2"]),
                "u2": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "v2": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "w2": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "uv": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "uw": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "vw": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "thl2": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "qt2": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "wthl": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
                "wqt": (
                    boundaries[boundary]["nx1turb"],
                    boundaries[boundary]["nx2turb"],
                ),
            }
            for boundary in boundaries
        }

        var_boundary_counts = {
            boundary: {
                var: {
                    var_boundary_idx[boundary][var][1]
                    - var_boundary_idx[boundary][var][0]
                    + 1
                }
                for var in openbc_initfields_idx
            }
            for boundary in boundaries
        }

        if not fortran_indexing:
            for boundary in var_boundary_idx:
                for var in var_boundary_idx[boundary]:
                    var_boundary_idx[boundary][var] = (
                        var_boundary_idx[boundary][var][0] - 1,
                        var_boundary_idx[boundary][var][1] - 1,
                    )
            for var in openbc_initfields_idx:
                for dim in openbc_initfields_idx[var]:
                    openbc_initfields_idx[var][dim] = (
                        openbc_initfields_idx[var][dim][0] - 1,
                        openbc_initfields_idx[var][dim][1] - 1,
                    )

        self.boundaries = boundaries
        self.var_boundary_idx = var_boundary_idx
        self.var_boundary_counts = var_boundary_counts
        self.var_initfield_counts = var_initfield_counts
        self.openbc_initfields_idx = openbc_initfields_idx
