import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import netCDF4
import numpy as np
from modular_dales.Geometry.geometry_modification import (
    GeometricModification,
    ModifierClass,
)
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class SLURBVariableModification:
    """Typed SLURB variable modification payload."""

    varname: str
    value: int | float
    dtype: Literal["float", "real", "int", "integer"] = "float"
    n_layers: int | None = None

    def to_numpy_dtype(self):
        """Map supported dtype labels to numpy-compatible dtypes."""
        if self.dtype in ("float", "real"):
            return float
        if self.dtype in ("int", "integer"):
            return int
        raise ValueError(f"Invalid dtype given: {self.dtype}")


@register_module
@dataclass
class SLURBModification(GeometricModification):
    """Single SLURB modification."""

    vars: list[SLURBVariableModification] = field(default_factory=list)
    """List of typed variable modifications."""


@register_module
@dataclass
class SLURBModifications:
    """Collection of SLURB modifications."""

    modifications: list[SLURBModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def __add__(
        self, modification: SLURBModification | list[SLURBModification]
    ) -> "SLURBModifications":
        """Add a modification."""
        if isinstance(modification, list):
            self.modifications.extend(modification)
        else:
            self.modifications.append(modification)
        return self

    def __iadd__(
        self, modification: SLURBModification | list[SLURBModification]
    ) -> "SLURBModifications":
        """In-place addition."""
        return self.__add__(modification)


class slbCreatorClass(ModifierClass):
    def __init__(self, grid: GridDales):
        super().__init__(grid)
        self.vars = {}

    def init_var(self, submod: SLURBVariableModification):
        varname = submod.varname
        dtype = submod.to_numpy_dtype()
        if not (varname in self.vars):
            shape = self.meshx.shape
            if submod.n_layers is not None:
                shape = (submod.n_layers, *shape)
                n_layers = submod.n_layers
            else:
                n_layers = None

            self.vars[varname] = {"dtype": dtype, "n_layers": n_layers}
            setattr(self, varname, np.zeros(shape, dtype=dtype))

    def do_modification(self, geometry, modification):
        for submod in modification.vars:
            self.init_var(submod)
            if submod.n_layers is not None:
                newgeometry = np.tile(geometry, [submod.n_layers, 1, 1])
                getattr(self, submod.varname)[newgeometry] = submod.value
            else:
                getattr(self, submod.varname)[geometry] = submod.value

    def output_nc(self, filename):
        with netCDF4.Dataset(filename, "w") as nc:

            nc.createDimension("x", len(self.x))
            nc.createDimension("y", len(self.y))

            var_x = nc.createVariable("x", float, "x")
            var_y = nc.createVariable("y", float, "y")

            var_x[:] = self.x[:]
            var_y[:] = self.y[:]

            for var, var_dic in self.vars.items():
                dtype = var_dic["dtype"]

                if var_dic["n_layers"]:
                    nc.createDimension(f"z_{var}", var_dic["n_layers"])
                    nc.createVariable(f"z_{var}", float, f"z_{var}")[:] = np.arange(
                        var_dic["n_layers"]
                    )
                    nc.createVariable(var, dtype, ["y", "x", f"z_{var}"])[:, :, :] = (
                        np.transpose(getattr(self, var)[:, :, :], [1, 2, 0])
                    )
                else:
                    nc.createVariable(var, dtype, ["y", "x"])[:, :] = getattr(
                        self, var
                    )[:, :]
            self.grid.set_cf_grid_mapping(nc, "crs", list(self.vars.keys()))


@register_module
@dataclass
class SLURBModule(simulation_module):
    """Surface Layer Urban Boundary (SLURB) simulation module."""

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    slb_modifications: SLURBModifications = field(
        default_factory=SLURBModifications,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    slb_generator: Any = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    urban_fraction: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "urban_fraction",
            "doc": "Urban tile fraction used by the SLURB parameterization.",
        },
    )
    urban_roughness_length: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "urban_roughness_length",
            "doc": "Bulk roughness length for urban canopy momentum exchange in m.",
        },
    )
    building_plan_area_fraction: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "building_plan_area_fraction",
            "doc": "Plan area fraction covered by buildings.",
        },
    )
    building_frontal_area_fraction: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "building_frontal_area_fraction",
            "doc": "Frontal area density controlling urban drag.",
        },
    )
    building_height: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "building_height",
            "doc": "Representative building height in m.",
        },
    )
    window_fraction: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "window_fraction",
            "doc": "Fraction of facade area occupied by windows.",
        },
    )
    street_canyon_aspect_ratio: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "street_canyon_aspect_ratio",
            "doc": "Street canyon height-to-width ratio.",
        },
    )
    building_type: int | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "building_type",
            "doc": "Index selecting building material/thermal parameter set.",
        },
    )
    pavement_type: int | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "pavement_type",
            "doc": "Index selecting pavement material parameter set.",
        },
    )
    anisotropic_street_canyons: bool | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "anisotropic_street_canyons",
            "doc": "Enable anisotropic canyon treatment with separate facade directions.",
        },
    )
    street_canyon_orientation: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "street_canyon_orientation",
            "doc": "Azimuth orientation of street canyon axis in degrees.",
        },
    )
    deep_soil_temperature: float = field(
        default=283.15,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "deep_soil_temperature",
            "required": True,
            "doc": "Deep soil temperature boundary condition in K.",
        },
    )
    building_indoor_temperature: float = field(
        default=293.15,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "building_indoor_temperature",
            "required": True,
            "doc": "Indoor building temperature used for wall/roof heat transfer in K.",
        },
    )
    shf_external: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "shf_external",
            "doc": "External anthropogenic sensible heat flux source in W m-2.",
        },
    )
    qsws_external: float | None = field(
        default=None,
        init=True,
        metadata={
            "serialize": True,
            "nml": "NAMSLURB",
            "key": "qsws_external",
            "doc": "External anthropogenic latent heat flux source in W m-2.",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "SLURBModule"

    def __add__(self, obj) -> "SLURBModule":
        """Add SLURB modifications to module."""
        if isinstance(obj, SLURBModification):
            self.slb_modifications += obj
        elif isinstance(obj, SLURBModifications):
            self.slb_modifications += obj.modifications
        else:
            raise TypeError(
                f"Expected SLURBModification/SLURBModifications, got {type(obj)}"
            )
        return self

    def __iadd__(self, obj) -> "SLURBModule":
        return self.__add__(obj)

    def do_config(self):
        """Apply SLURB modifications to configuration."""
        return

    def prepare_calculation(self):
        return self.prepare_calculations()

    def init_generator(self):
        """Initialize SLURB generator."""
        if self.slb_generator is None:

            self.slb_generator = slbCreatorClass(self.grid)

    def prepare_calculations(self):
        """Set up SLURB geometry and update namelist."""
        # just to check, is grid even there???
        if self.grid is None:
            raise ValueError(
                "Grid must be set before preparing SLURB calculations (you did something wrong!)"
            )
        # Initialize generator if not already done
        self.init_generator()

        # Apply SLURB modifications
        for modification in self.slb_modifications.modifications:
            if modification is not None:
                self.slb_generator.apply_modification(modification)

    def check_settings(self):
        """Check SLURB settings validity."""
        return

    def write_files(self):
        """Write SLURB files."""
        # Note: NAMSLURB section exists but doesn't have a simple enable flag
        # The presence of SLURB data controls whether it's used
        logger.info("SLURBModule: SLURB configured")
        if self.slb_generator is not None:
            self.slb_generator.output_nc(
                self.output_path / "input" / f"inslurb.{self.exp_id:03d}.nc"
            )
