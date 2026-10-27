import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import netCDF4
import numpy as np

from MoDa.Geometry.geometry_modification import (
    GeometricModification,
    ModifierClass,
)
from MoDa.Geometry.grids import GridDales
from MoDa.modular.namelist import nml_field
from MoDa.modular.simulation_module import simulation_module
from MoDa.module_registry import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class SLURBVariableModification:
    """Typed SLURB variable modification payload."""

    varname: str
    value: int | float
    dtype: Literal["float", "real", "int", "integer"] = "float"
    n_layers: int | None = None

    def to_numpy_dtype(self) -> type[int | float]:
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

    def add(self, modification: SLURBModification | list[SLURBModification]) -> Self:
        """Add one or more SLURB modifications."""
        if isinstance(modification, list):
            self.modifications.extend(modification)
        else:
            self.modifications.append(modification)
        return self

    def __add__(
        self, modification: SLURBModification | list[SLURBModification]
    ) -> Self:
        """Add a modification."""
        return self.add(modification)

    def __iadd__(
        self, modification: SLURBModification | list[SLURBModification]
    ) -> Self:
        """In-place addition."""
        warnings.warn(
            "Using '+=' to add SLURB modifications is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(modification)


@dataclass
class SLURBVariableMetadata:
    """Storage metadata for an initialized SLURB variable."""

    dtype: type[int] | type[float]
    n_layers: int | None = None


class slbCreatorClass(ModifierClass[SLURBModification]):
    def __init__(self, grid: GridDales):
        super().__init__(grid)
        self.vars: dict[str, SLURBVariableMetadata] = {}

    def init_var(self, submod: SLURBVariableModification) -> None:
        varname = submod.varname
        dtype = submod.to_numpy_dtype()
        if not (varname in self.vars):
            shape = self.meshx.shape
            if submod.n_layers is not None:
                shape = (submod.n_layers, *shape)
                n_layers = submod.n_layers
            else:
                n_layers = None

            self.vars[varname] = SLURBVariableMetadata(dtype=dtype, n_layers=n_layers)
            setattr(self, varname, np.zeros(shape, dtype=dtype))

    def do_modification(
        self, geometry: np.ndarray, modification: SLURBModification
    ) -> None:
        for submod in modification.vars:
            self.init_var(submod)
            if submod.n_layers is not None:
                newgeometry = np.tile(geometry, [submod.n_layers, 1, 1])
                getattr(self, submod.varname)[newgeometry] = submod.value
            else:
                getattr(self, submod.varname)[geometry] = submod.value

    def output_nc(self, filename: str | Path) -> None:
        with netCDF4.Dataset(filename, "w") as nc:
            nc.createDimension("x", len(self.x))
            nc.createDimension("y", len(self.y))

            var_x = nc.createVariable("x", float, "x")
            var_y = nc.createVariable("y", float, "y")

            var_x[:] = self.x[:]
            var_y[:] = self.y[:]

            for var, metadata in self.vars.items():
                dtype = metadata.dtype

                if metadata.n_layers:
                    nc.createDimension(f"z_{var}", metadata.n_layers)
                    nc.createVariable(f"z_{var}", float, f"z_{var}")[:] = np.arange(
                        metadata.n_layers
                    )
                    nc.createVariable(var, dtype, ["y", "x", f"z_{var}"])[:, :, :] = (
                        np.transpose(getattr(self, var)[:, :, :], [1, 2, 0])
                    )
                else:
                    nc.createVariable(var, dtype, ["y", "x"])[:, :] = getattr(
                        self, var
                    )[:, :]
            self.grid.set_cf_grid_mapping_netcdf(nc, "crs", list(self.vars.keys()))


@register_module
@dataclass
class SLURBModule(simulation_module):
    """Surface Layer Urban Boundary (SLURB) simulation module."""

    slb_modifications: SLURBModifications = field(
        default_factory=SLURBModifications,
        init=True,
        repr=False,
        metadata={"serialize": True},
    )
    slb_generator: slbCreatorClass | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    urban_fraction: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="urban_fraction",
        description="Urban tile fraction used by the SLURB parameterization.",
        metadata={"serialize": True},
    )
    urban_roughness_length: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="urban_roughness_length",
        description="Bulk roughness length for urban canopy momentum exchange in m.",
        metadata={"serialize": True},
    )
    building_plan_area_fraction: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="building_plan_area_fraction",
        description="Plan area fraction covered by buildings.",
        metadata={"serialize": True},
    )
    building_frontal_area_fraction: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="building_frontal_area_fraction",
        description="Frontal area density controlling urban drag.",
        metadata={"serialize": True},
    )
    building_height: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="building_height",
        description="Representative building height in m.",
        metadata={"serialize": True},
    )
    window_fraction: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="window_fraction",
        description="Fraction of facade area occupied by windows.",
        metadata={"serialize": True},
    )
    street_canyon_aspect_ratio: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="street_canyon_aspect_ratio",
        description="Street canyon height-to-width ratio.",
        metadata={"serialize": True},
    )
    building_type: int | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="building_type",
        description="Index selecting building material/thermal parameter set.",
        metadata={"serialize": True},
    )
    pavement_type: int | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="pavement_type",
        description="Index selecting pavement material parameter set.",
        metadata={"serialize": True},
    )
    anisotropic_street_canyons: bool | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="anisotropic_street_canyons",
        description="Enable anisotropic canyon treatment with separate facade directions.",
        metadata={"serialize": True},
    )
    street_canyon_orientation: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="street_canyon_orientation",
        description="Azimuth orientation of street canyon axis in degrees.",
        metadata={"serialize": True},
    )
    deep_soil_temperature: float = nml_field(
        default=283.15,
        init=True,
        section="NAMSLURB",
        key="deep_soil_temperature",
        required=True,
        description="Deep soil temperature boundary condition in K.",
        metadata={"serialize": True},
    )
    building_indoor_temperature: float = nml_field(
        default=293.15,
        init=True,
        section="NAMSLURB",
        key="building_indoor_temperature",
        required=True,
        description="Indoor building temperature used for wall/roof heat transfer in K.",
        metadata={"serialize": True},
    )
    shf_external: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="shf_external",
        description="External anthropogenic sensible heat flux source in W m-2.",
        metadata={"serialize": True},
    )
    qsws_external: float | None = nml_field(
        default=None,
        init=True,
        section="NAMSLURB",
        key="qsws_external",
        description="External anthropogenic latent heat flux source in W m-2.",
        metadata={"serialize": True},
    )

    module_name: str = field(
        default="SLURBModule", init=False, metadata={"serialize": False}
    )

    def add(self, obj: SLURBModification | SLURBModifications) -> Self:
        """Add SLURB modifications to module."""
        if isinstance(obj, SLURBModification):
            self.slb_modifications.add(obj)
        elif isinstance(obj, SLURBModifications):
            self.slb_modifications.add(obj.modifications)
        else:
            raise TypeError(
                f"Expected SLURBModification/SLURBModifications, got {type(obj)}"
            )
        return self

    def __add__(self, obj: SLURBModification | SLURBModifications) -> Self:
        return self.add(obj)

    def __iadd__(self, obj: SLURBModification | SLURBModifications) -> Self:
        warnings.warn(
            "Using '+=' to add SLURB modifications is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(obj)

    def init_generator(self) -> None:
        """Initialize SLURB generator."""
        if self.slb_generator is None:
            self.slb_generator = slbCreatorClass(self.grid)

    def prepare_calculation(self) -> None:
        """Set up SLURB geometry and update namelist."""
        # Initialize generator if not already done
        self.init_generator()
        if self.slb_generator is None:
            raise RuntimeError("SLURB generator not initialized")

        # Apply SLURB modifications
        for modification in self.slb_modifications.modifications:
            if modification is not None:
                self.slb_generator.apply_modification(modification)

    def write_files(self) -> None:
        """Write SLURB files."""
        # Note: NAMSLURB section exists but doesn't have a simple enable flag
        # The presence of SLURB data controls whether it's used
        logger.info("SLURBModule: SLURB configured")
        if self.slb_generator is not None:
            self.slb_generator.output_nc(
                self.output_path / "input" / f"inslurb.{self.exp_id:03d}.nc"
            )
