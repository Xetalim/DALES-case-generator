import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import netCDF4
import numpy as np
import xarray as xr

from moda.Geometry.grids import GridDales
from moda.Geometry.ModifierClass import GeometricModification, ModifierClass
from moda.module_registry import register_module
from moda.Surface.LSM.LCZ import get_from_lcz
from moda.Surface.LSM.lsm_data import FloatArray

logger = logging.getLogger(__name__)


@register_module
@dataclass
class SlurbModification(GeometricModification):
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
class SlurbModifications:
    """Collection of SLURB modifications."""

    modifications: list[SlurbModification] = field(
        default_factory=list, metadata={"serialize": True}
    )

    def add(self, modification: SlurbModification | list[SlurbModification]) -> Self:
        """Add one or more SLURB modifications."""
        if isinstance(modification, list):
            self.modifications.extend(modification)
        else:
            self.modifications.append(modification)
        return self

    def __add__(
        self, modification: SlurbModification | list[SlurbModification]
    ) -> Self:
        """Add a modification."""
        return self.add(modification)

    def __iadd__(
        self, modification: SlurbModification | list[SlurbModification]
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


class slbCreatorClass(ModifierClass[SlurbModification]):
    def __init__(self, grid: GridDales):
        super().__init__(grid)
        self.vars: dict[str, SLURBVariableMetadata] = {}

    def init_var(self, submod: SlurbModification) -> None:
        self._init_var(submod.varname, submod.to_numpy_dtype(), submod.n_layers)

    def _init_var(
        self, varname: str, dtype: type[int] | type[float], n_layers: int | None
    ) -> None:
        metadata = SLURBVariableMetadata(dtype=dtype, n_layers=n_layers)
        if varname in self.vars:
            if self.vars[varname] != metadata:
                raise ValueError(
                    f"Incompatible dtype or layer count for SLURB field {varname!r}"
                )
            return
        shape = self.meshx.shape if n_layers is None else (n_layers, *self.meshx.shape)
        self.vars[varname] = metadata
        setattr(self, varname, np.zeros(shape, dtype=dtype))

    def apply_field(
        self,
        varname: str,
        values: np.ndarray,
        *,
        dtype: type[int] | type[float] = float,
    ) -> None:
        """Copy a full (y, x) or (layer, y, x) field into SLURB storage."""
        values = np.asarray(values, dtype=dtype)
        if (
            values.ndim not in (2, 3)
            or values.shape[-2:] != self.meshx.shape
            or (values.ndim == 3 and values.shape[0] == 0)
        ):
            raise ValueError(
                f"Incompatible SLURB field shape {values.shape}; expected "
                f"{self.meshx.shape} or (n_layers, {self.meshx.shape[0]}, "
                f"{self.meshx.shape[1]})"
            )
        n_layers = values.shape[0] if values.ndim == 3 else None
        self._init_var(varname, dtype, n_layers)
        getattr(self, varname)[:] = values

    def do_modification(
        self, geometry: np.ndarray, modification: SlurbModification
    ) -> None:
        self.init_var(modification)
        if modification.n_layers is not None:
            newgeometry = np.tile(geometry, [modification.n_layers, 1, 1])
            getattr(self, modification.varname)[newgeometry] = modification.value
        else:
            getattr(self, modification.varname)[geometry] = modification.value

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


def apply_from_netcdf(ds: xr.Dataset, generator: slbCreatorClass) -> None:
    """Apply the optional SLuRB morphology fields supplied by a dataset."""
    for field in ("hw_can", "f_bld", "h_bld", "z0_urb"):
        if field in ds:
            generator.apply_field(field, ds[field].values)


def apply_slurb_parameters_lcz(
    dataset: xr.Dataset, generator: slbCreatorClass, urban_cover: FloatArray | None
) -> None:
    """Apply LCZ morphology, scaling building fraction by urban tile cover."""
    for source, target in get_from_lcz.LCZ_field_to_slurb.items():
        values = np.asarray(dataset[source].values, dtype=np.float64)
        if target == "f_bld":
            if urban_cover is None:
                logger.warning("No urban tile (IFS id 20); using unscaled f_bld")
            else:
                values = np.clip(np.nan_to_num(values, nan=0.01), 0.001, 1.0) * np.clip(
                    urban_cover, 0.0, 1.0
                )
        generator.apply_field(target, values)
