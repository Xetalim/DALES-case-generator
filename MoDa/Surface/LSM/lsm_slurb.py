"""Apply source morphology and LCZ building-fraction scaling to SLuRB."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from MoDa.Geometry.geometry_modification import AllGeometry
from MoDa.Surface.LSM.LCZ import get_from_lcz
from MoDa.Surface.LSM.lsm_data import FloatArray
from MoDa.Surface.LSM.SLuRB.slurb import (
    SLURBModification,
    SLURBVariableModification,
)

if TYPE_CHECKING:
    from MoDa.Surface.LSM.SLuRB.slurb import slbCreatorClass

logger = logging.getLogger(__name__)


def _set_slurb_field(
    generator: slbCreatorClass, field: str, values: FloatArray
) -> None:
    generator.apply_modification(
        SLURBModification(
            geometry=AllGeometry(),
            vars=[SLURBVariableModification(varname=field, value=0, dtype="real")],
        )
    )
    getattr(generator, field)[:] = values


def apply_from_netcdf(ds: xr.Dataset, generator: slbCreatorClass) -> None:
    """Apply the optional SLuRB morphology fields supplied by a dataset."""
    for field in ("hw_can", "f_bld", "h_bld", "z0_urb"):
        if field in ds:
            _set_slurb_field(generator, field, ds[field].values)


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
        _set_slurb_field(generator, target, values)
