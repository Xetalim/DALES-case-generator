"""Map LCZ and fractional NetCDF sources onto typed land-use state."""

from __future__ import annotations

import numpy as np
import xarray as xr

from modular_dales.Surface.LSM.lsm_data import (
    FloatArray,
    LandUseName,
    LandUseState,
    SoilFields,
)


def distribute_fraction(
    land_use: dict[LandUseName, LandUseState], fraction: FloatArray, ifs_id: int
) -> None:
    """Share an IFS fraction equally between tiles with the same mapping."""
    matching = [
        state for state in land_use.values() if state.definition.ifs_id == ifs_id
    ]
    for state in matching:
        state.lu_frac[:] = fraction / len(matching)


def apply_lcz(
    land_use: dict[LandUseName, LandUseState], soil: SoilFields, dataset: xr.Dataset
) -> None:
    """Populate soil classes and land-use fractions from an LCZ dataset."""
    soil.index_soil[:] = dataset["index_soil"].values
    cover = dataset["ifs_land_cover"].values
    ifs_ids = {state.definition.ifs_id for state in land_use.values()}
    for ifs_id in ifs_ids:
        if ifs_id is not None:
            distribute_fraction(
                land_use, np.asarray(cover == ifs_id, dtype=np.float64), ifs_id
            )


def apply_from_netcdf(
    land_use: dict[LandUseName, LandUseState], soil: SoilFields, ds: xr.Dataset
) -> None:
    """Populate soil classes and clipped fractional land cover from NetCDF."""
    soil.index_soil[:] = ds["index_soil"].values
    water = np.asarray(ds["frac_water"].values, dtype=np.float64).clip(0, 1)
    sea = np.asarray(ds["frac_sea"].values, dtype=np.float64).clip(0, 1)
    town = np.asarray(ds["frac_town"].values, dtype=np.float64).clip(0, 1)
    nature = np.asarray(ds["frac_nature"].values, dtype=np.float64).clip(0, 1)
    ifs_cover = np.asarray(ds["ifs_land_cover"].values, dtype=np.float64)
    distribute_fraction(land_use, (water + sea).clip(0, 1), ifs_id=22)
    distribute_fraction(land_use, town, ifs_id=20)
    natural_states = [
        state
        for state in land_use.values()
        if state.definition.ifs_id not in (None, 20, 22)
    ]
    if natural_states:
        weights = np.stack(
            [ifs_cover == state.definition.ifs_id for state in natural_states]
        ).astype(np.float64)
        total = weights.sum(axis=0)
        weights[:, total == 0] = 1.0 / len(natural_states)
        total[total == 0] = 1.0
        for state, weight in zip(natural_states, weights, strict=True):
            state.lu_frac[:] = nature * weight / total
