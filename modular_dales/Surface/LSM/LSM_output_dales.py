"""Own and modify DALES land-surface state using typed grid fields."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from modular_dales.Geometry.geometry_modification import (
    ModifierClass,
)
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.Surface.LSM.LandUseModifications import LandUseModification
from modular_dales.Surface.LSM.lsm_data import (
    LAND_USE_PARAMETERS,
    CoverParameter,
    FloatArray,
    LandUseFields,
    LandUseName,
    LandUseSelection,
    LandUseState,
    SoilFields,
)
from modular_dales.Surface.LSM.translation_tables.landuse_types import (
    LandUseSet,
    get_land_use_definitions,
)
from modular_dales.Surface.LSM.translation_tables.vegetation_properties import (
    ifs_vegetation,
)

logger = logging.getLogger(__name__)


class LSM_output_dales:
    """Land-surface fields on a DALES grid.

    ``land_use`` contains catalogue metadata and source fractions;
    ``land_use_fields`` contains the resulting DALES tile parameters;
    ``soil`` contains the three-dimensional soil fields. Flattened field names
    exist only at the NetCDF output boundary.
    """

    grid: GridDales
    land_use_set: LandUseSet
    soil_levels: int
    shape: tuple[int, int]
    soil: SoilFields
    land_use: dict[LandUseName, LandUseState]
    land_use_fields: dict[LandUseName, LandUseFields]
    cover_tot: FloatArray
    c_veg_tot: FloatArray
    LCZ_ds: xr.Dataset | None

    def __init__(
        self,
        grid: GridDales,
        lu_types: LandUseSet = LandUseSet.DEPAC,
        soil_levels: int = 4,
        debug: bool = False,
        *,
        include_slurb: bool = False,
    ) -> None:
        if soil_levels < 1:
            raise ValueError("soil_levels must be positive")
        self.grid = grid
        self.land_use_set = LandUseSet(lu_types)
        self.soil_levels = soil_levels
        self.shape = (grid.jtot, grid.itot)
        self.soil = SoilFields.zeros((soil_levels, *self.shape))
        definitions = get_land_use_definitions(self.land_use_set)
        self.land_use = {
            name: LandUseState(
                definition=definition,
                lu_frac=np.zeros(self.shape, dtype=np.float64),
                lu_domid=np.full(
                    self.shape,
                    definition.ifs_id if definition.ifs_id is not None else np.nan,
                    dtype=np.float64,
                ),
            )
            for name, definition in definitions.items()
        }
        self.land_use_fields = {
            name: LandUseFields.zeros(self.shape) for name in self.land_use
        }
        self.cover_tot = np.zeros(self.shape, dtype=np.float64)
        self.c_veg_tot = np.zeros(self.shape, dtype=np.float64)
        self.LCZ_ds = None
        if debug:
            self.soil.t_soil.fill(np.nan)
            self.soil.theta_soil.fill(np.nan)
            self.cover_tot.fill(np.nan)
            self.c_veg_tot.fill(np.nan)
            for tile in self.land_use_fields.values():
                for parameter in LAND_USE_PARAMETERS:
                    tile.get(parameter).fill(np.nan)

    @property
    def nlu(self) -> int:
        return len(self.land_use)

    def standard_fill_geometry_modification(self) -> None:
        """Initialize the four-layer reference soil and fill uncovered land."""
        self.soil.index_soil.fill(3)
        self.soil.set_uniform_moisture([0.36867549, 0.25300502, 0.14997292, 0.16459982])
        self.soil.set_uniform_temperature(
            [283.26038083, 286.79894009, 290.88998902, 288.09079126]
        )
        self.init_lutypes_ifs()
        self.set_skin_temperature(273.15)
        self.recalculate_remaining_cover()

    def trim_landuse(self) -> None:
        """Remove unused tiles and all their parameters together."""
        unused = [
            name
            for name, tile in self.land_use_fields.items()
            if not np.any(tile.cover > 0)
        ]
        if unused:
            logger.info("Trimming unused land use types: %s", unused)
        for name in unused:
            del self.land_use[name]
            del self.land_use_fields[name]

    def _selected_tiles(self, selection: LandUseSelection) -> Iterator[LandUseFields]:
        if selection == "all":
            yield from self.land_use_fields.values()
        else:
            yield self.land_use_fields[selection]

    def set_skin_temperature(
        self, temperature: float, lu_type: LandUseSelection = "all"
    ) -> None:
        for tile in self._selected_tiles(lu_type):
            tile.tskin.fill(temperature)

    def get_laqu_lu_names(self) -> list[LandUseName]:
        return [name for name, state in self.land_use.items() if state.definition.laqu]

    def set_skin_temperature_laqu(self, temperature: float) -> None:
        for name in self.get_laqu_lu_names():
            self.land_use_fields[name].tskin.fill(temperature)

    def set_skin_temperature_array(
        self, temperature_array: FloatArray, lu_type: LandUseSelection = "all"
    ) -> None:
        if temperature_array.shape != self.shape:
            raise ValueError(
                f"Incompatible array shape {temperature_array.shape}; expected {self.shape}"
            )
        for tile in self._selected_tiles(lu_type):
            tile.tskin[:] = temperature_array

    def init_lutypes_ifs(self) -> None:
        """Derive tile parameters from source fractions and IFS lookup tables."""
        for name, state in self.land_use.items():
            ifs_id = state.definition.ifs_id
            if ifs_id is None:
                raise ValueError(
                    f"Land-use set {self.land_use_set.value!r}, tile {name!r}, has no IFS mapping"
                )
            tile = self.land_use_fields[name]
            tile.cover[:] = state.lu_frac
            tile.c_veg[:] = state.lu_frac
            tile.lutype.fill(ifs_id)
            tile.tskin.fill(0.0)
            for parameter in LAND_USE_PARAMETERS:
                if parameter in ("cover", "c_veg", "lutype", "tskin"):
                    continue
                ifs_parameter = (
                    "a_r"
                    if parameter == "ar"
                    else "b_r"
                    if parameter == "br"
                    else parameter
                )
                tile.get(parameter).fill(getattr(ifs_vegetation, ifs_parameter)[ifs_id])
        self._update_totals()

    def _update_totals(self) -> None:
        self.cover_tot[:] = self.calc_totcover("cover")
        self.c_veg_tot[:] = self.calc_totcover("c_veg")

    def recalculate_remaining_cover(self) -> None:
        """Assign the uncovered fraction to the catalogue's bare-soil tile."""
        bare_soil = next(
            (
                name
                for name, state in self.land_use.items()
                if state.definition.bare_soil
            ),
            None,
        )
        if bare_soil is None:
            raise ValueError(
                f"Land-use set {self.land_use_set.value!r} has no bare-soil tile"
            )
        tile = self.land_use_fields[bare_soil]
        tile.cover[:] = np.round(1.0 - self.calc_totcover("cover") + tile.cover, 6)
        self._update_totals()

    def calc_totcover(self, ctype: CoverParameter) -> FloatArray:
        total = np.zeros(self.shape, dtype=np.float64)
        for tile in self.land_use_fields.values():
            total += tile.get(ctype)
        return total


class LsmModifier(ModifierClass[LandUseModification]):
    """Apply geometric fraction changes to typed land-use state."""

    def __init__(self, lsm_input: LSM_output_dales, grid: GridDales) -> None:
        self.lsm_input = lsm_input
        super().__init__(grid)

    def set_type(
        self,
        mask: NDArray[np.bool_],
        lu_type: LandUseName,
        frac: float = 1.0,
        mode: Literal["replace", "add"] = "replace",
    ) -> None:
        if not np.isfinite(frac) or not 0 <= frac <= 1:
            raise ValueError(f"Land-use fraction must be in [0, 1], got {frac}")
        if mask.shape != self.lsm_input.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match {self.lsm_input.shape}"
            )
        if lu_type not in self.lsm_input.land_use:
            raise KeyError(
                f"Land-use type {lu_type!r} is not in {self.lsm_input.land_use_set.value!r}"
            )
        fraction = self.lsm_input.land_use[lu_type].lu_frac
        if mode == "replace":
            fraction[mask] = frac
            if frac == 1:
                for name, state in self.lsm_input.land_use.items():
                    if name != lu_type:
                        state.lu_frac[mask] = 0
        elif mode == "add":
            target = fraction[mask] + frac
            if np.any(target > 1):
                raise ValueError(
                    f"Add mode would set land-use fraction above 1 for {lu_type!r}"
                )
            fraction[mask] = target
        else:
            raise ValueError(f"Unsupported land-use modification mode {mode!r}")

    def do_modification(
        self, geometry: NDArray[np.bool_], modification: LandUseModification
    ) -> None:

        if not isinstance(modification, LandUseModification):
            raise TypeError("LsmModifier requires a LandUseModification")
        self.set_type(geometry, modification.type, modification.frac, modification.mode)
