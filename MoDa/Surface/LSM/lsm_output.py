"""Own and modify DALES land-surface state using typed grid fields."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import netCDF4 as nc4
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from moda.Geometry.grids import GridDales
from moda.Geometry.ModifierClass import (
    ModifierClass,
)
from moda.Surface.LSM.lsm_data import (
    LAND_USE_PARAMETERS,
    CoverParameter,
    FloatArray,
    LandUseFields,
    LandUseName,
    LandUseSelection,
    LandUseState,
    LSMArray,
    SoilFields,
)
from moda.Surface.LSM.lsm_sources.lu_modifications import LandUseModification
from moda.Surface.LSM.translation_tables.landuse_types import (
    LandUseSet,
    get_land_use_definitions,
)
from moda.Surface.LSM.translation_tables.vegetation_properties import (
    ifs_vegetation,
)

logger = logging.getLogger(__name__)


class LSM_output_dales(ModifierClass[LandUseModification]):
    """Land-surface fields on a DALES grid.

    ``land_use`` contains catalogue metadata and source fractions;
    ``land_use_fields`` contains the resulting DALES tile parameters;
    ``soil`` contains the three-dimensional soil fields. Flattened field names
    exist only at the NetCDF output boundary. Geometric land-use modifications
    are applied directly with ``apply_modification``.
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
        super().__init__(grid)
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

    def set_type(
        self,
        mask: NDArray[np.bool_],
        lu_type: LandUseName,
        frac: float = 1.0,
        mode: Literal["replace", "add"] = "replace",
    ) -> None:
        """Set or add a land-use source fraction in the masked cells."""
        if not np.isfinite(frac) or not 0 <= frac <= 1:
            raise ValueError(f"Land-use fraction must be in [0, 1], got {frac}")
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match {self.shape}")
        if lu_type not in self.land_use:
            raise KeyError(
                f"Land-use type {lu_type!r} is not in {self.land_use_set.value!r}"
            )
        fraction = self.land_use[lu_type].lu_frac
        if mode == "replace":
            fraction[mask] = frac
            if frac == 1:
                for name, state in self.land_use.items():
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
            raise TypeError("LSM_output_dales requires a LandUseModification")
        self.set_type(geometry, modification.type, modification.frac, modification.mode)

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
            tile.tskin[tile.tskin <= 0.0] = -1
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


def iter_output_fields(writer: LSM_output_dales) -> Iterator[tuple[str, LSMArray]]:
    """Yield sorted DALES variable names and their backing arrays."""
    output: list[tuple[str, LSMArray]] = [
        ("t_soil", writer.soil.t_soil),
        ("index_soil", writer.soil.index_soil),
        ("theta_soil", writer.soil.theta_soil),
        ("cover_tot", writer.cover_tot),
        ("c_veg_tot", writer.c_veg_tot),
    ]
    for name, tile in writer.land_use_fields.items():
        output.extend(
            (f"{parameter}_{name}", tile.get(parameter))
            for parameter in LAND_USE_PARAMETERS
        )
        if tile.ags is not None:
            output.extend(
                (f"{parameter}_{name}", values)
                for parameter, values in tile.ags.items()
            )
    yield from sorted(output, key=lambda item: item[0])


def save_netcdf(writer: LSM_output_dales, output_path: str | Path, exp_id: int) -> None:
    """Write grid fields, fixed-width tile metadata, and optional LCZ data."""
    output_path = Path(output_path)
    with nc4.Dataset(output_path / f"lsm.inp_{exp_id:03d}.nc", "w") as dataset:
        for name, size in (
            ("x", writer.grid.itot),
            ("y", writer.grid.jtot),
            ("z", writer.soil_levels),
            ("nlu", writer.nlu),
            ("str1", 1),
            ("str3", 3),
            ("str32", 32),
        ):
            dataset.createDimension(name, size)
        dataset.createVariable("x", float, ("x",))[:] = writer.grid.xt
        dataset.createVariable("y", float, ("y",))[:] = writer.grid.yt
        field_names: list[str] = []
        for name, values in iter_output_fields(writer):
            dims = ("y", "x") if values.ndim == 2 else ("z", "y", "x")
            dataset.createVariable(name, float, dims)[:] = values
            field_names.append(name)
        writer.grid.set_cf_grid_mapping_netcdf(dataset, "crs", field_names)
        definitions = [state.definition for state in writer.land_use.values()]
        for name, width, texts in (
            ("luname", 32, [item.lu_long for item in definitions]),
            ("lushort", 3, [item.lu_short for item in definitions]),
            ("lveg", 1, ["True" if item.lveg else "False" for item in definitions]),
            ("laqu", 1, ["True" if item.laqu else "False" for item in definitions]),
            (
                "lslurb",
                1,
                ["True" if item.lslurb else "False" for item in definitions],
            ),
        ):
            matrix = np.zeros((writer.nlu, width), dtype="S1")
            for index, text in enumerate(texts):
                encoded = text.strip()[:width].encode("ascii", errors="replace")
                matrix[index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
            dataset.createVariable(name, "S1", ("nlu", f"str{width}"))[:] = matrix
        dataset.createVariable("ilu", int, ("nlu",))[:] = np.arange(1, writer.nlu + 1)
    if writer.LCZ_ds is not None:
        writer.grid.set_cf_grid_mapping_xr(
            writer.LCZ_ds, "crs", [str(name) for name in writer.LCZ_ds.data_vars]
        )
        writer.LCZ_ds.to_netcdf(output_path / f"lcz_ds_{exp_id:03d}.nc")
