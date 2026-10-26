"""Serialize typed land-surface fields and tile metadata in DALES format."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4 as nc4
import numpy as np

from modular_dales.Surface.LSM.lsm_data import LAND_USE_PARAMETERS, LSMArray

if TYPE_CHECKING:
    from modular_dales.Surface.LSM.lsm_output import LSM_output_dales


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
