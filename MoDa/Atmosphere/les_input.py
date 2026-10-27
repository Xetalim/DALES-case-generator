"""Schema for external LES input, retained as a regular xarray Dataset.

Atmospheric fields use (time, z), soil fields (time, zs), and radiation
fields (time, lay) or (time, lev). time_sec(time) contains simulation seconds;
time itself may contain datetime labels. Dimension order is unrestricted.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class FieldSpec:
    """Expected dimensions, canonical units and physical meaning of a field.

    Units document the contract; validation does not convert units or require
    matching unit strings. Required fields must be present in every provider.
    """

    dims: tuple[str, ...]
    units: str
    description: str
    required: bool = False


LES_INPUT_SCHEMA: Mapping[str, FieldSpec] = MappingProxyType(
    {
        "z": FieldSpec(
            ("z",), "m", "Atmospheric full-level height matching grid.zt", True
        ),
        "zs": FieldSpec(
            ("zs",), "m", "Soil full-level height; LS2D uses negative depths"
        ),
        "lay": FieldSpec(("lay",), "1", "Radiation full-level index"),
        "lev": FieldSpec(("lev",), "1", "Radiation half-level index"),
        "time_sec": FieldSpec(("time",), "s", "Seconds since simulation start", True),
        "u": FieldSpec(("time", "z"), "m s-1", "Eastward wind", True),
        "v": FieldSpec(("time", "z"), "m s-1", "Northward wind", True),
        "thl": FieldSpec(
            ("time", "z"), "K", "Liquid water potential temperature", True
        ),
        "qt": FieldSpec(("time", "z"), "kg kg-1", "Total specific humidity", True),
        "ug": FieldSpec(("time", "z"), "m s-1", "Eastward geostrophic wind"),
        "vg": FieldSpec(("time", "z"), "m s-1", "Northward geostrophic wind"),
        "wls": FieldSpec(("time", "z"), "m s-1", "Large-scale vertical velocity"),
        "p": FieldSpec(("time", "z"), "Pa", "Air pressure"),
        "o3": FieldSpec(("time", "z"), "ppmv", "Ozone volume mixing ratio"),
        "dtthl_advec": FieldSpec(("time", "z"), "K s-1", "Advective thl tendency"),
        "dtqt_advec": FieldSpec(
            ("time", "z"), "kg kg-1 s-1", "Advective humidity tendency"
        ),
        "dtu_advec": FieldSpec(
            ("time", "z"), "m s-2", "Advective eastward wind tendency"
        ),
        "dtv_advec": FieldSpec(
            ("time", "z"), "m s-2", "Advective northward wind tendency"
        ),
        "ps": FieldSpec(("time",), "Pa", "Surface pressure"),
        "ts": FieldSpec(("time",), "K", "Skin temperature"),
        "sst": FieldSpec(("time",), "K", "Sea surface temperature"),
        "wth": FieldSpec(("time",), "K m s-1", "Kinematic sensible heat flux"),
        "wq": FieldSpec(("time",), "kg kg-1 m s-1", "Kinematic moisture flux"),
        "z0m": FieldSpec(("time",), "m", "Momentum roughness length"),
        "z0h": FieldSpec(("time",), "m", "Scalar roughness length"),
        "t_soil": FieldSpec(("time", "zs"), "K", "Soil temperature"),
        "theta_soil": FieldSpec(("time", "zs"), "m3 m-3", "Volumetric soil moisture"),
        "type_soil": FieldSpec((), "1", "ECMWF soil type, Fortran indexing"),
        "p_lay": FieldSpec(("time", "lay"), "Pa", "Radiation full-level pressure"),
        "t_lay": FieldSpec(("time", "lay"), "K", "Radiation full-level temperature"),
        "h2o_lay": FieldSpec(
            ("time", "lay"),
            "1",
            "Water vapour volume mixing ratio, not specific humidity",
        ),
        "o3_lay": FieldSpec(
            ("time", "lay"), "ppmv", "Radiation ozone volume mixing ratio"
        ),
        "z_lay": FieldSpec(("time", "lay"), "m", "Radiation full-level height"),
        "p_lev": FieldSpec(("time", "lev"), "Pa", "Radiation half-level pressure"),
        "t_lev": FieldSpec(("time", "lev"), "K", "Radiation half-level temperature"),
        "z_lev": FieldSpec(("time", "lev"), "m", "Radiation half-level height"),
        "type_low_veg": FieldSpec((), "1", "ECMWF low vegetation type"),
        "type_high_veg": FieldSpec((), "1", "ECMWF high vegetation type"),
        "root_frac_low_veg": FieldSpec(("zs",), "1", "Low vegetation root fraction"),
        "root_frac_high_veg": FieldSpec(("zs",), "1", "High vegetation root fraction"),
        "lai_low_veg": FieldSpec(("time",), "1", "Low vegetation leaf area index"),
        "lai_high_veg": FieldSpec(("time",), "1", "High vegetation leaf area index"),
        "c_low_veg": FieldSpec(("time",), "1", "Low vegetation coverage fraction"),
        "c_high_veg": FieldSpec(("time",), "1", "High vegetation coverage fraction"),
    }
)

type LESInput = Annotated[xr.Dataset, "LES_INPUT_SCHEMA"]
"""An xr.Dataset satisfying LES_INPUT_SCHEMA; checked by validate_les_input."""


def validate_les_input(
    dataset: LESInput,
    *,
    z: Iterable[float],
    nudging: bool = False,
    tracers: Iterable[str] = (),
) -> LESInput:
    """Validate the provider contract without loading its atmospheric data.

    Only coordinate values are inspected eagerly. Unknown variables are
    allowed; requested tracer sources must have dimensions (time, z).
    Returns the original Dataset, preserving lazy arrays and metadata.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("LES input must be an xr.Dataset")
    required = {name for name, spec in LES_INPUT_SCHEMA.items() if spec.required}
    if nudging:
        required.update(("u", "v", "thl", "qt"))
    tracer_names = set(tracers)
    required.update(tracer_names)
    missing = required.difference(dataset.variables)
    if missing:
        raise ValueError(
            f"LES input is missing required fields: {', '.join(sorted(missing))}"
        )

    for name, spec in LES_INPUT_SCHEMA.items():
        if name in dataset and set(dataset[name].dims) != set(spec.dims):
            raise ValueError(
                f"LES input '{name}' has dimensions {dataset[name].dims}; expected {spec.dims}"
            )
    for name in tracer_names:
        if set(dataset[name].dims) != {"time", "z"}:
            raise ValueError(
                f"LES input tracer '{name}' must have dimensions (time, z)"
            )

    times = np.asarray(dataset["time_sec"].values, dtype=float)
    if times.size < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(
            "LES input time_sec must contain at least two finite, strictly increasing times"
        )
    if "z" not in dataset.coords or dataset["z"].dims != ("z",):
        raise ValueError("LES input requires a one-dimensional 'z' coordinate")
    heights = np.asarray(dataset["z"].values, dtype=float)
    expected = np.asarray(list(z), dtype=float)
    if (
        heights.shape != expected.shape
        or not np.isfinite(heights).all()
        or not np.allclose(heights, expected)
    ):
        raise ValueError("LES input z coordinate must match the simulation grid.zt")
    radiation = {"p_lay", "t_lay", "h2o_lay"}
    present = radiation.intersection(dataset.data_vars)
    if present and present != radiation:
        raise ValueError(
            "LES input radiation sounding requires p_lay, t_lay and h2o_lay together"
        )
    return dataset
