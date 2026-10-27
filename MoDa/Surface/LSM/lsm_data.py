"""Typed land-use definitions and grid fields used by the DALES LSM writer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int32]
type LSMArray = FloatArray | IntArray
type LandUseName = Literal[
    "cmf",
    "sg",
    "en",
    "dn",
    "db",
    "eb",
    "tg",
    "de",
    "tu",
    "ic",
    "sd",
    "bm",
    "es",
    "ds",
    "mf",
    "if",
    "rd",
    "wat",
    "ara",
    "crp",
    "fce",
    "fbd",
    "aqu",
    "brn",
    "sem",
    "grs",
    "urb",
    "lv",
    "hv",
    "bs",
    "aq",
    "ap",
    "bu",
    "ba",
    "fl",
    "fo",
    "ma",
    "oa",
    "oc",
    "po",
    "ra",
    "ry",
    "su",
    "wh",
]
type LandUseSelection = LandUseName | Literal["all"]
type LandUseParameter = Literal[
    "cover",
    "c_veg",
    "z0m",
    "z0h",
    "lai",
    "ar",
    "br",
    "lambda_s",
    "lambda_us",
    "rs_min",
    "gD",
    "tskin",
    "lutype",
]
type CoverParameter = Literal["cover", "c_veg"]
type AGSParameter = Literal[
    "gm25", "Ammax25", "f0", "alpha0", "co2_comp298", "T1gm", "T1Am"
]


@dataclass
class AGSFields:
    """Optional assimilation-conductance parameters on the horizontal grid."""

    gm25: FloatArray
    Ammax25: FloatArray
    f0: FloatArray
    alpha0: FloatArray
    co2_comp298: FloatArray
    T1gm: FloatArray
    T1Am: FloatArray

    def items(self) -> list[tuple[str, FloatArray]]:
        return [
            (item.name, cast(FloatArray, getattr(self, item.name)))
            for item in fields(self)
        ]


@dataclass(frozen=True, slots=True)
class LandUseDefinition:
    """Static metadata; missing IFS ids require an explicit caller mapping.

    Source ids are ordered, with the last id used as the default.
    An empty tuple means no source ids have been assigned.
    """

    lu_long: str
    lu_short: str
    ifs_id: int | None = None
    lveg: bool = False
    laqu: bool = False
    lslurb: bool = False
    lu_ids: tuple[int, ...] = ()
    bare_soil: bool = False


@dataclass
class LandUseState:
    """A definition and the mutable float64 arrays owned by one writer instance."""

    definition: LandUseDefinition
    lu_frac: FloatArray
    lu_domid: FloatArray


@dataclass
class SoilFields:
    t_soil: FloatArray
    index_soil: IntArray
    theta_soil: FloatArray

    def _set_profile(self, target: FloatArray, profile: Sequence[float]) -> None:
        values = np.asarray(profile, dtype=np.float64)
        levels = target.shape[0]
        if values.shape != (levels,):
            raise ValueError(f"Expected {levels} soil values, got {values.shape}")
        target[:] = values[:, None, None]

    def set_uniform_temperature(self, temperature: Sequence[float]) -> None:
        self._set_profile(self.t_soil, temperature)

    def set_uniform_moisture(self, moisture: Sequence[float]) -> None:
        self._set_profile(self.theta_soil, moisture)

    def set_temperature(self, temperature: FloatArray) -> None:
        if temperature.shape != self.t_soil.shape:
            raise ValueError(
                f"Incompatible array shape {temperature.shape}; expected {self.t_soil.shape}"
            )
        self.t_soil[:] = temperature

    def set_moisture(self, moisture: FloatArray) -> None:
        if moisture.shape != self.theta_soil.shape:
            raise ValueError(
                f"Incompatible array shape {moisture.shape}; expected {self.theta_soil.shape}"
            )
        self.theta_soil[:] = moisture

    @classmethod
    def zeros(cls, shape: tuple[int, int, int]) -> SoilFields:
        return cls(
            t_soil=np.zeros(shape, dtype=np.float64),
            index_soil=np.zeros(shape, dtype=np.int32),
            theta_soil=np.zeros(shape, dtype=np.float64),
        )


@dataclass
class LandUseFields:
    """DALES parameters for a single land-use tile, all on the (y, x) grid."""

    cover: FloatArray
    c_veg: FloatArray
    z0m: FloatArray
    z0h: FloatArray
    lai: FloatArray
    ar: FloatArray
    br: FloatArray
    lambda_s: FloatArray
    lambda_us: FloatArray
    rs_min: FloatArray
    gD: FloatArray
    tskin: FloatArray
    lutype: FloatArray
    ags: AGSFields | None = None

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> LandUseFields:
        return cls(
            cover=np.zeros(shape, dtype=np.float64),
            c_veg=np.zeros(shape, dtype=np.float64),
            z0m=np.zeros(shape, dtype=np.float64),
            z0h=np.zeros(shape, dtype=np.float64),
            lai=np.zeros(shape, dtype=np.float64),
            ar=np.zeros(shape, dtype=np.float64),
            br=np.zeros(shape, dtype=np.float64),
            lambda_s=np.zeros(shape, dtype=np.float64),
            lambda_us=np.zeros(shape, dtype=np.float64),
            rs_min=np.zeros(shape, dtype=np.float64),
            gD=np.zeros(shape, dtype=np.float64),
            tskin=np.zeros(shape, dtype=np.float64),
            lutype=np.zeros(shape, dtype=np.float64),
        )

    def get(self, parameter: LandUseParameter) -> FloatArray:
        return cast(FloatArray, getattr(self, parameter))

    def set(self, parameter: LandUseParameter, values: FloatArray) -> None:
        setattr(self, parameter, values)


LAND_USE_PARAMETERS: tuple[LandUseParameter, ...] = (
    "cover",
    "c_veg",
    "z0m",
    "z0h",
    "lai",
    "ar",
    "br",
    "lambda_s",
    "lambda_us",
    "rs_min",
    "gD",
    "tskin",
    "lutype",
)
