"""Typed radiation modules with explicit scheme-specific namelist fields."""

import pathlib
from dataclasses import dataclass, field
from typing import Optional

from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.Radiation.backrad_profile import (
    BackradInterpolatedProfile,
    BackradPressureProfile,
    register_radiation_files,
)
from modular_dales.Surface.surface import SurfaceModule


@dataclass
class _RadiationTypedBase(simulation_module):
    """Shared behavior for typed radiation modules."""

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    timerad: int = field(
        default=60,
        metadata={
            "nml": "PHYSICS",
            "key": "timerad",
            "required": True,
            "serialize": True,
            "doc": "Radiation update interval in seconds.",
        },
    )
    surface_module: Optional["SurfaceModule"] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    backrad_profile: BackradPressureProfile | None = field(
        default=None,
        metadata={
            "serialize": True,
            "doc": "Optional pressure-based profile (Pa, K, kg/kg) used to generate backrad files.",
        },
    )
    backrad_source_file: pathlib.Path | None = field(
        default=None,
        metadata={
            "serialize": True,
            "doc": "Optional path to existing backrad.inp.* or backrad.inp.*.nc profile.",
        },
    )
    backrad_interpolated_profile: BackradInterpolatedProfile | None = field(
        default=None,
        metadata={
            "serialize": True,
            "doc": "Optional interpolated-profile style backrad specification.",
        },
    )

    def __post_init__(self):
        super().__init__(self.sim)

    def do_config(self):
        return None

    def prepare_calculation(self):
        if self.module_exists(SurfaceModule):
            self.surface_module = self.retrieve_module(SurfaceModule)
        else:
            raise ValueError(f"{self.module_name} requires a SurfaceModule.")

    def check_settings(self):
        return None

    def write_files(self):
        if self.iradiation != 0 and self.surface_module is not None:
            if getattr(self.surface_module, "albedoav", None) is None:
                raise ValueError(
                    f"{self.module_name}: albedoav must be set in surface config for radiation to work properly"
                )
        if self.iradiation != 0 and self.surface_module is None:
            raise ValueError(
                f"{self.module_name}: albedoav must be set in surface config for radiation to work properly"
            )

        register_radiation_files(self, self.iradiation)


@register_module
@dataclass
class NoRadiationModule(_RadiationTypedBase):
    """No-radiation scheme (iradiation=0)."""

    iradiation: int = field(
        default=0,
        init=False,
        metadata={
            "nml": "PHYSICS",
            "key": "IRADIATION",
            "required": True,
            "doc": "Radiation scheme selector 0: no radiation.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "NoRadiationModule"


@register_module
@dataclass
class ParameterizedRadiationModule(_RadiationTypedBase):
    """Parameterized radiation scheme (iradiation=2)."""

    iradiation: int = field(
        default=2,
        init=False,
        metadata={
            "nml": "PHYSICS",
            "key": "IRADIATION",
            "required": True,
            "doc": "Radiation scheme selector 2: parameterized radiation.",
        },
    )
    ssa: float | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "ssa",
            "doc": "Representative single scattering albedo (0 <= ssa <= 1).",
        },
    )
    laero: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "laero",
            "doc": "Use aerosol optical properties when true; cloud optical properties when false.",
        },
    )
    ide: int | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "ide",
            "doc": "Scalar index used for aerosols when laero=true.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "ParameterizedRadiationModule"


@register_module
@dataclass
class SurfaceLSMRadiationModule(_RadiationTypedBase):
    """Simple surface radiation for LSM (iradiation=3)."""

    iradiation: int = field(
        default=3,
        init=False,
        metadata={
            "nml": "PHYSICS",
            "key": "IRADIATION",
            "required": True,
            "doc": "Radiation scheme selector 3: simple surface radiation for LSM.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "SurfaceLSMRadiationModule"


@register_module
@dataclass
class RRTMGRadiationModule(_RadiationTypedBase):
    """RRTMG radiation scheme (iradiation=4)."""

    iradiation: int = field(
        default=4,
        init=False,
        metadata={
            "nml": "PHYSICS",
            "key": "IRADIATION",
            "required": True,
            "doc": "Radiation scheme selector 4: RRTMG.",
        },
    )
    ssa: float | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "ssa",
            "doc": "Representative single scattering albedo (0 <= ssa <= 1).",
        },
    )
    laero: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "laero",
            "doc": "Use aerosol optical properties when true; cloud optical properties when false.",
        },
    )
    ide: int | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "ide",
            "doc": "Scalar index used for aerosols when laero=true.",
        },
    )
    lCnstZenith: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "lCnstZenith",
            "doc": "Use a constant solar zenith angle.",
        },
    )
    ioverlap: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "ioverlap",
            "doc": "Cloud overlap method selector.",
        },
    )
    inflglw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "inflglw",
            "doc": "RRTMG longwave input selector.",
        },
    )
    iceflglw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "iceflglw",
            "doc": "Ice particle specification method for longwave.",
        },
    )
    liqflglw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "liqflglw",
            "doc": "Liquid water specification method for longwave.",
        },
    )
    inflgsw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "inflgsw",
            "doc": "RRTMG shortwave input selector.",
        },
    )
    iceflgsw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "iceflgsw",
            "doc": "Ice particle specification method for shortwave.",
        },
    )
    liqflgsw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "liqflgsw",
            "doc": "Liquid water specification method for shortwave.",
        },
    )
    iyear: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "iyear",
            "doc": "Simulation year used by radiation calculations.",
        },
    )
    ocean: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "ocean",
            "doc": "Enable ocean radiation treatment.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "RRTMGRadiationModule"


@register_module
@dataclass
class RteRrtmgpRadiationModule(_RadiationTypedBase):
    """RTE-RRTMGP radiation scheme (iradiation=5)."""

    iradiation: int = field(
        default=5,
        init=False,
        metadata={
            "nml": "PHYSICS",
            "key": "IRADIATION",
            "required": True,
            "doc": "Radiation scheme selector 5: RTE-RRTMGP.",
        },
    )
    ssa: float | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "ssa",
            "doc": "Representative single scattering albedo (0 <= ssa <= 1).",
        },
    )
    laero: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "laero",
            "doc": "Use aerosol optical properties when true; cloud optical properties when false.",
        },
    )
    ide: int | None = field(
        default=None,
        metadata={
            "nml": "NAMDE",
            "key": "ide",
            "doc": "Scalar index used for aerosols when laero=true.",
        },
    )
    lCnstZenith: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "lCnstZenith",
            "doc": "Use a constant solar zenith angle.",
        },
    )
    ioverlap: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "ioverlap",
            "doc": "Cloud overlap method selector.",
        },
    )
    inflglw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "inflglw",
            "doc": "RRTMG longwave input selector.",
        },
    )
    iceflglw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "iceflglw",
            "doc": "Ice particle specification method for longwave.",
        },
    )
    liqflglw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "liqflglw",
            "doc": "Liquid water specification method for longwave.",
        },
    )
    inflgsw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "inflgsw",
            "doc": "RRTMG shortwave input selector.",
        },
    )
    iceflgsw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "iceflgsw",
            "doc": "Ice particle specification method for shortwave.",
        },
    )
    liqflgsw: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "liqflgsw",
            "doc": "Liquid water specification method for shortwave.",
        },
    )
    iyear: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "iyear",
            "doc": "Simulation year used by radiation calculations.",
        },
    )
    ocean: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMRADIATION",
            "key": "ocean",
            "doc": "Enable ocean radiation treatment.",
        },
    )
    nbatch: int | None = field(
        default=None,
        metadata={
            "nml": "NAMRTERRTMGP",
            "key": "nbatch",
            "doc": "Number of column batches sent to RTE-RRTMGP kernels.",
        },
    )
    usepade: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMRTERRTMGP",
            "key": "usepade",
            "doc": "Use Pade coefficients for cloud optical properties.",
        },
    )
    doclearsky: bool | None = field(
        default=None,
        metadata={
            "nml": "NAMRTERRTMGP",
            "key": "doclearsky",
            "doc": "Enable clear-sky radiation diagnostics.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "RteRrtmgpRadiationModule"


@register_module
@dataclass
class UserRadiationModule(_RadiationTypedBase):
    """User-defined radiation scheme (iradiation=10)."""

    iradiation: int = field(
        default=10,
        init=False,
        metadata={
            "nml": "PHYSICS",
            "key": "IRADIATION",
            "required": True,
            "doc": "Radiation scheme selector 10: user-defined radiation behavior.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.module_name = "UserRadiationModule"
