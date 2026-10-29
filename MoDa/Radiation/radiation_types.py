"""Typed radiation modules with explicit scheme-specific namelist fields."""

import pathlib
from dataclasses import dataclass, field
from typing import Optional

from moda.Core.namelist import nml_field
from moda.modular.simulation_module import simulation_module
from moda.module_registry import register_module
from moda.Radiation.backrad_profile import (
    BackradInterpolatedProfile,
    BackradPressureProfile,
    register_radiation_files,
)
from moda.Surface.surfacemodules import SurfaceModule


@dataclass
class _RadiationTypedBase(simulation_module):
    """Shared behavior for typed radiation modules."""

    iradiation: int

    timerad: int = nml_field(
        default=60,
        section="PHYSICS",
        key="timerad",
        required=True,
        description="Radiation update interval in seconds.",
        metadata={"serialize": True},
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

    def prepare_calculation(self) -> None:
        if self.module_exists(SurfaceModule):
            self.surface_module = self.retrieve_module(SurfaceModule)
        else:
            raise ValueError(f"{self.module_name} requires a SurfaceModule.")

    def write_files(self) -> None:
        if (
            self.iradiation != 0
            and getattr(self.surface_module, "albedoav", None) is None
        ):
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

    iradiation: int = nml_field(
        default=0,
        init=False,
        section="PHYSICS",
        key="IRADIATION",
        required=True,
        description="Radiation scheme selector 0: no radiation.",
    )

    module_name: str = field(
        default="NoRadiationModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class ParameterizedRadiationModule(_RadiationTypedBase):
    """Parameterized radiation scheme (iradiation=2)."""

    iradiation: int = nml_field(
        default=2,
        init=False,
        section="PHYSICS",
        key="IRADIATION",
        required=True,
        description="Radiation scheme selector 2: parameterized radiation.",
    )
    ssa: float | None = nml_field(
        default=None,
        section="NAMDE",
        key="ssa",
        description="Representative single scattering albedo (0 <= ssa <= 1).",
    )
    laero: bool | None = nml_field(
        default=None,
        section="NAMDE",
        key="laero",
        description="Use aerosol optical properties when true; cloud optical properties when false.",
    )
    ide: int | None = nml_field(
        default=None,
        section="NAMDE",
        key="ide",
        description="Scalar index used for aerosols when laero=true.",
    )

    module_name: str = field(
        default="ParameterizedRadiationModule",
        init=False,
        metadata={"serialize": False},
    )


@register_module
@dataclass
class SurfaceLSMRadiationModule(_RadiationTypedBase):
    """Simple surface radiation for LSM (iradiation=3)."""

    iradiation: int = nml_field(
        default=3,
        init=False,
        section="PHYSICS",
        key="IRADIATION",
        required=True,
        description="Radiation scheme selector 3: simple surface radiation for LSM.",
    )

    module_name: str = field(
        default="SurfaceLSMRadiationModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class RRTMGRadiationModule(_RadiationTypedBase):
    """RRTMG radiation scheme (iradiation=4)."""

    iradiation: int = nml_field(
        default=4,
        init=False,
        section="PHYSICS",
        key="IRADIATION",
        required=True,
        description="Radiation scheme selector 4: RRTMG.",
    )
    ssa: float | None = nml_field(
        default=None,
        section="NAMDE",
        key="ssa",
        description="Representative single scattering albedo (0 <= ssa <= 1).",
    )
    laero: bool | None = nml_field(
        default=None,
        section="NAMDE",
        key="laero",
        description="Use aerosol optical properties when true; cloud optical properties when false.",
    )
    ide: int | None = nml_field(
        default=None,
        section="NAMDE",
        key="ide",
        description="Scalar index used for aerosols when laero=true.",
    )
    lCnstZenith: bool | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="lCnstZenith",
        description="Use a constant solar zenith angle.",
    )
    ioverlap: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="ioverlap",
        description="Cloud overlap method selector.",
    )
    inflglw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="inflglw",
        description="RRTMG longwave input selector.",
    )
    iceflglw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="iceflglw",
        description="Ice particle specification method for longwave.",
    )
    liqflglw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="liqflglw",
        description="Liquid water specification method for longwave.",
    )
    inflgsw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="inflgsw",
        description="RRTMG shortwave input selector.",
    )
    iceflgsw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="iceflgsw",
        description="Ice particle specification method for shortwave.",
    )
    liqflgsw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="liqflgsw",
        description="Liquid water specification method for shortwave.",
    )
    iyear: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="iyear",
        description="Simulation year used by radiation calculations.",
    )
    ocean: bool | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="ocean",
        description="Enable ocean radiation treatment.",
    )

    module_name: str = field(
        default="RRTMGRadiationModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class RteRrtmgpRadiationModule(_RadiationTypedBase):
    """RTE-RRTMGP radiation scheme (iradiation=5)."""

    iradiation: int = nml_field(
        default=5,
        init=False,
        section="PHYSICS",
        key="IRADIATION",
        required=True,
        description="Radiation scheme selector 5: RTE-RRTMGP.",
    )
    ssa: float | None = nml_field(
        default=None,
        section="NAMDE",
        key="ssa",
        description="Representative single scattering albedo (0 <= ssa <= 1).",
    )
    laero: bool | None = nml_field(
        default=None,
        section="NAMDE",
        key="laero",
        description="Use aerosol optical properties when true; cloud optical properties when false.",
    )
    ide: int | None = nml_field(
        default=None,
        section="NAMDE",
        key="ide",
        description="Scalar index used for aerosols when laero=true.",
    )
    lCnstZenith: bool | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="lCnstZenith",
        description="Use a constant solar zenith angle.",
    )
    ioverlap: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="ioverlap",
        description="Cloud overlap method selector.",
    )
    inflglw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="inflglw",
        description="RRTMG longwave input selector.",
    )
    iceflglw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="iceflglw",
        description="Ice particle specification method for longwave.",
    )
    liqflglw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="liqflglw",
        description="Liquid water specification method for longwave.",
    )
    inflgsw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="inflgsw",
        description="RRTMG shortwave input selector.",
    )
    iceflgsw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="iceflgsw",
        description="Ice particle specification method for shortwave.",
    )
    liqflgsw: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="liqflgsw",
        description="Liquid water specification method for shortwave.",
    )
    iyear: int | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="iyear",
        description="Simulation year used by radiation calculations.",
    )
    ocean: bool | None = nml_field(
        default=None,
        section="NAMRADIATION",
        key="ocean",
        description="Enable ocean radiation treatment.",
    )
    nbatch: int | None = nml_field(
        default=None,
        section="NAMRTERRTMGP",
        key="nbatch",
        description="Number of column batches sent to RTE-RRTMGP kernels.",
    )
    usepade: bool | None = nml_field(
        default=None,
        section="NAMRTERRTMGP",
        key="usepade",
        description="Use Pade coefficients for cloud optical properties.",
    )
    doclearsky: bool | None = nml_field(
        default=None,
        section="NAMRTERRTMGP",
        key="doclearsky",
        description="Enable clear-sky radiation diagnostics.",
    )

    module_name: str = field(
        default="RteRrtmgpRadiationModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class UserRadiationModule(_RadiationTypedBase):
    """User-defined radiation scheme (iradiation=10)."""

    iradiation: int = nml_field(
        default=10,
        init=False,
        section="PHYSICS",
        key="IRADIATION",
        required=True,
        description="Radiation scheme selector 10: user-defined radiation behavior.",
    )

    module_name: str = field(
        default="UserRadiationModule", init=False, metadata={"serialize": False}
    )
