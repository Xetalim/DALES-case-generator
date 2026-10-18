"""Radiation module for solar and thermal radiation handling."""

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from modular_dales.modular.namelist import nml_field
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.Radiation.backrad_profile import (
    BackradInterpolatedProfile,
    BackradPressureProfile,
    register_radiation_files,
)
from modular_dales.Surface.surface import SurfaceModule

logger = logging.getLogger(__name__)


@register_module
@dataclass
class RadiationModule(simulation_module):
    """Radiation simulation module.

    Possible values for `iradiation`:
        0: no radiation
        2: parameterized radiation
        3: simple surface radiation for land surface model
        4: RRTMG radiation
        5: RTE-RRTMGP radiation
        10: user specified radiation

    Args:
        sim: Parent dales_simulation instance
        iradiation: Radiation type
        ssa: Representative single scattering albedo (0 <= x <= 1)
        ide: Scalar field used as aerosols if laero set to .true.
        laero: .true. for aerosols, .false. for clouds
        lCnstZenith: Switch to apply a fixed solar zenith angle
        ioverlap: Flag for cloud overlap method
        inflglw: Flag for RRTMG longwave input
        iceflglw: Flag for ice particle specification in longwave
        liqflglw: Flag for effect of liquid water in longwave
        inflgsw: Flag for RRTMG shortwave input
        iceflgsw: Flag for ice particle specification in shortwave
        liqflgsw: Flag for effect of liquid water in shortwave
        iyear: Year of the simulation
        ocean: Switch to calculate radiation over ocean
        nbatch: Number of batch of vertical columns sent to RTE-RRTMGP routines
        usepade: Use Pade coefficients for cloud optical properties instead of lookup tables
        doclearsky: Use clear sky radiation in the calculation
    """

    iradiation: int | None = nml_field(
        default=None, section="PHYSICS", key="IRADIATION"
    )
    # NAMDE parameters
    ssa: float | None = nml_field(default=None, section="NAMDE", key="ssa")
    laero: float | None = nml_field(default=None, section="NAMDE", key="laero")
    ide: int | None = nml_field(default=None, section="NAMDE", key="ide")

    # NAMRADIATION parameters
    lCnstZenith: bool | None = nml_field(
        default=None, section="NAMRADIATION", key="lCnstZenith"
    )
    ioverlap: int | None = nml_field(
        default=None, section="NAMRADIATION", key="ioverlap"
    )
    inflglw: int | None = nml_field(default=None, section="NAMRADIATION", key="inflglw")
    iceflglw: int | None = nml_field(
        default=None, section="NAMRADIATION", key="iceflglw"
    )
    liqflglw: int | None = nml_field(
        default=None, section="NAMRADIATION", key="liqflglw"
    )
    inflgsw: int | None = nml_field(default=None, section="NAMRADIATION", key="inflgsw")
    iceflgsw: int | None = nml_field(
        default=None, section="NAMRADIATION", key="iceflgsw"
    )
    liqflgsw: int | None = nml_field(
        default=None, section="NAMRADIATION", key="liqflgsw"
    )
    iyear: int | None = nml_field(default=None, section="NAMRADIATION", key="iyear")
    ocean: bool | None = nml_field(default=None, section="NAMRADIATION", key="ocean")

    # NAMRTERRMTGP parameters
    nbatch: int | None = nml_field(default=None, section="NAMRTERRTMGP", key="nbatch")
    usepade: bool | None = nml_field(
        default=None, section="NAMRTERRTMGP", key="usepade"
    )
    doclearsky: bool | None = nml_field(
        default=None, section="NAMRTERRTMGP", key="doclearsky"
    )

    timerad: int = nml_field(
        default=60,
        section="PHYSICS",
        key="timerad",
        required=True,
        metadata={"serialize": True},
        init=True,
    )

    surface_module: Optional["SurfaceModule"] = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
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
        self.module_name = "RadiationModule"

    def do_config(self):
        """Ensure radiation configuration is set."""
        return

    def prepare_calculation(self):
        """No additional preparation needed."""
        if self.module_exists(SurfaceModule):
            self.surface_module = self.retrieve_module(SurfaceModule)
        else:
            raise ValueError("RadiationModule requires a SurfaceModule.")

    def check_settings(self):
        """Validate constant fluxes settings."""

    def write_files(self):
        iradiation = self.iradiation or 0
        if iradiation == 1:
            raise ValueError(
                "RadiationModule: iradiation=1 is not supported. Use iradiation=4 or iradiation=5 instead."
            )
        if iradiation != 0 and getattr(self.surface_module, "albedoav", None) is None:
            raise ValueError(
                "RadiationModule: albedoav must be set in surface config for radiation to work properly"
            )
        if iradiation != 0 and self.surface_module is None:
            raise ValueError(
                "RadiationModule: albedoav must be set in surface config for radiation to work properly"
            )
        register_radiation_files(self, iradiation)
