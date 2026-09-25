"""Radiation module for solar and thermal radiation handling."""

from dataclasses import dataclass, field
import pathlib
import logging
from typing import Optional

from modular_dales.IO_helpers.external_data_cache import (
    cache_root,
)
from modular_dales.IO_helpers.dales_external_data import resolve_rrtmg_data_paths
from modular_dales.MODULE_REGISTRY import register_module
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.Radiation.backrad_profile import (
    BackradInterpolatedProfile,
    BackradPressureProfile,
    default_profile,
    profile_from_path,
    write_profile,
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

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    iradiation: Optional[int] = field(
        default=None, metadata={"nml": "PHYSICS", "key": "IRADIATION"}
    )
    # NAMDE parameters
    ssa: Optional[float] = field(default=None, metadata={"nml": "NAMDE", "key": "ssa"})
    laero: Optional[float] = field(
        default=None, metadata={"nml": "NAMDE", "key": "laero"}
    )
    ide: Optional[int] = field(default=None, metadata={"nml": "NAMDE", "key": "ide"})

    # NAMRADIATION parameters
    lCnstZenith: Optional[bool] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "lCnstZenith"}
    )
    ioverlap: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "ioverlap"}
    )
    inflglw: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "inflglw"}
    )
    iceflglw: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "iceflglw"}
    )
    liqflglw: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "liqflglw"}
    )
    inflgsw: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "inflgsw"}
    )
    iceflgsw: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "iceflgsw"}
    )
    liqflgsw: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "liqflgsw"}
    )
    iyear: Optional[int] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "iyear"}
    )
    ocean: Optional[bool] = field(
        default=None, metadata={"nml": "NAMRADIATION", "key": "ocean"}
    )

    # NAMRTERRMTGP parameters
    nbatch: Optional[int] = field(
        default=None, metadata={"nml": "NAMRTERRTMGP", "key": "nbatch"}
    )
    usepade: Optional[bool] = field(
        default=None, metadata={"nml": "NAMRTERRTMGP", "key": "usepade"}
    )
    doclearsky: Optional[bool] = field(
        default=None, metadata={"nml": "NAMRTERRTMGP", "key": "doclearsky"}
    )

    timerad: int = field(
        default=60,
        metadata={
            "nml": "PHYSICS",
            "key": "timerad",
            "required": True,
            "serialize": True,
        },
        init=True,
    )

    surface_module: Optional["SurfaceModule"] = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    backrad_profile: Optional[BackradPressureProfile] = field(
        default=None,
        metadata={
            "serialize": True,
            "doc": "Optional pressure-based profile (Pa, K, kg/kg) used to generate backrad files.",
        },
    )
    backrad_source_file: Optional[pathlib.Path] = field(
        default=None,
        metadata={
            "serialize": True,
            "doc": "Optional path to existing backrad.inp.* or backrad.inp.*.nc profile.",
        },
    )
    backrad_interpolated_profile: Optional[BackradInterpolatedProfile] = field(
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
        return None

    def prepare_calculation(self):
        """No additional preparation needed."""
        if self.module_exists(SurfaceModule):
            self.surface_module = self.retrieve_module(SurfaceModule)
        else:
            raise ValueError("RadiationModule requires a SurfaceModule.")

        return None

    def check_settings(self):
        """Validate constant fluxes settings."""
        pass

    def write_files(self):
        iradiation = self.iradiation or 0
        if iradiation == 1:
            raise ValueError(
                "RadiationModule: iradiation=1 is no longer supported. Use iradiation=4 or iradiation=5 instead."
            )
        if iradiation != 0 and self.surface_module is not None:
            if getattr(self.surface_module, "albedoav", None) is None:
                raise ValueError(
                    "RadiationModule: albedoav must be set in surface config for radiation to work properly"
                )
        if iradiation != 0 and self.surface_module is None:
            raise ValueError(
                "RadiationModule: albedoav must be set in surface config for radiation to work properly"
            )
        exp_id = self.exp_id

        profile_sources = sum(
            value is not None
            for value in (
                self.backrad_profile,
                self.backrad_source_file,
                self.backrad_interpolated_profile,
            )
        )
        if profile_sources > 1:
            raise ValueError(
                "RadiationModule: provide at most one of backrad_profile, backrad_source_file, or backrad_interpolated_profile."
            )

        selected_profile = self.backrad_profile
        if selected_profile is None and self.backrad_source_file is not None:
            selected_profile = profile_from_path(pathlib.Path(self.backrad_source_file))
        if selected_profile is None and self.backrad_interpolated_profile is not None:
            selected_profile = self.backrad_interpolated_profile.to_profile(
                template_profile=default_profile()
            )
        if selected_profile is None:
            selected_profile = default_profile()

        backrad_cache = cache_root("backrad", self.sim)
        backrad_cache.mkdir(parents=True, exist_ok=True)

        if iradiation == 4:
            backrad_nc = write_profile(
                selected_profile,
                backrad_cache / f"backrad.inp.{exp_id:03d}.nc",
            )
            self.sim.required_files[f"backrad.inp.{exp_id:03d}.nc"] = (
                backrad_nc.as_posix()
            )
            external = resolve_rrtmg_data_paths(self.sim)
            self.sim.required_files["rrtmg_lw.nc"] = external.rrtmg_lw.as_posix()
            self.sim.required_files["rrtmg_sw.nc"] = external.rrtmg_sw.as_posix()
        elif iradiation == 5:
            backrad_nc = write_profile(
                selected_profile,
                backrad_cache / f"backrad.inp.{exp_id:03d}.nc",
            )
            self.sim.required_files[f"backrad.inp.{exp_id:03d}.nc"] = (
                backrad_nc.as_posix()
            )
            external = resolve_rrtmg_data_paths(self.sim)
            self.sim.required_files["rrtmg_lw.nc"] = external.rrtmg_lw.as_posix()
            self.sim.required_files["rrtmg_sw.nc"] = external.rrtmg_sw.as_posix()
            for file in external.rrtmgp_data_dir.glob("*.nc"):
                self.sim.required_files[file.name] = file.as_posix()
        return None
