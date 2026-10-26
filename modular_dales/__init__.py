"""Public API for the modular_dales package.

This module exposes the main high-level classes and helpers that are
intended for users of the library, while still keeping the internal
package structure available for advanced use.
"""

from .Atmosphere import (
    AtmosphereModule,
    AtmosphericProfile,
    HarmonieAtmosphereModule,
    InterpolatedProfile,
    LS2DAtmosphereModule,
)
from .Configuration.defaultnamelist import DefaultNamelistModule
from .Configuration.output_modules import (
    CapeModule,
    ColumnStatisticsOutputModule,
    CrossSectionOutputModule,
    EasyOutputModule,
    FielddumpModule,
    LSMCrossModule,
    RadfieldModule,
    SamplingModule,
    StatsModule,
    TimestatModule,
    VirtualMeasurementOutputModule,
)
from .Configuration.run_and_time import RunModule, TimeModule
from .Configuration.settings_modules import (
    GeneralPhysicsModule,
    LateralSpongeModule,
    SprayingModule,
)
from .Emission.emission import (
    EmissionModule,
    EmissionPointSource,
    EmissionTracer,
)
from .Geometry.geometry_modification import (
    AllGeometry,
    CircleIdxGeometry,
    CircleRealGeometry,
    FuncGeometry,
    MaskGeometry,
    ModifierClass,
    RectangleIdxGeometry,
    RectangleRealGeometry,
)
from .Geometry.grids import GridDales, GridDalesOpenBC
from .IBM.ibm import (
    FromAHN,
    FromGlobalDEM,
    IBMModification,
    IBMModifications,
    IBMModule,
)
from .LBC import (
    KNMIHarmonieForecastDownloadModule,
    Nest_in_AtmosphereProfiles,
    Nest_in_Dales,
    Nest_in_Harmonie,
    Nest_in_KNMI,
    NestingTopology,
    OpenBoundaryConfig,
    do_openboundary,
)
from .logging_wrapper import logwrap, setup_logging
from .modular.dales_simulation import dales_simulation
from .modular.simulation_module import simulation_module
from .modular.time_dependent import (
    TimedependentModule,
)
from .modular.time_dependent_scalars import TimeDependentScalar
from .module_registry import (
    module_registry,
    register_module,
)
from .Radiation.backrad_profile import (
    BackradInterpolatedProfile,
    BackradPressureProfile,
)
from .Radiation.radiation import RadiationModule
from .Radiation.radiation_types import (
    NoRadiationModule,
    ParameterizedRadiationModule,
    RRTMGRadiationModule,
    RteRrtmgpRadiationModule,
    SurfaceLSMRadiationModule,
    UserRadiationModule,
)
from .Surface.LSM.base import BaseLSMModule
from .Surface.LSM.homogeneous import LSMHomogeneousModule
from .Surface.LSM.lsm_module import (
    LSMModule,
)
from .Surface.LSM.lsm_sources import FromBofek, FromLCZ, FromTop10
from .Surface.LSM.lu_modifications import LandUseModification, LandUseModifications
from .Surface.LSM.modular_temps_moisture import (
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
)
from .Surface.LSM.SLuRB.slurb import (
    SLURBModification,
    SLURBModifications,
    SLURBModule,
    SLURBVariableModification,
)
from .Surface.surface import (
    ConstantFluxesModule,
    ConstantFluxesWithShearModule,
    ConstantSurfaceTemperatureModule,
    SurfaceModule,
)
from .vars import VariableDefinition, get_all_vars

__all__ = [
    "AllGeometry",
    # AtmosphereModule, AtmosphericProfile, InterpolatedProfile
    "AtmosphereModule",
    "AtmosphericProfile",
    "BackradInterpolatedProfile",
    "BackradPressureProfile",
    "BaseLSMModule",
    "CapeModule",
    "CircleIdxGeometry",
    "CircleRealGeometry",
    "ColumnStatisticsOutputModule",
    "ConstantFluxesModule",
    "ConstantFluxesWithShearModule",
    "ConstantSurfaceTemperatureModule",
    "CrossSectionOutputModule",
    # Configuration modules
    "DefaultNamelistModule",
    "EasyOutputModule",
    # Emissions
    "EmissionModule",
    "EmissionPointSource",
    "EmissionTracer",
    "FielddumpModule",
    "FromAHN",
    "FromBofek",
    "FromGlobalDEM",
    "FromLCZ",
    "FromTop10",
    "FuncGeometry",
    "GeneralPhysicsModule",
    # Geometry
    "GridDales",
    "GridDalesOpenBC",
    "HarmonieAtmosphereModule",
    "IBMModification",
    "IBMModifications",
    # IBM
    "IBMModule",
    "InterpolatedProfile",
    "KNMIHarmonieForecastDownloadModule",
    "LS2DAtmosphereModule",
    "LSMCrossModule",
    "LSMHomogeneousModule",
    "LSMModule",
    "LandUseModification",
    "LandUseModifications",
    "LateralSpongeModule",
    "MaskGeometry",
    "ModifierClass",
    "Nest_in_AtmosphereProfiles",
    "Nest_in_Dales",
    "Nest_in_Harmonie",
    "Nest_in_KNMI",
    "NestingTopology",
    "NoRadiationModule",
    "OpenBoundaryConfig",
    "ParameterizedRadiationModule",
    "RRTMGRadiationModule",
    "RadfieldModule",
    # Radiation
    "RadiationModule",
    "RectangleIdxGeometry",
    "RectangleRealGeometry",
    "RteRrtmgpRadiationModule",
    "RunModule",
    "SLURBModification",
    "SLURBModifications",
    "SLURBModule",
    "SLURBVariableModification",
    "SamplingModule",
    "SprayingModule",
    "StatsModule",
    "SurfaceLSMRadiationModule",
    # Surface and LSM
    "SurfaceModule",
    "TimeDependentScalar",
    "TimeModule",
    "TimedependentModule",
    "TimestatModule",
    "UniformSkinTemperature",
    "UniformSoilMoisture",
    "UniformSoilTemperature",
    "UserRadiationModule",
    # LBC / nesting
    # variables
    "VariableDefinition",
    "VaryingSkinTemperature",
    "VaryingSoilMoisture",
    "VaryingSoilTemperature",
    "VirtualMeasurementOutputModule",
    # Core simulation framework
    "dales_simulation",
    # Lateral boundary conditions
    "do_openboundary",
    "get_all_vars",
    "logwrap",
    # Registry and logging helpers
    "module_registry",
    "register_module",
    "setup_logging",
    "simulation_module",
]
