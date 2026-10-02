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
from .Geometry.GridDales import GridDales, GridDalesOpenBC
from .IBM.IBM import (
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
from .MODULE_REGISTRY import (
    MODULE_REGISTRY,
    SINGLETON_REGISTRY,
    register_module,
    register_singleton,
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
from .Surface.LSM.LSM import (
    AGSParameters,
    FromBofek,
    FromLCZ,
    FromTop10,
    LandUseModification,
    LandUseModifications,
    LSMModule,
)
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
    # Core simulation framework
    "dales_simulation",
    "simulation_module",
    "TimeDependentScalar",
    "TimedependentModule",
    # Configuration modules
    "DefaultNamelistModule",
    "RunModule",
    "TimeModule",
    "GeneralPhysicsModule",
    "SprayingModule",
    "LateralSpongeModule",
    "EasyOutputModule",
    "ColumnStatisticsOutputModule",
    "VirtualMeasurementOutputModule",
    "CapeModule",
    "LSMCrossModule",
    "TimestatModule",
    "StatsModule",
    "RadfieldModule",
    "CrossSectionOutputModule",
    "FielddumpModule",
    "SamplingModule",
    # Geometry
    "GridDales",
    "GridDalesOpenBC",
    "ModifierClass",
    "AllGeometry",
    "CircleRealGeometry",
    "FuncGeometry",
    "RectangleRealGeometry",
    "RectangleIdxGeometry",
    "CircleIdxGeometry",
    "MaskGeometry",
    # AtmosphereModule, AtmosphericProfile, InterpolatedProfile
    "AtmosphereModule",
    "AtmosphericProfile",
    "InterpolatedProfile",
    "HarmonieAtmosphereModule",
    "LS2DAtmosphereModule",
    # Surface and LSM
    "SurfaceModule",
    "ConstantFluxesModule",
    "ConstantFluxesWithShearModule",
    "ConstantSurfaceTemperatureModule",
    "LSMModule",
    "BaseLSMModule",
    "LSMHomogeneousModule",
    "LandUseModification",
    "LandUseModifications",
    "FromLCZ",
    "FromTop10",
    "FromBofek",
    "AGSParameters",
    "UniformSkinTemperature",
    "UniformSoilTemperature",
    "UniformSoilMoisture",
    "VaryingSkinTemperature",
    "VaryingSoilTemperature",
    "VaryingSoilMoisture",
    "SLURBModule",
    "SLURBModification",
    "SLURBVariableModification",
    "SLURBModifications",
    # Lateral boundary conditions
    "do_openboundary",
    "Nest_in_Dales",
    "Nest_in_AtmosphereProfiles",
    "OpenBoundaryConfig",
    "NestingTopology",
    "KNMIHarmonieForecastDownloadModule",
    # Emissions
    "EmissionModule",
    "EmissionTracer",
    "EmissionPointSource",
    # Radiation
    "RadiationModule",
    "BackradPressureProfile",
    "BackradInterpolatedProfile",
    "NoRadiationModule",
    "ParameterizedRadiationModule",
    "SurfaceLSMRadiationModule",
    "RRTMGRadiationModule",
    "RteRrtmgpRadiationModule",
    "UserRadiationModule",
    # IBM
    "IBMModule",
    "IBMModification",
    "IBMModifications",
    "FromAHN",
    "FromGlobalDEM",
    # LBC / nesting
    # variables
    "VariableDefinition",
    "get_all_vars",
    # Registry and logging helpers
    "MODULE_REGISTRY",
    "SINGLETON_REGISTRY",
    "register_module",
    "register_singleton",
    "logwrap",
    "setup_logging",
]
