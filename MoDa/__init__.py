"""Core simulation API and domain namespaces.

Use ``import MoDa`` for core modules and ``MoDa.Surface``,
``MoDa.Atmosphere``, or the other domain namespaces for specialist tools.
Historical top-level class attributes remain available for compatibility;
``__all__`` advertises the core API and domain namespaces.
"""

from .Atmosphere import (
    AtmosphereModule,
    HarmonieAtmosphereModule,
    LS2DAtmosphereModule,
)
from .Atmosphere.Profile import AtmosphericProfile, InterpolatedProfile
from .Configuration import (
    CapeModule,
    ColumnStatisticsOutputModule,
    CrossSectionOutputModule,
    DefaultNamelistModule,
    EasyOutputModule,
    FielddumpModule,
    GeneralPhysicsModule,
    LateralSpongeModule,
    LSMCrossModule,
    RadfieldModule,
    RunModule,
    SamplingModule,
    SprayingModule,
    StatsModule,
    TimeModule,
    TimestatModule,
    VirtualMeasurementOutputModule,
)
from .Emission import (
    EmissionModule,
    EmissionPointSource,
    EmissionTracer,
)
from .Geometry import (
    AllGeometry,
    CircleIdxGeometry,
    CircleRealGeometry,
    FuncGeometry,
    GridDales,
    GridDalesOpenBC,
    MaskGeometry,
    ModifierClass,
    RectangleIdxGeometry,
    RectangleRealGeometry,
)
from .IBM import (
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
    OpenBoundaryModule,
)
from .logging_wrapper import logwrap, setup_logging
from .modular.dales_simulation import dales_simulation
from .modular.simulation_module import set_nml_section, simulation_module
from .modular.time_dependent import (
    TimedependentModule,
)
from .modular.time_dependent_scalars import TimeDependentScalar
from .module_registry import (
    module_registry,
    register_module,
)
from .Radiation import (
    BackradInterpolatedProfile,
    BackradPressureProfile,
    NoRadiationModule,
    ParameterizedRadiationModule,
    RadiationModule,
    RRTMGRadiationModule,
    RteRrtmgpRadiationModule,
    SurfaceLSMRadiationModule,
    UserRadiationModule,
)
from .Surface import (
    ConstantFluxesModule,
    ConstantFluxesWithShearModule,
    ConstantSurfaceTemperatureModule,
    SurfaceModule,
)
from .Surface.LSM import (
    BaseLSMModule,
    FromBofek,
    FromLCZ,
    FromTop10,
    LandUseModification,
    LandUseModifications,
    LSMHomogeneousModule,
    LSMModule,
    SLURBModification,
    SLURBModifications,
    SLURBModule,
    SLURBVariableModification,
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
)
from .vars import VariableDefinition, get_all_vars

setup_logging()

__all__ = [
    "Atmosphere",
    "AtmosphereModule",
    "AtmosphericProfile",
    "Configuration",
    "DefaultNamelistModule",
    "Emission",
    "EmissionModule",
    "GeneralPhysicsModule",
    "Geometry",
    "GridDales",
    "GridDalesOpenBC",
    "IBM",
    "IBMModule",
    "InterpolatedProfile",
    "LBC",
    "OpenBoundaryConfig",
    "OpenBoundaryModule",
    "Radiation",
    "RadiationModule",
    "RunModule",
    "Surface",
    "SurfaceModule",
    "TimeDependentScalar",
    "TimeModule",
    "TimedependentModule",
    "VariableDefinition",
    "dales_simulation",
    "get_all_vars",
    "logwrap",
    "module_registry",
    "register_module",
    "set_nml_section",
    "setup_logging",
    "simulation_module",
    "vars",
]
