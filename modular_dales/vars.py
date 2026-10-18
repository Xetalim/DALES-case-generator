from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableDefinition:
    """Definition of a variable exchanged between modules.

    This is the user-facing object you should import and pass to
    AtmosphericProfile / InterpolatedProfile instead of raw strings.

        Subclasses describe the physical kind and output role. A
        ``time_dependent_name`` selects a series in ``forcings.<id>.nc``;
        time-height variables instead write to ``init.<id>.nc``.
    """

    name: str
    long_name: str
    unit: str
    time_dependent_name: str | None = None
    file_name: str | None = None
    """Name in ``init.<id>.nc`` when it differs from ``name``."""

    @property
    def init_name(self) -> str:
        return self.file_name or self.name

    @property
    def is_profile(self) -> bool:
        return isinstance(self, ProfileVariable)

    @property
    def can_be_time_dependent(self) -> bool:
        return self.is_profile or self.time_dependent_name is not None

    @property
    def must_only_be_time_dependent(self) -> bool:
        return isinstance(self, SeriesOnlyVariable)

    @property
    def init_time_height(self) -> bool:
        return isinstance(self, TimeHeightVariable)

    @property
    def is_initial_profile(self) -> bool:
        return (
            self.is_profile
            and not self.must_only_be_time_dependent
            and not self.init_time_height
        )


@dataclass(frozen=True)
class ProfileVariable(VariableDefinition):
    """An atmospheric height profile."""

    @property
    def physical_variable(self) -> ProfileVariable:
        return self

    @property
    def nudger(self) -> NudgingVariable:
        """Return the registered nudging definition for this physical field."""
        target = self.physical_variable
        if not target.is_initial_profile:
            raise TypeError("Nudging requires an atmospheric state profile")
        for variable in get_all_vars():
            if isinstance(variable, NudgingVariable) and variable.target == target:
                return variable
        nudger = NudgingVariable(target)
        register_var(nudger)
        return nudger

    @property
    def boundary_name(self) -> str:
        return self.name


class SeriesOnlyVariable(ProfileVariable):
    """A forcing profile with no initial-state output."""


class TimeHeightVariable(ProfileVariable):
    """A profile written on time and height axes in the initialization file."""


@dataclass(frozen=True, init=False)
class NudgingVariable(TimeHeightVariable):
    """A time-height nudging target linked to its physical state variable."""

    target: ProfileVariable

    def __init__(
        self,
        target: ProfileVariable,
        *,
        name: str | None = None,
        file_name: str | None = None,
        long_name: str | None = None,
    ) -> None:
        if not isinstance(target, ProfileVariable) or isinstance(
            target, NudgingVariable
        ):
            raise TypeError("A nudging target requires a non-nudging profile variable")
        super().__init__(
            name=name or f"{target.name}_nudge",
            long_name=long_name or f"Nudging target for {target.long_name}",
            unit=target.unit,
            file_name=file_name or f"{target.name}_nud",
        )
        object.__setattr__(self, "target", target)

    @property
    def physical_variable(self) -> ProfileVariable:
        return self.target.physical_variable

    @property
    def boundary_name(self) -> str:
        return self.target.boundary_name


@dataclass(frozen=True)
class SurfaceVariable(VariableDefinition):
    """A surface value, optionally varying in time."""


@dataclass(frozen=True)
class SoilVariable(VariableDefinition):
    """A soil-layer value requested by the land-surface model."""


@dataclass(frozen=True)
class RadiationBackgroundVariable(VariableDefinition):
    """A radiation sounding on pressure levels rather than atmospheric heights."""


class UVariable(ProfileVariable):
    """An eastward wind profile."""

    @property
    def boundary_name(self) -> str:
        return "u"


class VVariable(ProfileVariable):
    """A northward wind profile."""

    @property
    def boundary_name(self) -> str:
        return "v"


class WVariable(ProfileVariable):
    """A vertical velocity profile."""

    @property
    def boundary_name(self) -> str:
        return "w"


class ThetalVariable(ProfileVariable):
    """A liquid water potential temperature profile."""

    @property
    def boundary_name(self) -> str:
        return "thl"


class QtVariable(ProfileVariable):
    """A total water mixing ratio profile."""


class TkeVariable(ProfileVariable):
    """A square-root turbulence kinetic energy profile."""

    @property
    def boundary_name(self) -> str:
        return "e120"


thls = SurfaceVariable(
    "thls",
    "Surface liquid water potential temperature",
    "K",
    time_dependent_name="thlsurf_timedep",
)
wtsurf = SurfaceVariable(
    "wtsurf",
    "Surface kinematic heat flux",
    "K m/s",
    time_dependent_name="wtsurf_timedep",
)
wqsurf = SurfaceVariable(
    "wqsurf",
    "Surface kinematic moisture flux",
    "kg/kg m/s",
    time_dependent_name="wqsurf_timedep",
)
qtsurf = SurfaceVariable(
    "qtsurf",
    "Surface total water mixing ratio",
    "kg/kg",
    time_dependent_name="qtsurf_timedep",
)
psurf = SurfaceVariable(
    "psurf",
    "Surface pressure",
    "Pa",
    time_dependent_name="psurf_timedep",
)
qnetav = SurfaceVariable(
    "qnetav",
    "Net radiative flux at the surface",
    "W/m^2",
    time_dependent_name="qnetavsurf_timedep",
)
ua = UVariable("ua", "Initial eastward velocity profile", "m/s")
va = VVariable("va", "Initial northward velocity profile", "m/s")
w = WVariable("w", "Vertical velocity profile", "m/s")
thetal = ThetalVariable(
    "thetal",
    "Initial liquid water potential temperature profile",
    "K",
)
qt = QtVariable(
    "qt",
    "Initial total water mixing ratio profile.",
    "kg/kg",
)
tke = TkeVariable(
    "tke",
    "Initial profile of the square root of the turbulence kinetic energy (TKE)",
    "m/s",
)
ug = UVariable(
    "ug",
    "Geostrophic eastward wind",
    "m/s",
    time_dependent_name="ug_timedep",
)
vg = VVariable(
    "vg",
    "Geostrophic northward wind",
    "m/s",
    time_dependent_name="vg_timedep",
)
dpdx = ProfileVariable(
    "dpdx",
    "Eastward pressure gradient",
    "Pa/m",
    time_dependent_name="dpdx_ls_timedep",
)
dpdy = ProfileVariable(
    "dpdy",
    "Northward pressure gradient",
    "Pa/m",
    time_dependent_name="dpdy_ls_timedep",
)
wa = WVariable(
    "wa",
    "Large-scale subsidence.",
    "m/s",
    time_dependent_name="wf_ls_timedep",
)
ua_nudge = NudgingVariable(ua, long_name="Nudging target for eastward wind")
va_nudge = NudgingVariable(va, long_name="Nudging target for northward wind")
thl_nudge = NudgingVariable(
    thetal,
    name="thl_nudge",
    long_name="Nudging target for liquid water potential temperature",
)
wa_nudge = NudgingVariable(
    w,
    name="wa_nudge",
    file_name="wa_nud",
    long_name="Nudging target for vertical velocity",
)
qt_nudge = NudgingVariable(
    qt, long_name="Nudging target for total water specific humidity"
)


def nudging_timescale(target: str) -> VariableDefinition:
    return TimeHeightVariable(
        f"nudging_constant_{target}",
        f"Nudging timescale for {target}",
        "s",
    )


nudging_constant_ua = nudging_timescale("ua")
nudging_constant_va = nudging_timescale("va")
nudging_constant_wa = nudging_timescale("wa")
nudging_constant_thetal = nudging_timescale("thetal")
nudging_constant_qt = nudging_timescale("qt")
dqtdxls = ProfileVariable(
    "dqtdxls",
    "Eastward gradient of the total water mixing ratio due to advection",
    "kg/kg/m",
    time_dependent_name="dqtdx_ls_timedep",
)
dqtdyls = ProfileVariable(
    "dqtdyls",
    "Northward gradient of the total water mixing ratio due to advection",
    "kg/kg/m",
    time_dependent_name="dqtdy_ls_timedep",
)
tnqt_adv = ProfileVariable(
    "tnqt_adv",
    "Tendency of the total water mixing ratio",
    "kg/kg/s",
    time_dependent_name="dqtdt_ls_timedep",
)
tnthetal_rad = ProfileVariable(
    "tnthetal_rad",
    "Tendency of the liquid water potential temperature due to radiative heating",
    "K/s",
    time_dependent_name="dthl_rad_timedep",
)
dudt_ls = SeriesOnlyVariable(
    "dudt_ls",
    "Tendency of the eastward velocity due to large-scale forcing",
    "m/s^2",
    time_dependent_name="dudt_ls_timedep",
)
dvdt_ls = SeriesOnlyVariable(
    "dvdt_ls",
    "Tendency of the northward velocity due to large-scale forcing",
    "m/s^2",
    time_dependent_name="dvdt_ls_timedep",
)
dthldt_ls = SeriesOnlyVariable(
    "dthldt_ls",
    "Tendency of the liquid water potential temperature due to large-scale forcing",
    "K/s",
    time_dependent_name="dthldt_ls_timedep",
)

# Soil state (dimension ``zs``), requested by the LSM.
t_soil = SoilVariable("t_soil", "Soil temperature", "K")
theta_soil = SoilVariable("theta_soil", "Volumetric soil moisture", "m3/m3")
type_soil = SoilVariable("type_soil", "Soil type index", "-")
z0m = SurfaceVariable("z0m", "Roughness length for momentum", "m")
z0h = SurfaceVariable("z0h", "Roughness length for heat", "m")

# Radiation background sounding (dimension ``lev`` = pressure in Pa), requested by radiation.
backrad_T = RadiationBackgroundVariable(
    "backrad_T", "Background temperature for radiation", "K"
)
backrad_q = RadiationBackgroundVariable(
    "backrad_q", "Background specific humidity for radiation", "kg/kg"
)
backrad_o3 = RadiationBackgroundVariable(
    "backrad_o3", "Background ozone for radiation", "kg/kg"
)

ALL_VARIABLES: list[VariableDefinition] = [
    thls,
    ua,
    va,
    w,
    thetal,
    qt,
    tke,
    ug,
    vg,
    dpdx,
    dpdy,
    wa,
    ua_nudge,
    va_nudge,
    thl_nudge,
    wa_nudge,
    qt_nudge,
    nudging_constant_ua,
    nudging_constant_va,
    nudging_constant_wa,
    nudging_constant_thetal,
    nudging_constant_qt,
    dqtdxls,
    dqtdyls,
    tnqt_adv,
    tnthetal_rad,
    dthldt_ls,
    dudt_ls,
    dvdt_ls,
    wtsurf,
    wqsurf,
    qtsurf,
    psurf,
    qnetav,
    t_soil,
    theta_soil,
    type_soil,
    z0m,
    z0h,
    backrad_T,
    backrad_q,
    backrad_o3,
]

ATMO_VARS_BY_NAME: dict[str, VariableDefinition] = {v.name: v for v in ALL_VARIABLES}


def register_var(var):
    ALL_VARIABLES.append(var)
    ATMO_VARS_BY_NAME.clear()
    ATMO_VARS_BY_NAME.update({v.name: v for v in ALL_VARIABLES})


def ensure_var(var: VariableDefinition) -> VariableDefinition:
    """Register ``var`` unless a variable with that name exists; return the registered one."""
    if var.name not in ATMO_VARS_BY_NAME:
        register_var(var)
    return ATMO_VARS_BY_NAME[var.name]


def get_var_by_name() -> dict[str, VariableDefinition]:
    return ATMO_VARS_BY_NAME


def get_all_vars() -> list[VariableDefinition]:
    return ALL_VARIABLES


__all__ = [
    "ALL_VARIABLES",
    "ATMO_VARS_BY_NAME",
    "NudgingVariable",
    "ProfileVariable",
    "QtVariable",
    "RadiationBackgroundVariable",
    "SeriesOnlyVariable",
    "SoilVariable",
    "SurfaceVariable",
    "ThetalVariable",
    "TimeHeightVariable",
    "TkeVariable",
    "UVariable",
    "VVariable",
    "VariableDefinition",
    "WVariable",
    "backrad_T",
    "backrad_o3",
    "backrad_q",
    "dpdx",
    "dpdy",
    "dqtdxls",
    "dqtdyls",
    "dthldt_ls",
    "dudt_ls",
    "dvdt_ls",
    "ensure_var",
    "nudging_constant_qt",
    "nudging_constant_thetal",
    "nudging_constant_ua",
    "nudging_constant_va",
    "nudging_constant_wa",
    "nudging_timescale",
    "psurf",
    "qnetav",
    "qt",
    "qt_nudge",
    "qtsurf",
    "t_soil",
    "theta_soil",
    "thetal",
    "thl_nudge",
    "thls",
    "tke",
    "tnqt_adv",
    "tnthetal_rad",
    "type_soil",
    "ua",
    "ua_nudge",
    "ug",
    "va",
    "va_nudge",
    "vg",
    "w",
    "wa",
    "wa_nudge",
    "wqsurf",
    "wtsurf",
    "z0h",
    "z0m",
]
