from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableDefinition:
    """Definition of a variable exchanged between modules.

    This is the user-facing object you should import and pass to
    AtmosphericProfile / InterpolatedProfile instead of raw strings.

    Where provided data end up follows from the definition:

    * ``is_profile`` and not ``must_only_be_time_dependent`` and not
      ``init_time_height``: initial profile in ``init.<id>.nc``.
    * ``time_dependent_name``: series in ``forcings.<id>.nc``.
    * ``init_time_height``: ``(time, zh)`` field in ``init.<id>.nc`` (nudging).
    * otherwise: only available to requesting modules (soil, backrad, ...).
    """

    name: str
    long_name: str
    unit: str
    can_be_time_dependent: bool = False
    must_only_be_time_dependent: bool = False
    is_profile: bool = True
    time_dependent_name: str | None = None
    can_nudge: bool = False
    init_time_height: bool = False
    file_name: str | None = None
    """Name in ``init.<id>.nc`` when it differs from ``name``."""
    fallback: str | None = None
    """Variable to use when this one is not provided (e.g. nudging target -> state)."""

    @property
    def init_name(self) -> str:
        return self.file_name or self.name

    @property
    def is_initial_profile(self) -> bool:
        return (
            self.is_profile
            and not self.must_only_be_time_dependent
            and not self.init_time_height
        )


thls = VariableDefinition(
    "thls",
    "Surface liquid water potential temperature",
    "K",
    is_profile=False,
    can_be_time_dependent=True,
    time_dependent_name="thlsurf_timedep",
)
wtsurf = VariableDefinition(
    "wtsurf",
    "Surface kinematic heat flux",
    "K m/s",
    is_profile=False,
    can_be_time_dependent=True,
    time_dependent_name="wtsurf_timedep",
)
wqsurf = VariableDefinition(
    "wqsurf",
    "Surface kinematic moisture flux",
    "kg/kg m/s",
    is_profile=False,
    can_be_time_dependent=True,
    time_dependent_name="wqsurf_timedep",
)
qtsurf = VariableDefinition(
    "qtsurf",
    "Surface total water mixing ratio",
    "kg/kg",
    is_profile=False,
    can_be_time_dependent=True,
    time_dependent_name="qtsurf_timedep",
)
psurf = VariableDefinition(
    "psurf",
    "Surface pressure",
    "Pa",
    is_profile=False,
    can_be_time_dependent=True,
    time_dependent_name="psurf_timedep",
)
qnetav = VariableDefinition(
    "qnetav",
    "Net radiative flux at the surface",
    "W/m^2",
    is_profile=False,
    can_be_time_dependent=True,
    time_dependent_name="qnetavsurf_timedep",
)
ua = VariableDefinition(
    "ua", "Initial eastward velocity profile", "m/s", can_nudge=True
)
va = VariableDefinition(
    "va", "Initial northward velocity profile", "m/s", can_nudge=True
)
w = VariableDefinition("w", "Vertical velocity profile", "m/s")
thetal = VariableDefinition(
    "thetal",
    "Initial liquid water potential temperature profile",
    "K",
    can_nudge=True,
)
qt = VariableDefinition(
    "qt",
    "Initial total water mixing ratio profile.",
    "kg/kg",
    can_nudge=True,
)
tke = VariableDefinition(
    "tke",
    "Initial profile of the square root of the turbulence kinetic energy (TKE)",
    "m/s",
)
ug = VariableDefinition(
    "ug",
    "Geostrophic eastward wind",
    "m/s",
    can_be_time_dependent=True,
    time_dependent_name="ug_timedep",
)
vg = VariableDefinition(
    "vg",
    "Geostrophic northward wind",
    "m/s",
    can_be_time_dependent=True,
    time_dependent_name="vg_timedep",
)
dpdx = VariableDefinition(
    "dpdx",
    "Eastward pressure gradient",
    "Pa/m",
    can_be_time_dependent=True,
    time_dependent_name="dpdx_ls_timedep",
)
dpdy = VariableDefinition(
    "dpdy",
    "Northward pressure gradient",
    "Pa/m",
    can_be_time_dependent=True,
    time_dependent_name="dpdy_ls_timedep",
)
wa = VariableDefinition(
    "wa",
    "Large-scale subsidence.",
    "m/s",
    can_be_time_dependent=True,
    time_dependent_name="wf_ls_timedep",
)
ua_nudge = VariableDefinition(
    "ua_nudge",
    "Nudging target for eastward wind",
    "m/s",
    can_be_time_dependent=True,
    can_nudge=True,
    init_time_height=True,
    file_name="ua_nud",
    fallback="ua",
)
va_nudge = VariableDefinition(
    "va_nudge",
    "Nudging target for northward wind",
    "m/s",
    can_be_time_dependent=True,
    can_nudge=True,
    init_time_height=True,
    file_name="va_nud",
    fallback="va",
)
thl_nudge = VariableDefinition(
    "thl_nudge",
    "Nudging target for liquid water potential temperature",
    "K",
    can_be_time_dependent=True,
    can_nudge=True,
    init_time_height=True,
    file_name="thetal_nud",
    fallback="thetal",
)
wa_nudge = VariableDefinition(
    "wa_nudge",
    "Nudging target for vertical velocity",
    "m/s",
    can_be_time_dependent=True,
    can_nudge=True,
    init_time_height=True,
    file_name="wa_nud",
    fallback="w",
)
qt_nudge = VariableDefinition(
    "qt_nudge",
    "Nudging target for total water specific humidity",
    "kg/kg",
    can_be_time_dependent=True,
    can_nudge=True,
    init_time_height=True,
    file_name="qt_nud",
    fallback="qt",
)


def nudging_timescale(target: str) -> VariableDefinition:
    return VariableDefinition(
        f"nudging_constant_{target}",
        f"Nudging timescale for {target}",
        "s",
        can_be_time_dependent=True,
        init_time_height=True,
    )


nudging_constant_ua = nudging_timescale("ua")
nudging_constant_va = nudging_timescale("va")
nudging_constant_wa = nudging_timescale("wa")
nudging_constant_thetal = nudging_timescale("thetal")
nudging_constant_qt = nudging_timescale("qt")
dqtdxls = VariableDefinition(
    "dqtdxls",
    "Eastward gradient of the total water mixing ratio due to advection",
    "kg/kg/m",
    can_be_time_dependent=True,
    time_dependent_name="dqtdx_ls_timedep",
)
dqtdyls = VariableDefinition(
    "dqtdyls",
    "Northward gradient of the total water mixing ratio due to advection",
    "kg/kg/m",
    can_be_time_dependent=True,
    time_dependent_name="dqtdy_ls_timedep",
)
tnqt_adv = VariableDefinition(
    "tnqt_adv",
    "Tendency of the total water mixing ratio",
    "kg/kg/s",
    can_be_time_dependent=True,
    time_dependent_name="dqtdt_ls_timedep",
)
tnthetal_rad = VariableDefinition(
    "tnthetal_rad",
    "Tendency of the liquid water potential temperature due to radiative heating",
    "K/s",
    can_be_time_dependent=True,
    time_dependent_name="dthl_rad_timedep",
)
dudt_ls = VariableDefinition(
    "dudt_ls",
    "Tendency of the eastward velocity due to large-scale forcing",
    "m/s^2",
    can_be_time_dependent=True,
    must_only_be_time_dependent=True,
    time_dependent_name="dudt_ls_timedep",
)
dvdt_ls = VariableDefinition(
    "dvdt_ls",
    "Tendency of the northward velocity due to large-scale forcing",
    "m/s^2",
    can_be_time_dependent=True,
    must_only_be_time_dependent=True,
    time_dependent_name="dvdt_ls_timedep",
)
dthldt_ls = VariableDefinition(
    "dthldt_ls",
    "Tendency of the liquid water potential temperature due to large-scale forcing",
    "K/s",
    can_be_time_dependent=True,
    must_only_be_time_dependent=True,
    time_dependent_name="dthldt_ls_timedep",
)

# Soil state (dimension ``zs``), requested by the LSM.
t_soil = VariableDefinition("t_soil", "Soil temperature", "K", is_profile=False)
theta_soil = VariableDefinition(
    "theta_soil", "Volumetric soil moisture", "m3/m3", is_profile=False
)
type_soil = VariableDefinition("type_soil", "Soil type index", "-", is_profile=False)
z0m = VariableDefinition("z0m", "Roughness length for momentum", "m", is_profile=False)
z0h = VariableDefinition("z0h", "Roughness length for heat", "m", is_profile=False)

# Radiation background sounding (dimension ``lev`` = pressure in Pa), requested by radiation.
backrad_T = VariableDefinition(
    "backrad_T", "Background temperature for radiation", "K", is_profile=False
)
backrad_q = VariableDefinition(
    "backrad_q", "Background specific humidity for radiation", "kg/kg", is_profile=False
)
backrad_o3 = VariableDefinition(
    "backrad_o3", "Background ozone for radiation", "kg/kg", is_profile=False
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
    "VariableDefinition",
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
