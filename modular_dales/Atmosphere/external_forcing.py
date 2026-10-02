"""Shared base for modules that derive DALES forcings from an LS2D-like ``les_input``."""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import xarray as xr

from modular_dales.Atmosphere.les_input import LESInput, validate_les_input
from modular_dales.modular.forcing import (
    EXTERNAL_FORCING_PRIORITY,
    ForcingSet,
    named,
    sounding_forcings,
    var_attrs,
)
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.modular.time_dependent_scalars import (
    SOIL_DIM,
    TIME_DIM,
    Z_DIM,
    check_time_series,
    profile,
)
from modular_dales.vars import (
    VariableDefinition,
    dthldt_ls,
    dudt_ls,
    dvdt_ls,
    ensure_var,
    nudging_constant_qt,
    nudging_constant_thetal,
    nudging_constant_ua,
    nudging_constant_va,
    nudging_constant_wa,
    nudging_timescale,
    psurf,
    qt,
    qt_nudge,
    qtsurf,
    t_soil,
    theta_soil,
    thetal,
    thl_nudge,
    thls,
    tke,
    tnqt_adv,
    type_soil,
    ua,
    ua_nudge,
    ug,
    va,
    va_nudge,
    vg,
    w,
    wa,
    wa_nudge,
    wqsurf,
    wtsurf,
    z0h,
    z0m,
)

logger = logging.getLogger(__name__)

LES_TIME_DIMS = ("time", "time_sec")

# DALES variable -> les_input field. State fields (ua, va, thetal, qt) are
# provided as series so that consumers such as open boundaries can use them.
LES_INPUT_SERIES: dict[VariableDefinition, str] = {
    ug: "ug",
    vg: "vg",
    wa: "wls",
    tnqt_adv: "dtqt_advec",
    dthldt_ls: "dtthl_advec",
    dudt_ls: "dtu_advec",
    dvdt_ls: "dtv_advec",
    psurf: "ps",
    thls: "ts",
    wtsurf: "wth",
    wqsurf: "wq",
    ua: "u",
    va: "v",
    thetal: "thl",
    qt: "qt",
}


@dataclass
class ExternalForcingModule(simulation_module):
    """Base class for atmosphere forcing from external data (ERA5 via LS2D, Harmonie).

    Subclasses only implement :meth:`load_les_input`, returning an LS2D-like
    ``xr.Dataset`` with a ``time_sec`` coordinate and fields ``u, v, thl, qt,
    ug, vg, wls, dt*_advec, ps, ts`` (and optionally ``wth, wq, z0m, z0h``,
    soil fields and ``p_lay, t_lay, h2o_lay``) on the DALES heights. This base
    converts it into a :class:`ForcingSet`: initial profiles and nudging for
    :class:`AtmosphereModule`, forcings for :class:`TimedependentModule`, soil
    state for the LSM and a background sounding for radiation. User-configured
    values in other modules take precedence.
    """

    forcing_priority = EXTERNAL_FORCING_PRIORITY

    sim: Optional["simulation_module"] = field(default=None, repr=False)

    do_nudging: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "NAMNUDGE", "key": "lnudge"},
    )
    central_lat: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    central_lon: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    init_tke: float = field(
        default=0.1,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    smooth_initial_uv_to_geostrophic: bool = field(
        default=False,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    write_nudging_netcdf: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    nudging_timescale_ua: float | None = field(
        default=10800.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    nudging_timescale_va: float | None = field(
        default=10800.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    nudging_timescale_wa: float | None = field(
        default=10800.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    nudging_timescale_thetal: float | None = field(
        default=10800.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    nudging_timescale_qt: float | None = field(
        default=10800.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    nudging_tracers: list[dict[str, Any]] = field(
        default_factory=list,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )

    les_input: LESInput | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    forcings: ForcingSet = field(
        default_factory=ForcingSet,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = type(self).__name__

    def check_settings(self):
        if self.grid is None:
            raise ValueError(
                f"{self.module_name} requires a GridDales grid (sim.grid) to be set"
            )
        if self.central_lat is None and getattr(self.grid, "xlat", None) is not None:
            self.central_lat = float(self.grid.xlat)
        if self.central_lon is None and getattr(self.grid, "xlon", None) is not None:
            self.central_lon = float(self.grid.xlon)

    @property
    def zt(self) -> np.ndarray:
        """DALES full-level heights, the vertical axis of all provided profiles."""
        return np.asarray(self.grid.zt, dtype=float)

    @abstractmethod
    def load_les_input(self) -> LESInput:
        """Return a Dataset satisfying LES_INPUT_SCHEMA on self.grid.zt.

        See Atmosphere.les_input for field names, dimensions and units.
        """

    def prepare_calculation(self):
        if self.grid is None:
            raise ValueError(
                f"{self.module_name} requires grid to be initialized before prepare_calculation"
            )
        tracer_sources = []
        if self.do_nudging and self.write_nudging_netcdf:
            for config in self.nudging_tracers:
                if not isinstance(config, dict):
                    raise TypeError("nudging_tracers entries must be dictionaries")
                if config.get("enabled", True):
                    tracer_sources.append(
                        str(config.get("source", config.get("name", ""))).strip()
                    )
        self.les_input = validate_les_input(
            self.load_les_input(),
            z=self.zt,
            nudging=self.do_nudging and self.write_nudging_netcdf,
            tracers=tracer_sources,
        )
        self.forcings = self._build_forcings()
        logger.info(
            "%s: prepared forcings for %d times and %d levels",
            self.module_name,
            self.les_input["time_sec"].size,
            self.zt.size,
        )

    def provide_forcings(self) -> ForcingSet:
        self.ensure_prepared()
        return self.forcings

    def write_files(self):
        return None

    # ------------------------------------------------------------------
    # les_input -> ForcingSet
    # ------------------------------------------------------------------

    def _les_field(self, name: str) -> xr.DataArray | None:
        """``les_input[name]`` with dims ``(time[, zt])`` and time in seconds."""

        if self.les_input is None or name not in self.les_input:
            return None
        times = np.asarray(self.les_input["time_sec"].values, dtype=float)
        da = self.les_input[name].reset_coords(drop=True)
        if set(da.dims) not in ({"time"}, {"time", "z"}):
            raise ValueError(
                f"{self.module_name}: les_input.{name} must have dimensions (time,) or (time, z); got {da.dims}"
            )
        da = da.assign_coords({TIME_DIM: times})
        if "z" in da.dims:
            da = da.rename({"z": Z_DIM}).transpose(TIME_DIM, Z_DIM)
        return check_time_series(da.astype(float).rename(name))

    def _build_forcings(self) -> ForcingSet:
        times = self.les_input["time_sec"]
        if times.ndim != 1 or times.size < 2:
            raise ValueError(
                f"{self.module_name} expects les_input.time_sec to be 1D with at least 2 entries"
            )
        forcings = ForcingSet()
        fields = {
            name: self._les_field(name) for name in set(LES_INPUT_SERIES.values())
        }

        def initial(name: str) -> xr.DataArray | None:
            series = fields.get(name)
            if series is None or Z_DIM not in series.dims:
                return None
            return series.isel({TIME_DIM: 0}, drop=True)

        u0, v0 = self._apply_initial_wind_sponge_to_geostrophic(
            initial("u"), initial("v"), initial("ug"), initial("vg")
        )
        for var_definition, values in (
            (thetal, initial("thl")),
            (qt, initial("qt")),
            (ua, u0),
            (va, v0),
        ):
            if values is not None:
                forcings.initial[var_definition.name] = named(values, var_definition)

        forcings.initial[w.name] = profile(0.0, self.zt, w.name, var_attrs(w))
        forcings.initial[tke.name] = profile(
            self.init_tke, self.zt, tke.name, var_attrs(tke)
        )

        for var_definition, name in LES_INPUT_SERIES.items():
            series = fields.get(name)
            if series is None:
                continue
            if (Z_DIM in series.dims) != var_definition.is_profile:
                logger.warning(
                    "%s: skipping les_input.%s for '%s' (dims %s)",
                    self.module_name,
                    name,
                    var_definition.name,
                    series.dims,
                )
                continue
            forcings.series[var_definition.name] = named(series, var_definition)

        qt_series = fields.get("qt")
        if qt_series is not None and Z_DIM in qt_series.dims:
            # Lowest model level as proxy for the surface value.
            forcings.series[qtsurf.name] = named(
                qt_series.isel({Z_DIM: 0}, drop=True), qtsurf
            )

        if self.do_nudging and self.write_nudging_netcdf:
            forcings.series.update(self._nudging_series(fields))
        forcings.initial.update(self._surface_state())
        sounding = backrad_sounding(self.les_input)
        if sounding is not None:
            forcings.initial.update(sounding_forcings(sounding))

        return forcings

    def _nudging_series(
        self,
        fields: dict[str, xr.DataArray | None],
    ) -> dict[str, xr.DataArray]:
        """Nudging targets and timescales, written as ``(time, zh)`` fields to ``init.nc``."""

        def require_profile(name: str, source: xr.DataArray | None) -> xr.DataArray:
            if source is None or Z_DIM not in source.dims:
                raise ValueError(
                    f"{self.module_name}: nudging source '{name}' is missing or not a profile"
                )
            return source

        def constant(template: xr.DataArray, target: str, timescale) -> xr.DataArray:
            if timescale is None or float(timescale) <= 0:
                raise ValueError(
                    f"{self.module_name}: nudging timescale for '{target}' must be > 0"
                )
            return xr.full_like(template, float(timescale))

        u = require_profile("u", fields.get("u"))
        targets = [
            (ua_nudge, u, nudging_constant_ua, self.nudging_timescale_ua),
            (
                va_nudge,
                require_profile("v", fields.get("v")),
                nudging_constant_va,
                self.nudging_timescale_va,
            ),
            (
                wa_nudge,
                xr.zeros_like(u),
                nudging_constant_wa,
                self.nudging_timescale_wa,
            ),
            (
                thl_nudge,
                require_profile("thl", fields.get("thl")),
                nudging_constant_thetal,
                self.nudging_timescale_thetal,
            ),
            (
                qt_nudge,
                require_profile("qt", fields.get("qt")),
                nudging_constant_qt,
                self.nudging_timescale_qt,
            ),
        ]

        for tracer_cfg in self.nudging_tracers:
            if not isinstance(tracer_cfg, dict):
                raise ValueError(
                    f"{self.module_name}.nudging_tracers entries must be dicts"
                )
            if not bool(tracer_cfg.get("enabled", True)):
                continue
            tracer_name = str(tracer_cfg.get("name", "")).strip()
            if not tracer_name:
                raise ValueError(
                    f"{self.module_name}.nudging_tracers requires a non-empty 'name'"
                )
            source_name = str(tracer_cfg.get("source", tracer_name)).strip()
            tracer = require_profile(source_name, self._les_field(source_name))
            target = ensure_var(
                VariableDefinition(
                    f"{tracer_name}_nudge",
                    str(
                        tracer_cfg.get(
                            "long_name", f"Nudging target for tracer {tracer_name}"
                        )
                    ),
                    str(
                        tracer_cfg.get("units")
                        or tracer.attrs.get("units")
                        or tracer.attrs.get("unit")
                        or "1"
                    ),
                    can_be_time_dependent=True,
                    init_time_height=True,
                    file_name=f"{tracer_name}_nud",
                )
            )
            targets.append(
                (
                    target,
                    tracer,
                    ensure_var(nudging_timescale(tracer_name)),
                    tracer_cfg.get("timescale", self.nudging_timescale_qt),
                )
            )

        nudging: dict[str, xr.DataArray] = {}
        for target, values, timescale_var, timescale in targets:
            nudging[target.name] = named(values, target)
            nudging[timescale_var.name] = named(
                constant(values, target.name, timescale), timescale_var
            )
        return nudging

    def _surface_state(self) -> dict[str, xr.DataArray]:
        """Initial soil state (on ``zs``), soil type and time-mean roughness lengths."""
        les_input = self.les_input
        state: dict[str, xr.DataArray] = {}

        def at_start(name: str) -> xr.DataArray:
            da = les_input[name].reset_coords(drop=True)
            return da.isel({d: 0 for d in da.dims if d in LES_TIME_DIMS}, drop=True)

        for var_definition in (t_soil, theta_soil):
            if var_definition.name not in les_input:
                continue
            da = at_start(var_definition.name)
            if da.ndim != 1:
                logger.warning(
                    "%s: skipping les_input.%s with dims %s; expected one soil dimension",
                    self.module_name,
                    var_definition.name,
                    da.dims,
                )
                continue
            soil_dim = da.dims[0]
            da = da.drop_vars(soil_dim, errors="ignore")
            if soil_dim != SOIL_DIM:
                da = da.rename({soil_dim: SOIL_DIM})
            if "zs" in les_input and les_input["zs"].size == da.size:
                # LS2D gives soil levels as negative heights; provide positive depths.
                depths = np.abs(np.asarray(les_input["zs"].values, dtype=float))
                da = da.assign_coords({SOIL_DIM: depths})
            state[var_definition.name] = named(da.astype(float), var_definition)

        if type_soil.name in les_input:
            state[type_soil.name] = named(at_start(type_soil.name), type_soil)

        for var_definition in (z0m, z0h):
            if var_definition.name in les_input:
                state[var_definition.name] = named(
                    les_input[var_definition.name].astype(float).mean(), var_definition
                )
        return state

    def _apply_initial_wind_sponge_to_geostrophic(
        self,
        prof_u: xr.DataArray | None,
        prof_v: xr.DataArray | None,
        prof_ug: xr.DataArray | None,
        prof_vg: xr.DataArray | None,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        """Blend initial wind toward geostrophic wind in the upper sponge layer.

        The sponge layer is the thicker of the top 15 levels and the top
        quarter of the domain.
        """

        if not self.smooth_initial_uv_to_geostrophic:
            return prof_u, prof_v

        required = {"u": prof_u, "v": prof_v, "ug": prof_ug, "vg": prof_vg}
        missing = [name for name, values in required.items() if values is None]
        if missing:
            logger.warning(
                "%s: cannot smooth initial wind to geostrophic, missing fields: %s",
                self.module_name,
                ", ".join(missing),
            )
            return prof_u, prof_v

        nz = prof_u.sizes[Z_DIM]
        sponge_start = min(max(nz - 15, 0), int(np.floor(0.75 * nz)))
        sponge_size = nz - sponge_start
        if sponge_size <= 0:
            return prof_u, prof_v
        weights = xr.zeros_like(prof_u)
        weights[sponge_start:] = (
            1.0 if sponge_size == 1 else np.linspace(0.0, 1.0, sponge_size) ** 0.2
        )

        logger.info(
            "%s: smoothed initial ua/va toward ug/vg from level %d to top (%d points)",
            self.module_name,
            sponge_start,
            sponge_size,
        )
        return (
            (1.0 - weights) * prof_u + weights * prof_ug,
            (1.0 - weights) * prof_v + weights * prof_vg,
        )


def backrad_sounding(les_input: xr.Dataset) -> xr.Dataset | None:
    """Time-mean radiation sounding (``T``, ``q`` on pressure ``lev``) from ``les_input``."""
    if les_input is None or not all(
        name in les_input for name in ("p_lay", "t_lay", "h2o_lay")
    ):
        return None

    def time_mean(name: str) -> np.ndarray:
        da = les_input[name]
        time_dims = [d for d in da.dims if d in LES_TIME_DIMS] or [da.dims[0]]
        return da.mean(time_dims).values

    return xr.Dataset(
        {"T": ("lev", time_mean("t_lay")), "q": ("lev", time_mean("h2o_lay"))},
        coords={"lev": time_mean("p_lay")},
    )
