from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from modular_dales.modular.namelist import nml_field
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.module_registry import register_module

if TYPE_CHECKING:
    from modular_dales.Geometry.grids import GridDales


def _normalize_horizontal_points(
    values: float | Sequence[float] | None,
) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, (int, float, np.number)):
        return [values]
    return list(values)


def _resolve_nearest_indices(
    grid_axis: ArrayLike | None, coordinates: Sequence[float], axis_name: str
) -> list[int]:
    if grid_axis is None:
        raise RuntimeError(
            f"Cannot resolve {axis_name} coordinates without an attached GridDales module"
        )
    axis = np.asarray(grid_axis)
    return [int(np.abs(axis - coord).argmin()) + 1 for coord in coordinates]


@overload
def prepare_point_indices(
    module_label: str,
    grid: "GridDales | None",
    *,
    allow_idx_list: Literal[True] = True,
    x_idx: int | Sequence[int] | None = None,
    y_idx: int | Sequence[int] | None = None,
    x: float | Sequence[float] | None = None,
    y: float | Sequence[float] | None = None,
    z_idx: int | Sequence[int] | None = None,
    z: float | Sequence[float] | None = None,
) -> tuple[list[int], list[int], list[int] | None]: ...


@overload
def prepare_point_indices(
    module_label: str,
    grid: "GridDales | None",
    *,
    allow_idx_list: Literal[False],
    x_idx: int | Sequence[int] | None = None,
    y_idx: int | Sequence[int] | None = None,
    x: float | Sequence[float] | None = None,
    y: float | Sequence[float] | None = None,
    z_idx: int | Sequence[int] | None = None,
    z: float | Sequence[float] | None = None,
) -> tuple[int, int, int | None]: ...


@overload
def prepare_point_indices(
    module_label: str,
    grid: "GridDales | None",
    *,
    allow_idx_list: bool,
    x_idx: int | Sequence[int] | None = None,
    y_idx: int | Sequence[int] | None = None,
    x: float | Sequence[float] | None = None,
    y: float | Sequence[float] | None = None,
    z_idx: int | Sequence[int] | None = None,
    z: float | Sequence[float] | None = None,
) -> tuple[list[int], list[int], list[int] | None] | tuple[int, int, int | None]: ...


def prepare_point_indices(
    module_label: str,
    grid: "GridDales | None",
    *,
    allow_idx_list: bool = True,
    x_idx: int | Sequence[int] | None = None,
    y_idx: int | Sequence[int] | None = None,
    x: float | Sequence[float] | None = None,
    y: float | Sequence[float] | None = None,
    z_idx: int | Sequence[int] | None = None,
    z: float | Sequence[float] | None = None,
) -> tuple[list[int], list[int], list[int] | None] | tuple[int, int, int | None]:
    """Resolve one-based indices as lists or first-point scalars, with optional z."""

    def resolve_axis(
        indices: int | Sequence[int] | None,
        coordinates: float | Sequence[float] | None,
        grid_axis: ArrayLike | None,
        axis_name: str,
    ) -> list[int]:
        normalized_indices = _normalize_horizontal_points(indices)
        if normalized_indices is not None:
            return [int(index) for index in normalized_indices]
        normalized_coordinates = _normalize_horizontal_points(coordinates)
        if normalized_coordinates is None:
            raise ValueError(
                f"{module_label} requires {axis_name}_idx or real {axis_name} coordinates"
            )
        return _resolve_nearest_indices(grid_axis, normalized_coordinates, axis_name)

    resolved_x = resolve_axis(x_idx, x, grid.xt if grid is not None else None, "x")
    resolved_y = resolve_axis(y_idx, y, grid.yt if grid is not None else None, "y")
    if len(resolved_x) != len(resolved_y):
        raise ValueError(
            f"{module_label} requires x and y point definitions with matching lengths"
        )
    if not resolved_x:
        raise ValueError(f"{module_label} requires at least one sampling point")

    resolved_z = None
    if z_idx is not None or z is not None:
        resolved_z = resolve_axis(z_idx, z, grid.zt if grid is not None else None, "z")
        if len(resolved_z) != len(resolved_x):
            raise ValueError(
                f"{module_label} requires z point definitions with matching lengths to x/y"
            )
    if allow_idx_list:
        return resolved_x, resolved_y, resolved_z
    return (
        resolved_x[0],
        resolved_y[0],
        resolved_z[0] if resolved_z is not None else None,
    )


@register_module
@dataclass
class EasyOutputModule(simulation_module):
    """Output configuration module for DALES simulation.
    Outputs fielddumps, cape, cross-sections, and statistics at specified intervals.
    When added to a simulation, automatically updates the corresponding namelist parameters in the appropriate sections.

    Args:
        sim: Parent dales_simulation instance
        output_interval: Time interval(s) for output (can be a single value or list for different output types)
        enable_output: Whether to enable output (controls lfielddump, lcape, lcross, lstat in namelist)

    """

    output_interval: int | list[int] | None = nml_field(
        default=60,
        section=[
            "namfielddump",
            "namcape",
            "namlsmcrosssection",
            "namgenstat",
            "namcrosssection",
            "namtimestat",
            "nambudget",
            "nambudget",
            "namgenstat",
            "namradfield",
            "namradfield",
        ],
        key=[
            "dtav",
            "dtav",
            "dtav",
            "dtav",
            "dtav",
            "dtav",
            "dtav",
            "timeav",
            "timeav",
            "dtav",
            "timeav",
        ],
        required=True,
        init=True,
    )
    enable_output: bool = nml_field(
        default=True,
        section=[
            "namfielddump",
            "namcape",
            "namlsmcrosssection",
            "namgenstat",
            "namcrosssection",
            "namtimestat",
            "nambudget",
            "namradfield",
        ],
        key=[
            "lfielddump",
            "lcape",
            "lcross",
            "lstat",
            "lcross",
            "ltimestat",
            "lbudget",
            "lradfield",
        ],
        required=True,
    )

    module_name: str = field(
        default="EasyOutputModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class CapeModule(simulation_module):
    """Dedicated output module for NAMCAPE statistics."""

    enabled: bool = nml_field(
        default=False,
        section="namcape",
        key="lcape",
        required=True,
    )
    dtav: int | None = nml_field(
        default=60,
        section="namcape",
        key="dtav",
        required=True,
    )

    module_name: str = field(
        default="CapeModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class LSMCrossModule(simulation_module):
    """Dedicated output module for NAMLSMCROSSSECTION statistics."""

    enabled: bool = nml_field(
        default=False,
        section="namlsmcrosssection",
        key="lcross",
        required=True,
    )
    dtav: int | None = nml_field(
        default=60,
        section="namlsmcrosssection",
        key="dtav",
        required=True,
    )

    module_name: str = field(
        default="LSMCrossModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class TimestatModule(simulation_module):
    """Dedicated output module for NAMTIMESTAT statistics."""

    enabled: bool = nml_field(
        default=False,
        section="namtimestat",
        key="ltimestat",
        required=True,
    )
    dtav: int | None = nml_field(
        default=60,
        section="namtimestat",
        key="dtav",
        required=True,
    )

    module_name: str = field(
        default="TimestatModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class StatsModule(simulation_module):
    """Combined profile-statistics module for GENSTAT, BUDGET, and RADSTAT."""

    enabled: bool = nml_field(
        default=False,
        section=["namgenstat", "nambudget", "NAMRADSTAT"],
        key=["lstat", "lbudget", "lstat"],
        required=True,
    )
    dtav: int | None = nml_field(
        default=60,
        section=["namgenstat", "nambudget", "NAMRADSTAT"],
        key=["dtav", "dtav", "dtav"],
        required=True,
    )
    timeav: int | None = nml_field(
        default=60,
        section=["namgenstat", "nambudget", "NAMRADSTAT"],
        key=["timeav", "timeav", "timeav"],
        required=True,
    )
    radstat_clear_air: bool = nml_field(
        default=False,
        section="NAMRADSTAT",
        key="lradclearair",
        required=True,
    )

    module_name: str = field(
        default="StatsModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class RadfieldModule(simulation_module):
    """Dedicated output module for NAMRADFIELD sampling."""

    enabled: bool = nml_field(
        default=False,
        section="namradfield",
        key="lradfield",
        required=True,
    )
    dtav: int | None = nml_field(
        default=60,
        section="namradfield",
        key="dtav",
        required=True,
    )
    timeav: int | None = nml_field(
        default=60,
        section="namradfield",
        key="timeav",
        required=True,
    )
    module_name: str = field(
        default="RadfieldModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class CrossSectionOutputModule(simulation_module):
    """Output configuration module for cross-sections only.

    This module controls atmospheric cross-section output interval, selected
    planes, and their corresponding indices.
    """

    cross_enabled: bool = nml_field(
        default=True,
        section="namcrosssection",
        key="lcross",
        required=True,
    )
    cross_dtav: int | None = nml_field(
        default=60,
        section="namcrosssection",
        key="dtav",
        required=True,
    )
    xy: list[int] = nml_field(
        default_factory=list,
        section="namcrosssection",
        key="crossheight",
        required=True,
    )
    xz: list[int] = nml_field(
        default_factory=list,
        section="namcrosssection",
        key="crossplane",
        required=True,
    )
    yz: list[int] = nml_field(
        default_factory=list,
        section="namcrosssection",
        key="crossortho",
        required=True,
    )
    xy_coords: list[float] = field(default_factory=list, metadata={"serialize": False})
    xz_coords: list[float] = field(default_factory=list, metadata={"serialize": False})
    yz_coords: list[float] = field(default_factory=list, metadata={"serialize": False})
    xy_enabled: bool = nml_field(
        default=False,
        section="namcrosssection",
        key="lxy",
        required=True,
    )
    xz_enabled: bool = nml_field(
        default=False,
        section="namcrosssection",
        key="lxz",
        required=True,
    )
    yz_enabled: bool = nml_field(
        default=False,
        section="namcrosssection",
        key="lyz",
        required=True,
    )

    module_name: str = field(
        default="CrossSectionOutputModule", init=False, metadata={"serialize": False}
    )

    @staticmethod
    def _nearest_index(values: list[float], target: float) -> int:
        return min(range(len(values)), key=lambda i: abs(values[i] - target))

    def _resolve_index_or_coords(
        self,
        indices: list[int],
        coords: list[float],
        axis_values: list[float],
        label: str,
        base_offset: int,
        min_allowed: int,
        max_allowed: int,
    ) -> list[int]:
        if indices and coords:
            raise ValueError(
                f"CrossSectionOutputModule.{label}: provide either indices or real-space coordinates, not both"
            )

        if coords:
            resolved = [
                self._nearest_index(axis_values, float(coord)) + base_offset
                for coord in coords
            ]
        else:
            resolved = list(indices)

        for value in resolved:
            if value < min_allowed or value > max_allowed:
                raise ValueError(
                    f"CrossSectionOutputModule.{label}: index {value} out of range [{min_allowed}, {max_allowed}]"
                )

        return resolved

    @staticmethod
    def _merge_unique(existing: list[int], incoming: list[int]) -> list[int]:
        return list(dict.fromkeys(list(existing) + list(incoming)))

    def prepare_calculation(self) -> None:

        self.xy = self._resolve_index_or_coords(
            indices=self.xy,
            coords=self.xy_coords,
            axis_values=[float(v) for v in self.grid.zt],
            label="xy",
            base_offset=1,
            min_allowed=1,
            max_allowed=int(self.grid.kmax),
        )
        self.xz = self._resolve_index_or_coords(
            indices=self.xz,
            coords=self.xz_coords,
            axis_values=[float(v) for v in self.grid.yt],
            label="xz",
            base_offset=2,
            min_allowed=2,
            max_allowed=int(self.grid.jtot) + 1,
        )
        self.yz = self._resolve_index_or_coords(
            indices=self.yz,
            coords=self.yz_coords,
            axis_values=[float(v) for v in self.grid.xt],
            label="yz",
            base_offset=2,
            min_allowed=2,
            max_allowed=int(self.grid.itot) + 1,
        )
        self.check_for_new_cross()
        """No preparation needed."""

    def check_for_new_cross(self) -> None:
        existing_cross = self.nml.get("namcrosssection", {}) if self.nml else {}
        self.xy = self._merge_unique(
            [int(v) for v in existing_cross.get("crossheight", [])], self.xy
        )
        self.xz = self._merge_unique(
            [int(v) for v in existing_cross.get("crossplane", [])], self.xz
        )
        self.yz = self._merge_unique(
            [int(v) for v in existing_cross.get("crossortho", [])], self.yz
        )

        if len(self.xy) > 0:
            self.xy_enabled = True
        else:
            self.xy_enabled = False
        if len(self.xz) > 0:
            self.xz_enabled = True
        else:
            self.xz_enabled = False
        if len(self.yz) > 0:
            self.yz_enabled = True
        else:
            self.yz_enabled = False

    def write_files(self) -> None:
        self.check_for_new_cross()
        self.apply_namelist_from_fields()
        """Write output files if needed."""


@register_module
@dataclass
class CheckSimulationModule(simulation_module):
    """
    Module for checking simulation validity and configuration.

    This module provides functionality to monitor and validate a DALES simulation
    during runtime. It allows output at an interval of `check_interval` seconds, at which DALES prints
    the currrent simulation time and ETA. It can also check for invalid values, verify tendencies, and
    optionally stop the simulation if invalid conditions are detected.

    Attributes:
        check_interval (Optional[int]): Time interval (in seconds) between output.
            Default is 60 seconds. Mapped to NAMCHECKSIM:tcheck in Fortran namelist.
            If 0, DALES prints every timestep.
        stop_on_invalid (bool): If True, stops the simulation when invalid values are detected.
            Default is False. Mapped to NAMCHECKSIM:lstop in Fortran namelist.
        check_tendencies (bool): If True, validates model tendencies during runtime checks.
            Default is False. Mapped to NAMCHECKSIM:lchecktend in Fortran namelist.
    """

    check_interval: int | None = nml_field(
        default=60,
        section="NAMCHECKSIM",
        key="tcheck",
        required=True,
        metadata={"serialize": True},
        init=True,
    )
    stop_on_invalid: bool = nml_field(
        default=False,
        section="NAMCHECKSIM",
        key="lstop",
        required=True,
        metadata={"serialize": True},
        init=True,
    )
    check_tendencies: bool = nml_field(
        default=False,
        section="NAMCHECKSIM",
        key="lchecktend",
        required=True,
        metadata={"serialize": True},
        init=True,
    )

    module_name: str = field(
        default="CheckSimulationModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class SamplingModule(simulation_module):
    """Output configuration module for DALES simulation.
    Outputs samples at specified intervals.
    When added to a simulation, automatically updates the corresponding namelist parameters in the appropriate sections.

    Args:
        sim: Parent dales_simulation instance
        output_interval: Time interval(s) for output (can be a single value or list for different output types)
        enable_output: Whether to enable output (controls lsampcl, lsampco, lsampup, etc. in namelist)

    """

    output_interval: int | list[int] | None = nml_field(
        default=60,
        section=["namsampling", "namsampling"],
        key=[
            "dtav",
            "timeav",
        ],
        required=True,
        init=True,
    )
    enable_output: bool = nml_field(
        default=True,
        section=[
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
            "namsampling",
        ],
        key=[
            "lsampcl",
            "lsampco",
            "lsampup",
            "lsampbuup",
            "lsampcldup",
            "lsamptend",
            "ltendleib",
            "lsamptendu",
            "lsamptendv",
            "lsamptendw",
            "lsamptendthl",
            "lsamptendqt",
            "lsamptendqr",
            "lsamptendnr",
            "lqlflux",
        ],
        required=True,
    )

    module_name: str = field(
        default="SamplingModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class FielddumpModule(simulation_module):
    """Dedicated output module for NAMFIELDDUMP with individual variable switches."""

    dtav: int = nml_field(
        default=60,
        section="namfielddump",
        key="dtav",
        required=True,
        description="Sampling interval in seconds for field dumps.",
    )
    lfielddump: bool = nml_field(
        default=True,
        section="namfielddump",
        key="lfielddump",
        required=True,
        description="Enable field dump output.",
    )
    # ldiracc: bool = field(
    #     default=False,
    #     metadata={
    #         "nml": "namfielddump",
    #         "key": "ldiracc",
    #         "doc": "Use direct-access output file mode.",
    #     },
    # )
    # lbinary: bool = field(
    #     default=False,
    #     metadata={
    #         "nml": "namfielddump",
    #         "key": "lbinary",
    #         "doc": "Write binary fielddump files instead of text.",
    #     },
    # )
    klow: int | None = nml_field(
        default=None,
        section="namfielddump",
        key="klow",
        description="Lowest model level index included in dump.",
    )
    khigh: int | None = nml_field(
        default=None,
        section="namfielddump",
        key="khigh",
        description="Highest model level index included in dump; 0 lets DALES decide.",
    )
    ncoarse: int | None = nml_field(
        default=None,
        section="namfielddump",
        key="ncoarse",
        description="Horizontal coarsening factor for dumped fields.",
    )
    tmin: float | None = nml_field(
        default=None,
        section="namfielddump",
        key="tmin",
        description="Start time in seconds for writing dumps.",
    )
    tmax: float | None = nml_field(
        default=None,
        section="namfielddump",
        key="tmax",
        description="End time in seconds for writing dumps.",
    )

    lu: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lu",
        description="Dump u velocity field.",
    )
    lv: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lv",
        description="Dump v velocity field.",
    )
    lw: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lw",
        description="Dump w velocity field.",
    )
    lqt: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lqt",
        description="Dump total water mixing ratio field.",
    )
    lql: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lql",
        description="Dump liquid water mixing ratio field.",
    )
    lthl: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lthl",
        description="Dump liquid water potential temperature field.",
    )
    lbuoy: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lbuoy",
        description="Dump buoyancy field.",
    )
    lcli: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lcli",
        description="Dump cloud ice field.",
    )
    lclw: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lclw",
        description="Dump cloud liquid water field.",
    )
    lta: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lta",
        description="Dump absolute temperature field.",
    )
    lplw: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lplw",
        description="Dump longwave pressure-related field.",
    )
    lpli: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lpli",
        description="Dump ice-phase pressure-related field.",
    )
    lhus: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lhus",
        description="Dump specific humidity field.",
    )
    lhur: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lhur",
        description="Dump relative humidity field.",
    )
    ltntr: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="ltntr",
        description="Dump total tendency field (resolved).",
    )
    ltntrs: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="ltntrs",
        description="Dump subgrid tendency field.",
    )
    ltntrl: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="ltntrl",
        description="Dump large-scale tendency field.",
    )
    le12: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="le12",
        description="Dump subgrid TKE field.",
    )
    lekh: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lekh",
        description="Dump heat diffusivity field.",
    )
    lekm: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lekm",
        description="Dump momentum diffusivity field.",
    )
    lsv: bool | None = nml_field(
        default=None,
        section="namfielddump",
        key="lsv",
        description="Dump scalar tracer fields.",
    )

    module_name: str = field(
        default="FielddumpModule", init=False, metadata={"serialize": False}
    )

    def check_settings(self) -> None:
        if self.khigh and self.klow and self.khigh != 0 and self.khigh < self.klow:
            raise ValueError(
                "FielddumpModule: khigh must be >= klow, or 0 for DALES default"
            )


@register_module
@dataclass
class NetCDFStatisticsSyncModule(simulation_module):
    """Configure NAMNETCDFSTATS synchronization settings."""

    lsync: bool = nml_field(
        default=True,
        section="NAMNETCDFSTATS",
        key="lsync",
        required=True,
    )

    module_name: str = field(
        default="NetCDFStatisticsSyncModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class BulkMicrophysicsStatisticsOutputModule(simulation_module):
    """Configure NAMBULKMICROSTAT output."""

    enabled: bool = nml_field(
        default=True,
        section="NAMBULKMICROSTAT",
        key="lmicrostat",
        required=True,
    )
    dtav: float | None = nml_field(
        default=60,
        section="NAMBULKMICROSTAT",
        key="dtav",
        required=True,
    )
    timeav: float | None = nml_field(
        default=60,
        section="NAMBULKMICROSTAT",
        key="timeav",
        required=True,
    )

    module_name: str = field(
        default="BulkMicrophysicsStatisticsOutputModule",
        init=False,
        metadata={"serialize": False},
    )


@register_module
@dataclass
class VirtualMeasurementOutputModule(simulation_module):
    """Configure NAMVIRTUALMEASUREMENT point output.

    Points can be specified directly via DALES indices or by providing real
    horizontal coordinates, which are snapped to the nearest cell center.
    """

    enabled: bool = nml_field(
        default=True,
        section="NAMVIRTUALMEASUREMENT",
        key="lvirtualmeasurement",
        required=True,
    )
    dtav: float | None = nml_field(
        default=60,
        section="NAMVIRTUALMEASUREMENT",
        key="dtav",
        required=True,
    )
    npoints: int | None = nml_field(
        default=None,
        section="NAMVIRTUALMEASUREMENT",
        key="npoints",
        init=False,
    )
    x_idx: int | list[int] | None = nml_field(
        default=None,
        section="NAMVIRTUALMEASUREMENT",
        key="x_idx",
    )
    y_idx: int | list[int] | None = nml_field(
        default=None,
        section="NAMVIRTUALMEASUREMENT",
        key="y_idx",
    )
    x: float | list[float] | None = field(default=None)
    y: float | list[float] | None = field(default=None)

    module_name: str = field(
        default="VirtualMeasurementOutputModule",
        init=False,
        metadata={"serialize": False},
    )

    def prepare_calculation(self) -> None:
        self.x_idx, self.y_idx, _ = prepare_point_indices(
            "VirtualMeasurementOutputModule",
            self.grid,
            allow_idx_list=True,
            x_idx=self.x_idx,
            y_idx=self.y_idx,
            x=self.x,
            y=self.y,
        )
        self.npoints = len(self.x_idx)


@register_module
@dataclass
class ColumnStatisticsOutputModule(simulation_module):
    """Configure NAMCOLSTAT point output.

    Points can be specified directly via DALES indices or by providing real
    horizontal coordinates, which are snapped to the nearest cell center.
    """

    enabled: bool = nml_field(
        default=True,
        section="NAMCOLSTAT",
        key="lcolstat",
        required=True,
    )
    npoints: int | None = nml_field(
        default=None,
        section="NAMCOLSTAT",
        key="npoints",
        init=False,
    )
    x_idx: int | list[int] | None = nml_field(
        default=None,
        section="NAMCOLSTAT",
        key="x_idx",
    )
    y_idx: int | list[int] | None = nml_field(
        default=None,
        section="NAMCOLSTAT",
        key="y_idx",
    )
    x: float | list[float] | None = field(default=None)
    y: float | list[float] | None = field(default=None)

    module_name: str = field(
        default="ColumnStatisticsOutputModule",
        init=False,
        metadata={"serialize": False},
    )

    def prepare_calculation(self) -> None:
        self.x_idx, self.y_idx, _ = prepare_point_indices(
            "ColumnStatisticsOutputModule",
            self.grid,
            allow_idx_list=True,
            x_idx=self.x_idx,
            y_idx=self.y_idx,
            x=self.x,
            y=self.y,
        )
        self.npoints = len(self.x_idx)
