"""Emission module for tracer and emission handling.

This module mirrors the IBM module pattern:

- Emission tracers and point sources are represented as dataclasses that can be
    added to the module using the ``+=`` operator.
- The module then constructs the internal emissions data structures and writes
    the required NetCDF files during ``prepare_calculation`` / ``write_files``.

"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Self

from modular_dales.Emission.create_emis import (
    PointSource,
    TracerInfo,
    emissions,
    get_emis_sim_hours,
)
from modular_dales.modular import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@dataclass
class EmissionTracer:
    """Single tracer configuration.

    This closely follows ``tracer_info`` in ``create_emis`` so that it can be
    converted directly when constructing the internal ``emissions`` object.
    """

    name: str
    long_name: str
    unit: str
    molar_mass: float
    lemis: bool = False
    lreact: bool = False
    ldep: bool = False
    lags: bool = False


@dataclass
class EmissionPointSource:
    """Single emission point source tied to a tracer.

    ``tracer_name`` must match the ``name`` of an ``EmissionTracer`` that is
    added to the same ``EmissionModule``.
    """

    tracer_name: str
    x_idx: int
    y_idx: int
    height: float
    temperature: float
    volume: float
    emission: float
    stack_exit_area: float


@register_module
@dataclass
class EmissionModule(simulation_module):
    """Emission simulation module for tracer and emission handling.

    Tracers and point sources can be added directly to this module, e.g.::

        emis = EmissionModule()
        emis += EmissionTracer(
            name="co2",
            long_name="Carbon Dioxide (CO2)",
            unit="ppm",
            molar_mass=44.009,
            lemis=True,
        )
        emis += EmissionPointSource(
            tracer_name="co2",
            x_idx=8,
            y_idx=8,
            height=10,
            temperature=293.0,
            volume=1.0,
            emission=10.0,
            stack_exit_area=1.0,
        )

    Args:
      sim: Parent simulation instance
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)

    # User-configurable content
    tracers: list[EmissionTracer] = field(
        default_factory=list, metadata={"serialize": True}
    )
    point_sources: list[EmissionPointSource] = field(
        default_factory=list, metadata={"serialize": True}
    )

    # Runtime-only helpers
    emissions_instance: emissions | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    l_emission: bool = field(
        default=True,
        metadata={
            "nml": "NAMEMISSION",
            "key": "l_emission",
            "serialize": True,
            "required": True,
        },
        init=True,
    )
    l_points: bool = field(
        default=False,
        metadata={
            "nml": "NAMEMISSION",
            "key": "l_points",
            "serialize": False,
            "required": True,
        },
        init=True,
    )
    explicit_plume_rise: bool = field(
        default=False,
        metadata={
            "nml": "NAMEMISSION",
            "key": "explicit_plume_rise",
            "serialize": True,
            "required": True,
        },
        init=True,
    )
    emisnames: list[str] = field(
        default_factory=list,
        metadata={
            "nml": "NAMEMISSION",
            "key": "emisnames",
            "serialize": False,
            "required": True,
        },
        init=True,
    )
    nemis: int = field(
        default=0,
        metadata={
            "nml": "NAMEMISSION",
            "key": "nemis",
            "serialize": False,
            "required": True,
        },
        init=True,
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "EmissionModule"

    # ------------------------------------------------------------------
    # Public API for adding tracers / point sources
    # ------------------------------------------------------------------
    def __add__(self, item: Any) -> "EmissionModule":
        """Add a tracer or point source to the module.

        This mirrors the pattern used by :class:`IBMModule`, allowing
        users to configure emissions programmatically via ``+=``.
        """

        if isinstance(item, EmissionTracer):
            self.tracers.append(item)
        elif isinstance(item, EmissionPointSource):
            self.point_sources.append(item)
        else:
            raise TypeError(
                f"Cannot add object of type {type(item)} to EmissionModule; "
                "expected EmissionTracer or EmissionPointSource."
            )
        return self

    def __iadd__(self, item: Any) -> Self:
        """In-place addition for ``+=`` syntax."""

        return self.__add__(item)

    def prepare_calculation(self):
        """Set up emission tracers and update namelist.

        If no tracers or point sources have been added, emissions are disabled
        and this method is a no-op.
        """

        # Build emissions instance directly from dataclasses
        emis = emissions(output_path=self.output_path, grid=self.grid)

        # Add tracers
        for tr in self.tracers:
            info = TracerInfo(
                name=tr.name,
                long_name=tr.long_name,
                unit=tr.unit,
                molar_mass=tr.molar_mass,
                lemis=tr.lemis,
                lreact=tr.lreact,
                ldep=tr.ldep,
                lags=tr.lags,
            )
            emis.add_tracer(info)

        # Add point sources
        for ps in self.point_sources:
            src = PointSource(
                x_idx=ps.x_idx,
                y_idx=ps.y_idx,
                height=ps.height,
                temperature=ps.temperature,
                volume=ps.volume,
                emission=ps.emission,
                stack_exit_area=ps.stack_exit_area,
            )
            emis.add_pointsource(ps.tracer_name, src)

        self.emissions_instance = emis

    def check_settings(self):
        """Check emission settings validity.

        If this module configures emissions, ensure the runtime knows the
        ``emissions`` folder is required so the job script can link/copy it.
        """

        if not (self.tracers or self.point_sources):
            logger.info("EmissionModule: no emissions configured; skipping setup")
            self.emissions_instance = None
            return

        if len(self.point_sources) > 0:
            self.l_points = True  # Enable point sources if they are added
        for tr in self.tracers:
            if tr.name not in self.emisnames:
                self.emisnames.append(tr.name)
        self.nemis = len(self.emisnames)
        if len(self.emisnames) > 0 and len(self.tracers) == 0:
            raise ValueError(
                "EmissionModule: emisnames provided in namelist but no tracers "
                "configured; please add at least one EmissionTracer to the module."
            )
        folders = self.required_folder_list
        if folders is None:
            return
        if "emissions" not in folders:
            folders.append("emissions")

    def write_files(self):
        """Write emission files to disk."""

        if self.emissions_instance is None:
            return

        # Write tracers and hourly emission data based on the namelist timing.
        self.emissions_instance.write_tracers_file()
        datetime_strs = get_emis_sim_hours(self.nml)
        for datetime_str in datetime_strs:
            self.emissions_instance.write_hourly_emission_data(datetime_str)
