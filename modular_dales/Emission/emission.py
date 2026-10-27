"""Emission module for tracer and emission handling.

This module mirrors the IBM module pattern:

- Emission tracers and point sources are represented as dataclasses that can be
    added to the module using :meth:`EmissionModule.add`.
- The module then constructs the internal emissions data structures and writes
    the required NetCDF files during ``prepare_calculation`` / ``write_files``.

"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Self

from modular_dales.Emission.create_emis import (
    PointSource,
    TracerInfo,
    emissions,
    get_emis_sim_hours,
)
from modular_dales.modular import simulation_module
from modular_dales.modular.namelist import nml_field
from modular_dales.module_registry import register_module

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
        emis.add(EmissionTracer(
            name="co2",
            long_name="Carbon Dioxide (CO2)",
            unit="ppm",
            molar_mass=44.009,
            lemis=True,
        ))
        emis.add(EmissionPointSource(
            tracer_name="co2",
            x_idx=8,
            y_idx=8,
            height=10,
            temperature=293.0,
            volume=1.0,
            emission=10.0,
            stack_exit_area=1.0,
        ))

    Args:
      sim: Parent simulation instance
    """

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
    l_emission: bool = nml_field(
        default=True,
        section="NAMEMISSION",
        key="l_emission",
        required=True,
        metadata={"serialize": True},
        init=True,
    )
    l_points: bool = nml_field(
        default=False,
        section="NAMEMISSION",
        key="l_points",
        required=True,
        metadata={"serialize": False},
        init=True,
    )
    explicit_plume_rise: bool = nml_field(
        default=False,
        section="NAMEMISSION",
        key="explicit_plume_rise",
        required=True,
        metadata={"serialize": True},
        init=True,
    )
    emisnames: list[str] = nml_field(
        default_factory=list,
        section="NAMEMISSION",
        key="emisnames",
        required=True,
        metadata={"serialize": False},
        init=True,
    )
    nemis: int = nml_field(
        default=0,
        section="NAMEMISSION",
        key="nemis",
        required=True,
        metadata={"serialize": False},
        init=True,
    )

    module_name: str = field(
        default="EmissionModule", init=False, metadata={"serialize": False}
    )

    # ------------------------------------------------------------------
    # Public API for adding tracers / point sources
    # ------------------------------------------------------------------
    def add(self, item: Any) -> Self:
        """Add a tracer or point source to the module.

        This allows users to configure emissions programmatically.
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

    def __add__(self, item: Any) -> Self:
        return self.add(item)

    def __iadd__(self, item: Any) -> Self:
        """In-place addition for ``+=`` syntax."""
        warnings.warn(
            "Using '+=' to add emission items is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(item)

    def prepare_calculation(self) -> None:
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

    def check_settings(self) -> None:
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

    def write_files(self) -> None:
        """Write emission files to disk."""

        if self.emissions_instance is None:
            return

        # Write tracers and hourly emission data based on the namelist timing.
        self.emissions_instance.write_tracers_file()
        datetime_strs = get_emis_sim_hours(self.nml)
        for datetime_str in datetime_strs:
            self.emissions_instance.write_hourly_emission_data(datetime_str)
