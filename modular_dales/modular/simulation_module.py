import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Union

from modular_dales.logging_wrapper import logwrap
from modular_dales.modular.forcing import (
    USER_FORCING_PRIORITY,
    ForcingSet,
    var_attrs,
)
from modular_dales.modular.namelist import get_namelist_spec
from modular_dales.modular.time_dependent_scalars import (
    Z_DIM,
    TimeDependentScalar,
    value_at,
)
from modular_dales.vars import get_var_by_name

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from modular_dales.Geometry.GridDales import GridDales
    from modular_dales.modular.dales_simulation import dales_simulation


def set_nml_section(
    nml,
    nml_docs,
    nml_module_setter: str,
    section: str,
    key: str,
    value: Any,
    raise_conflict: bool = False,
) -> None:
    """Set multiple key-value pairs in a namelist section."""
    if section not in nml:
        nml[section] = {}
    if section not in nml_docs:
        nml_docs[section] = {}
    if raise_conflict and key in nml[section] and nml[section][key] != value:
        raise ValueError(
            f"Conflict detected in module '{nml_module_setter}': {key} already exists in section {section} with value {nml[section][key]}, cannot set to {value}! (use manual override)"
        )
    nml[section][key] = value
    nml_docs[section][key] = f"{value} (Set by {nml_module_setter} module)"


@dataclass(init=False, repr=False, eq=False)
class simulation_module(ABC):
    """Base class for DALES simulation modules."""

    sim: "dales_simulation | None" = field(default=None, repr=False, kw_only=True)

    forcing_priority: ClassVar[int] = USER_FORCING_PRIORITY
    """Priority of this module's forcings when several modules provide the same variable."""

    def __init__(self, sim: "dales_simulation | None" = None):
        """Initialize module with reference to parent simulation.

        Args:
            sim: Parent dales_simulation instance providing config, grid, paths, etc.
                 Can be None if module will be added to simulation later.
        """
        self.sim = sim
        self.module_name = None  # Optional name for logging and identification
        self.do_config_done = False
        self.prepare_calculation_done = False
        self._prepare_in_progress = False
        self._provided_forcings: ForcingSet | None = None
        self.check_settings_done = False
        self.write_files_done = False

    def __post_init__(self):
        simulation_module.__init__(self, self.sim)

    def apply_namelist_from_fields(self) -> None:
        """Apply dataclass field values to the namelist using field metadata.

        Namelist bindings are declared with ``nml_field`` / ``NamelistSpec``.
        Other field metadata keys:
            forcing_var: Variable whose t=0 value (from another module's time
                series, e.g. LS2D) replaces a plain value of this field
        """

        for dataclass_field in fields(self):
            meta = dataclass_field.metadata or {}
            spec = get_namelist_spec(dataclass_field)
            if spec is None:
                continue
            value = getattr(self, dataclass_field.name)

            forcing_var = meta.get("forcing_var")
            if forcing_var is not None and not isinstance(value, TimeDependentScalar):
                external_value = self._external_forcing_start_value(forcing_var)
                if external_value is not None:
                    logger.info(
                        "%s: '%s' taken from provided time series at t=0 (%s)",
                        self.module_name,
                        dataclass_field.name,
                        external_value,
                    )
                    value = external_value
            if isinstance(value, TimeDependentScalar):
                value = value.value_at_start()
            if value is None:
                if spec.required:
                    raise ValueError(
                        f"Module '{self.module_name}' requires '{dataclass_field.name}'"
                    )
                continue
            targets = spec.targets(dataclass_field.name)
            multiple = isinstance(spec.section, tuple) or isinstance(spec.key, tuple)
            if multiple and isinstance(value, dict):
                for target, item in value.items():
                    section, key = target.split(":", 1)
                    self.set_nml_section(
                        section, key, item, raise_conflict=spec.raise_conflict
                    )
                continue
            values = value if multiple and isinstance(value, list) else [value] * len(targets)
            for (section, key), item in zip(targets, values, strict=True):
                self.set_nml_section(
                    section, key, item, raise_conflict=spec.raise_conflict
                )

    def set_nml_section(
        self,
        section: str,
        key: str,
        value: Any,
        raise_conflict: bool = False,
    ) -> None:
        """Helper to set a single namelist section/key from within module methods."""
        set_nml_section(
            self.nml,
            self.nml_docs,
            self.module_name,
            section,
            key,
            value,
            raise_conflict=raise_conflict,
        )

    @property
    def grid(self) -> "GridDales":
        """Access grid from parent simulation."""
        return self.sim.grid if self.sim is not None else None

    @property
    def output_path(self):
        """Access output_path from parent simulation."""
        return self.sim.output_path if self.sim is not None else None

    @property
    def nml(self):
        """Access namelist from parent simulation."""
        return self.sim.nml if self.sim is not None else None

    @property
    def nml_docs(self):
        """Access namelist documentation from parent simulation."""
        return self.sim.nml_docs if self.sim is not None else None

    @property
    def exp_id(self):
        """Access experiment ID from parent simulation."""
        return self.sim.exp_id if self.sim is not None else None

    @property
    def required_folder_list(self):
        """Access required_folder_list from parent simulation.

        This is used by modules that need to register additional
        folders whose contents must be linked or copied to the
        runtime directory (e.g. emissions, radiation tables).
        """

        return self.sim.required_folder_list if self.sim is not None else None

    def retrieve_module(
        self, module_name: Union[str, "simulation_module", type]
    ) -> "simulation_module":
        """Helper to retrieve another module from the parent simulation by name."""
        if self.sim is None:
            raise RuntimeError(
                f"Module '{self.module_name}' is not attached to a simulation!"
            )
        try:
            return self.sim.retrieve_module(module_name)
        except KeyError:
            raise KeyError(
                f"Module with name '{module_name}' not found in simulation for retrieval by module '{self.module_name}'"
            )

    def module_exists(self, module_name: Union[str, "simulation_module", type]) -> bool:
        """Check if a module with the given name exists in the parent simulation."""
        if self.sim is None:
            raise RuntimeError(
                f"Module '{self.module_name}' is not attached to a simulation!"
            )
        return self.sim.module_exists(module_name)

    @logwrap
    def _initialize_from_sim(self, sim: "dales_simulation"):
        """Initialize module with parent simulation reference."""
        self.sim = sim

    def do_config(self):
        """Configure namelist and settings for this module.

        This is called before prepare_calculation() and should be used by
        configuration modules (like DomainConfigModule, RunConfigModule, etc.)
        to set up namelist parameters.

        Default implementation does nothing. Override in subclasses as needed.
        """

    def check_settings(self):
        """Check and validate settings for this module.

        This is called after all do_config() methods have completed, allowing
        modules to check for conflicts and validate that the full configuration
        is valid.

        Default implementation does nothing. Override in subclasses as needed.
        """

    @abstractmethod
    def prepare_calculation(self):
        """Prepare data and setup for calculations."""

    @abstractmethod
    def write_files(self):
        """Write output files for this module."""

    def ensure_prepared(self) -> None:
        """Run ``prepare_calculation`` once; safe to call from dependent modules."""

        if self.prepare_calculation_done:
            return
        if getattr(self, "_prepare_in_progress", False):
            raise RuntimeError(
                f"Circular prepare dependency involving module '{self.module_name}'"
            )
        self._prepare_in_progress = True
        try:
            self.prepare_calculation()
        finally:
            self._prepare_in_progress = False
        self.prepare_calculation_done = True

    @staticmethod
    def _forcing_var_name(dataclass_field) -> str | None:
        meta = dataclass_field.metadata or {}
        if "forcing_var" in meta:
            return meta["forcing_var"]
        if dataclass_field.name in get_var_by_name():
            return dataclass_field.name
        return None

    def provided_forcings(self) -> ForcingSet:
        """``provide_forcings()``, evaluated only once."""
        if getattr(self, "_provided_forcings", None) is None:
            self._provided_forcings = self.provide_forcings()
        return self._provided_forcings

    def provide_forcings(self) -> ForcingSet:
        """Forcings provided by this module.

        The default exposes dataclass fields holding a ``TimeDependentScalar``
        as time series of the variable named by ``metadata["forcing_var"]``
        (or by the field name). Override for richer providers.
        """

        forcings = ForcingSet()
        if not is_dataclass(self):
            return forcings
        for dataclass_field in fields(self):
            value = getattr(self, dataclass_field.name, None)
            if not isinstance(value, TimeDependentScalar):
                continue
            var_name = self._forcing_var_name(dataclass_field)
            if var_name is None:
                raise ValueError(
                    f"Module '{self.module_name}' field '{dataclass_field.name}' holds a TimeDependentScalar "
                    "but maps to no known variable; set metadata['forcing_var']"
                )
            forcings.series[var_name] = value.as_series(var_name).assign_attrs(
                var_attrs(get_var_by_name()[var_name])
            )
        return forcings

    def _external_forcing_start_value(self, var_name: str) -> float | None:
        """t=0 value of a series for ``var_name`` provided by another module, if any."""

        if self.sim is None or not self.check_settings_done:
            return None
        series = self.sim.forcings().series.get(var_name)
        if series is None or Z_DIM in series.dims:
            return None
        return float(value_at(series, 0.0))
