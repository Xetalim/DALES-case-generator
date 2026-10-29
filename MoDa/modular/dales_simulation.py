"""Base classes for DALES simulation modules."""

import logging
import os
import pathlib
import subprocess
import warnings
from dataclasses import dataclass, field, is_dataclass
from datetime import date
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any, Self, cast

import f90nml
import numpy as np
import xarray as xr
import yaml

from moda.Configuration.run_and_time import logger
from moda.Core.namelist import Namelist, NamelistDoc, nml_field
from moda.helpers import package_resource_path
from moda.IO_helpers.atmosphere_writer import AtmosphereProfileWriter
from moda.logging_wrapper import case_logging
from moda.modular.forcing import ForcingSet, collect_forcings, var_attrs
from moda.modular.simulation_module import set_nml_section, simulation_module
from moda.modular.time_dependent_scalars import TIME_DIM, Z_DIM, resolve_time_axis
from moda.module_registry import module_registry, register_module
from moda.vars import get_var_by_name

from .serialize_deserialize import _deserialize_dataclass, asdict_user_set

if TYPE_CHECKING:
    from moda.Geometry.grids import GridDales

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def yaml_equivalent_of_path(dumper: yaml.Dumper, data: Path) -> yaml.Node:
    return dumper.represent_str(str(data))


yaml.add_multi_representer(Path, yaml_equivalent_of_path)
yaml.add_multi_representer(PosixPath, yaml_equivalent_of_path)


def check_core_amounts(machine_conf: Any, nml: Namelist) -> tuple[int, int, int]:
    """Adjust core count and MPI decomposition for DALES.

    This helper chooses a decomposition (nprocx, nprocy, nprocs) such that:

    - The total number of cores is as large as possible but not larger than
      the requested ``machine_conf["job_conf"]["numcores"]``.
    - The decomposition is as "square" as possible, with a slight preference
      for more cores in the y-direction (nprocy >= nprocx). Examples:
        - 8 cores  -> nprocx = 2, nprocy = 4
        - 16 cores -> nprocx = 4, nprocy = 4
    - Each subdomain is at least 4 grid points wide in both x and y
      directions: itot / nprocx >= 4 and jtot / nprocy >= 4.

    If no such decomposition exists for the requested core count, the number
    of cores is reduced until a valid decomposition is found. If even a single
    core cannot satisfy the minimum subdomain size, we fall back to a fully
    serial configuration (numcores = nprocx = nprocy = nprocs = 1).

    If the user has already specified nprocx, nprocy and nprocs in the RUN
    group, this function leaves the configuration unchanged.

    Returns: ncores_final, nprocx, nprocy
    """

    ncores_available = int(machine_conf["job_conf"]["numcores"])

    domain_group = nml.get("DOMAIN", {})
    itot = int(domain_group.get("itot", 0))
    jtot = int(domain_group.get("jtot", 0))

    if itot <= 0 or jtot <= 0:
        raise ValueError("itot < 0??? this is not valid")

    run_group = nml.get("RUN", {})
    nprocx = run_group.get("nprocx", None)
    nprocy = run_group.get("nprocy", None)

    def valid_decomposition(nprocx: int, nprocy: int) -> bool:
        return (itot / nprocx) >= 4 and (jtot / nprocy) >= 4

    if (nprocx == 0) and (nprocy == 0):
        while not valid_decomposition(1, ncores_available):
            ncores_available -= 1
            if ncores_available <= 0:
                return 1, 0, 0
        return ncores_available, 0, 0

    # Respect an explicitly specified decomposition
    if (nprocx is not None) and (nprocy is not None):
        if ncores_available != nprocx * nprocy:
            if nprocx * nprocy <= ncores_available:
                ncores_available = nprocx * nprocy
            else:
                raise ValueError(
                    "Explicit MPI decomposition nprocx*nprocy in namelist RUN does not match amount of specified cores for machine and required cores greater than amount available!"
                )
        if valid_decomposition(nprocx, nprocy):
            return ncores_available, run_group["nprocx"], run_group["nprocy"]

    best = None  # (ncores, nprocx, nprocy)

    # Search downwards from the requested core count until we find
    # a valid, reasonably square decomposition.
    for ncores in range(ncores_available, 0, -1):
        factor_pairs = []
        for nx in range(1, int(ncores**0.5) + 1):
            if ncores % nx != 0:
                continue
            ny = ncores // nx

            # Enforce nprocx <= nprocy so that y gets equal or more cores
            nprocx = min(nx, ny)
            nprocy = max(nx, ny)

            if not valid_decomposition(nprocx, nprocy):
                continue

            factor_pairs.append((nprocx, nprocy))

        if not factor_pairs:
            continue

        # Choose the most "square" pair: minimize |x - y|, then maximize y
        factor_pairs.sort(key=lambda pair: (abs(pair[0] - pair[1]), -pair[1]))
        nprocx, nprocy = factor_pairs[0]
        best = (ncores, nprocx, nprocy)
        break

    if best is None:
        logger.warning(
            "Could not find any MPI decomposition with at least 4 grid points per subdomain "
            "for itot=%s, jtot=%s and requested numcores=%s. "
            "Falling back to serial run (numcores=1, nprocx=nprocy=nprocs=1).",
            itot,
            jtot,
            ncores_available,
        )
        ncores_final = 1
        nprocx = nprocy = 1
    else:
        ncores_final, nprocx, nprocy = best

        if ncores_final != ncores_available:
            logger.warning(
                "Requested numcores=%s reduced to %s to ensure at least 4 grid points per subdomain "
                "(itot=%s, jtot=%s, nprocx=%s, nprocy=%s).",
                ncores_available,
                ncores_final,
                itot,
                jtot,
                nprocx,
                nprocy,
            )
        else:
            logger.info(
                "Using numcores=%s with decomposition nprocx=%s, nprocy=%s, "
                "(itot=%s, jtot=%s).",
                ncores_final,
                nprocx,
                nprocy,
                itot,
                jtot,
            )

    # # Update machine configuration
    # machine_conf["job_conf"]["numcores"] = ncores_final

    # # Ensure RUN group exists and update decomposition parameters in the namelist
    # if "RUN" not in nml:
    #     nml["RUN"] = {}
    # run_group = nml["RUN"]

    # run_group["nprocx"] = nprocx
    # run_group["nprocy"] = nprocy

    return ncores_final, nprocx, nprocy


@register_module
@dataclass
class DefaultNamelistModule(simulation_module):
    """Module that configures default RUN section.

    This module sets up default values for the RUN section.

    Args:
        sim: Parent dales_simulation instance
    """

    iexpnr: int = nml_field(
        default=1,
        init=True,
        repr=False,
        section="RUN",
        key="iexpnr",
        required=True,
        metadata={"serialize": True},
    )
    iinput: int = nml_field(
        default=2,
        init=True,
        repr=False,
        section="RUN",
        key="iinput",
        required=True,
        metadata={"serialize": True},
    )
    lwarmstart: bool = nml_field(
        default=False,
        init=True,
        repr=False,
        section="RUN",
        key="lwarmstart",
        required=True,
        metadata={"serialize": True},
    )
    trestart: int = nml_field(
        default=3600,
        init=True,
        repr=False,
        section="RUN",
        key="trestart",
        required=True,
        metadata={"serialize": True},
    )
    startfile: str = nml_field(
        default="initd_latest_x000y000.001",
        init=True,
        repr=False,
        section="RUN",
        key="startfile",
        required=True,
        metadata={"serialize": True},
    )
    ladaptive: bool = nml_field(
        default=True,
        init=True,
        repr=False,
        section="RUN",
        key="ladaptive",
        required=True,
        metadata={"serialize": True},
    )

    module_name: str = field(
        default="DefaultNamelistModule", init=False, metadata={"serialize": False}
    )


@register_module
@dataclass
class TimeModule(simulation_module):
    """Simulation time configuration module.

    Handles time-related parameters in the RUN and DOMAIN sections.

    Args:
        sim: Parent dales_simulation instance
        xtime: Time within the day in seconds (0-86400)
        xday: Day of year (1-365/366)
        xyear: Year
        runtime: Total runtime in seconds
        trestart: Restart file write interval in seconds
    """

    xtime: float = nml_field(
        default=0.0,
        section="DOMAIN",
        key="xtime",
        required=True,
        description="UTC start time in hours within the day for DOMAIN:xtime.",
    )
    xday: int = nml_field(
        default=265,
        section="DOMAIN",
        key="xday",
        required=True,
        description="Day of year (1-365/366) used by DOMAIN:xday.",
    )
    xyear: int = nml_field(
        default=2025,
        section="DOMAIN",
        key="xyear",
        required=True,
        description="Simulation year used by DOMAIN:xyear.",
    )
    enable_datetime: bool = nml_field(
        default=True,
        section="namdatetime",
        key="l_datetime",
        required=True,
        description="Enable NAMDATETIME-based calendar interpretation.",
    )
    timezone: int = nml_field(
        default=0,
        section="namdatetime",
        key="timezone",
        description="Timezone offset in hours for NAMDATETIME.",
    )
    startyear: int = nml_field(
        default=2025,
        section="namdatetime",
        key="startyear",
        required=True,
        description="Calendar start year for NAMDATETIME.",
    )
    startmonth: int = nml_field(
        default=1,
        section="namdatetime",
        key="startmonth",
        required=True,
        description="Calendar start month for NAMDATETIME.",
    )
    startday: int = nml_field(
        default=1,
        section="namdatetime",
        key="startday",
        required=True,
        description="Calendar start day for NAMDATETIME.",
    )
    inferfromdatetime: bool = field(
        default=False,
        metadata={
            "serialize": True,
            "doc": "If true, infer DOMAIN:xday and DOMAIN:xyear from NAMDATETIME start date.",
        },
    )

    runtime: float | None = nml_field(
        default=None,
        section="RUN",
        key="runtime",
        required=True,
        description="Total runtime in seconds for RUN:runtime.",
    )
    trestart: float | None = nml_field(
        default=None,
        section="RUN",
        key="trestart",
        description="Restart write interval in seconds for RUN:trestart.",
    )

    module_name: str = field(
        default="TimeModule", init=False, metadata={"serialize": False}
    )

    def _apply_datetime_inference(self) -> None:
        """Infer DOMAIN xyear/xday from NAMDATETIME start date when enabled."""
        if not self.inferfromdatetime:
            return

        start_dt = date(self.startyear, self.startmonth, self.startday)
        self.xyear = int(start_dt.year)
        self.xday = int(start_dt.timetuple().tm_yday)

    def check_settings(self) -> None:
        """Check time settings validity."""
        self._apply_datetime_inference()
        if self.xyear != self.startyear:
            logger.warning(
                "TimeModule: xyear (%d) does not match startyear (%d)",
                self.xyear,
                self.startyear,
            )


class dales_simulation:
    """Main simulation class that manages multiple simulation modules.

    Usage:
        sim = dales_simulation(case_name, machine_conf)
        sim.add(DefaultNamelistModule())
        sim.add(GridModule())
        sim.add(LSM.LSMModule())
        # ... add other modules ...
        sim._initialize_modules()  # Initialize grid and validation
        sim.do_config()             # Configure namelist
        sim.check_settings()        # Validate settings
        sim.prepare_all_calculations()  # Prepare calculations
        sim.write_module_files()       # Write output files
    """

    def __init__(
        self,
        case_name: str,
        machine_conf: Any,
        *,
        timesteps: list[float] | np.ndarray | None = None,
    ):
        """Initialize DALES simulation with case name and machine settings.

        Only reads configuration at this stage. All work is deferred to
        explicit method calls (_initialize_modules, do_config, check_settings,
        prepare_all_calculations, write_module_files).

        Args:
            case_name: Name of the simulation case
            machine_conf: Machine configuration dictionary
        """
        self.case_name = case_name
        self.machine_conf = machine_conf
        self.modules: list[simulation_module] = []
        self.has_surface_module = False  # set by SurfaceModule when added
        self.has_atmosphere_module = False  # set by AtmosphereModule when added
        # Start with empty namelist - will be populated by modules
        # we cast to dict[str,dict[str,any]] to make the type checking okay. this function does nothing at runtime
        self.nml: Namelist = cast(Namelist, f90nml.namelist.Namelist())
        self.nml_docs: NamelistDoc = {}  # stores info about which module set which namelist variables for better error messages and debugging

        # These will be initialized by init_output_folder()
        self.output_path_: pathlib.Path | None = None
        self.exp_id: int | None = None
        self.grid: GridDales | None = None
        # map output files to their paths
        self.required_files: dict[str, str] = {}
        self.required_folder_list: list[str] = []
        self._initialized = False
        self._forcings: ForcingSet | None = None
        self._forcings_in_progress = False
        self.timesteps = list(timesteps) if timesteps is not None else []
        self.timedep = xr.Dataset()

    def prepare_time_dependent_forcings(self) -> None:
        """Interpolate module forcings onto their combined time axis by default."""
        self.timedep = xr.Dataset()
        vars_by_name = get_var_by_name()
        forcings = self.forcings()
        names = [
            name for name in forcings.series if vars_by_name[name].time_dependent_name
        ]
        if names:
            runtime = self.retrieve_module(TimeModule).runtime
            time_axis = resolve_time_axis(
                (forcings.series[name] for name in names),
                explicit=self.timesteps,
                runtime=runtime,
            )
            if len(time_axis) < 3:
                raise ValueError(
                    "Time-dependent forcings require at least 3 time points "
                    f"(including time=0) for interpolation; got {time_axis.tolist()}"
                )
            resampled = forcings.series_dataset(time_axis, names)
            self.timedep = xr.Dataset(
                {
                    vars_by_name[name].time_dependent_name: resampled[
                        name
                    ].assign_attrs(
                        var_attrs(
                            vars_by_name[name],
                            " (time-dependent)"
                            if Z_DIM in resampled[name].dims
                            else "",
                        )
                    )
                    for name in names
                }
            )
            if Z_DIM not in self.timedep.dims:
                if self.grid is None:
                    raise ValueError(
                        "Time-dependent forcings require a simulation grid"
                    )
                self.timedep = self.timedep.assign_coords(
                    {Z_DIM: np.asarray(self.grid.zt, dtype=float)}
                )
        for key, value in (
            ("ltimedep", bool(names)),
            ("ntimedep", self.timedep.sizes.get(TIME_DIM, 1) - 1),
        ):
            set_nml_section(
                self.nml, self.nml_docs, "dales_simulation", "PHYSICS", key, value
            )

    def forcings(self) -> ForcingSet:
        """Merged data provided by all modules; computed once and then reused."""
        if self._forcings is None:
            if self._forcings_in_progress:
                raise RuntimeError(
                    "sim.forcings() was requested while it is being computed; "
                    "a provider must not request forcings from its provide_forcings()/prepare_calculation()"
                )
            self._forcings_in_progress = True
            try:
                self._forcings = collect_forcings(self.modules)
            finally:
                self._forcings_in_progress = False
        return self._forcings

    @property
    def output_path(self) -> pathlib.Path:
        if self.output_path_ is None:
            raise RuntimeError("Output folder has not been initialized")
        return self.output_path_

    def init_output_folder(self) -> None:
        """Initialize output folder and paths.

        This method is called explicitly at the start of the workflow.
        """
        if self._initialized:
            return

        if self.machine_conf["case_conf"]["BASE_OUTPUT_PATH"]:
            output_path = (
                pathlib.Path(self.machine_conf["case_conf"]["BASE_OUTPUT_PATH"])
                / self.case_name
            )
        else:
            logger.warning(
                "No BASE_OUTPUT_PATH set in machine configuration, using current directory for output"
            )
            output_path = pathlib.Path.cwd() / "generated_cases" / self.case_name
        self.output_path_ = output_path

        os.makedirs(self.output_path_, exist_ok=True)
        os.makedirs(self.output_path_ / "input", exist_ok=True)

        # Note: grid creation and additional validation are deferred to
        # the explicit do_config() and check_settings() phases so that
        # configuration modules can modify the namelist first.

        self._initialized = True

    def add(self, module: "simulation_module") -> Self:
        """Add a module to the simulation registry.

        Args:
            module: A simulation_module instance to add

        Returns:
            self: Returns the simulation instance for chaining
        """

        if not isinstance(module, simulation_module):
            raise TypeError(f"Expected simulation_module instance, got {type(module)}")

        if module.module_name is None:
            raise ValueError(
                f"Module {module} must have a module_name for identification"
            )

        module._attach_to_sim(self)
        if any(existing is module for existing in self.modules):
            return self
        self.modules.append(module)
        self._forcings = None
        # Imported lazily: grids imports moda.modular, which imports this module.
        from moda.Geometry.grids import GridDales

        if isinstance(module, GridDales):
            self.grid = module
        return self

    def __add__(self, module: "simulation_module") -> "dales_simulation":
        return self.add(module)

    def __iadd__(self, module: "simulation_module") -> Self:
        """In-place addition of a module (+=).

        Args:
            module: A simulation_module instance to add

        Returns:
            self: Returns the simulation instance
        """
        warnings.warn(
            "Using '+=' to add simulation modules is deprecated; use add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add(module)

    def do_config(self) -> None:
        """Configure all modules.

        Call this explicitly after adding all modules but before check_settings().
        """
        for module in self.modules:
            module.do_config()
            module.do_config_done = True

    def check_settings(self) -> None:
        """Check settings and validate configuration for all modules.

        Call this explicitly after do_config() but before prepare_all_calculations().
        """
        # Global checks that depend on a configured namelist

        # Per-module checks

        if not self.module_exists(DefaultNamelistModule):
            raise ValueError(
                "No DefaultNamelistModule found in simulation; a default namelist module is required"
            )

        if not self.has_surface_module:
            raise ValueError(
                "No SurfaceModule found in simulation; a surface module is required"
            )
        if not self.has_atmosphere_module:
            raise ValueError(
                "No AtmosphereModule found in simulation; an atmosphere module is required"
            )
        for module in self.modules:
            module.check_settings()
            module.check_settings_done = True

    def prepare_all_calculations(self) -> None:
        """Prepare all modules for calculations.

        Call this explicitly after check_settings().
        Modules that depend on others (e.g. forcing consumers) prepare their
        dependencies on demand via ``ensure_prepared()``.
        """
        for module in self.modules:
            module.ensure_prepared()
            module.apply_namelist_from_fields()
        self.prepare_time_dependent_forcings()
        # check core counts against namelist
        ncores_final, nprocx, nprocy = check_core_amounts(self.machine_conf, self.nml)
        self.machine_conf["job_conf"]["numcores"] = ncores_final
        set_nml_section(
            self.nml, self.nml_docs, "dales_simulation", "RUN", "nprocx", nprocx
        )
        set_nml_section(
            self.nml, self.nml_docs, "dales_simulation", "RUN", "nprocy", nprocy
        )

        if not self.nml.get("RUN", {}).get("iexpnr", None):
            raise ValueError(
                "iexpnr is required in namelist RUN section but not found. Please set iexpnr in the namelist explicitly or in a DefaultNamelistModule"
            )
        else:
            self.exp_id = self.nml["RUN"]["iexpnr"]

    def retrieve_module[T: simulation_module](self, module_type: type[T]) -> T:
        """Retrieve a module instance matching the supplied module class."""
        for module in self.modules:
            if isinstance(module, module_type):
                return module
        raise KeyError(f"Module with type '{module_type}' not found!")

    def module_exists(self, module_type: type["simulation_module"]) -> bool:
        """Check if a module with the given type exists."""
        return any(isinstance(module, module_type) for module in self.modules)

    def write_module_files(self) -> None:
        """Call write_files on all registered modules."""
        if self.exp_id is None:
            raise ValueError("Experiment ID (exp_id) is not set.")

        for module in self.modules:
            module.write_files()
            module.write_files_done = True

        if self.timedep.data_vars:
            AtmosphereProfileWriter().write_forcings(
                self.timedep, self.output_path / "input", self.exp_id
            )

    def write_simulation_files(self) -> None:
        """Write namelist and required files to case directory."""
        for file, originalpath in self.required_files.items():
            subprocess.call(
                [
                    "rsync",
                    "-t",
                    str(originalpath),
                    str(self.output_path / f"input/{file}"),
                ]
            )

        with package_resource_path("input_template", "input") as template_input:
            subprocess.call(
                [
                    "rsync",
                    "-a",
                    template_input.as_posix() + "/",
                    str(self.output_path / "input"),
                ]
            )

        # Write the namelist to the case directory
        cast(Any, self.nml).write(
            self.output_path / f"input/namoptions.{self.exp_id:03d}", force=True
        )
        with open(
            self.output_path / f"input/docs_namoptions.{self.exp_id:03d}.md",
            "w",
            encoding="utf-8",
        ) as f:
            for group, variables in self.nml_docs.items():
                f.write(f"## &{group}\n\n")
                f.writelines(
                    f"- **{var}**: {desc}\n" for var, desc in variables.items()
                )
                f.write("\n")
        try:
            yaml_str = self.save_sim_to_yaml()
            with open(
                self.output_path / f"input/simulation_config.{self.exp_id:03d}.yaml",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(yaml_str)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Failed to save simulation configuration to YAML: %s", exc)

    def apply_job_configuration(self) -> None:
        """Apply job configuration and set up file transfers."""
        job_conf = self.machine_conf.get("job_conf", {})
        configured_template = job_conf.get("job_template", None)
        if configured_template is None:
            logger.warning(
                "No job_template specified in machine configuration; using the bundled job.001 template"
            )
            with package_resource_path("input_template", "job.001") as template_path:
                content = template_path.read_text(encoding="utf-8")
        else:
            configured_path = pathlib.Path(configured_template).expanduser()
            local_paths = (
                configured_path,
                pathlib.Path("input_template") / configured_path,
            )
            local_template = next(
                (path for path in local_paths if path.is_file()), None
            )
            if local_template is not None:
                content = local_template.read_text(encoding="utf-8")
            else:
                resource_name = configured_path.name
                with package_resource_path(
                    "input_template", resource_name
                ) as template_path:
                    if not template_path.is_file():
                        raise ValueError(
                            f"Specified job_template {configured_template} is neither a local file nor a bundled template"
                        )
                    content = template_path.read_text(encoding="utf-8")

        # Set up required file transfers
        required_transfers = ""
        if len(self.required_files) != 0:
            for file in self.required_files:
                required_transfers += (
                    f'ln -sf "$(pwd)/input/{file}" "$RUNDIR/{file}" || exit\n'
                )

        content = content.replace("{{required_filetransfers}}", required_transfers)

        # Set up required folders
        required_folders = ""
        if len(self.required_folder_list) != 0:
            for foldername in self.required_folder_list:
                required_folders += f'mkdir -p "$RUNDIR/{foldername}" || exit\n'

        content = content.replace("{{required_folders}}", required_folders)

        # Set up folder rsync
        required_folder_rsyncs = ""
        for foldername in self.required_folder_list:
            required_folder_rsyncs += f'for inp in input/{foldername}/*\ndo\nln -sf $(pwd)/$inp "$(pwd)/${{RUNDIR}}/{foldername}/$(basename $inp)" || exit\ndone\n'

        content = content.replace("{{required_folder_rsyncs}}", required_folder_rsyncs)

        # Replace job configuration variables
        for varname, replacement in job_conf.items():
            content = content.replace(f"{{{{{varname}}}}}", f'"{replacement!s}"')

        logger.info("Writing job file to %s", self.output_path / "job.001")
        with open(self.output_path / "job.001", "w", encoding="utf-8") as f:
            f.write(content)

        subprocess.call(
            [
                "chmod",
                "+x",
                str(self.output_path / "job.001"),
            ]
        )

    def setup_module_links(self) -> None:
        pass

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        parts = ""
        if self.exp_id is not None:
            parts += f"sim_exp_id={self.exp_id!r}\n"
        if self.grid is not None:
            parts += f"grid:\n{self.grid}\n"
        if self.modules:
            parts += "modules=\n" + "\n".join([f"{module}" for module in self.modules])
        return f"{cls}:\n{parts}"

    def sim_preprocessing_pipeline(self) -> None:
        """Run the full modular pipeline to produce input files.

        This matches the intended workflow:
        - init_output_folder
        - setup_module_links
        - do_config
        - check_settings
        - prepare_all_calculations
        - apply_job_configuration
        - write_module_files (per-module outputs)
        - write_simulation_files (namelist and required files)
        """
        self.init_output_folder()
        with case_logging(self.output_path):
            self.setup_module_links()
            self.do_config()
            self.check_settings()
            self.prepare_all_calculations()
            self.write_module_files()
            self.apply_job_configuration()
            self.write_simulation_files()

    def save_sim_to_yaml(self) -> str:
        """
        Save a dales_simulation instance to YAML including case_name and modules.

        Args:
            sim: dales_simulation instance
        """
        # Build a more concise, module-centric representation:
        #
        # - Modules without any user-set fields are written as plain strings:
        #     - DefaultNamelistModule
        # - Modules with configuration are written as a single-key mapping where
        #   the key is the module class name and the value is the config dict:
        #     - GridModule:
        #         domain_info: {...}
        modules_yaml: list[str | dict[str, Any]] = []
        for m in self.modules:
            name = type(m).__name__
            body = asdict_user_set(m)
            if not body:
                modules_yaml.append(name)
            else:
                modules_yaml.append({name: body})

        sim_dict = {
            "case_name": getattr(self, "case_name", None),
            "modules": modules_yaml,
        }
        if self.timesteps:
            sim_dict["timesteps"] = [float(time) for time in self.timesteps]

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data: Any) -> bool:
                return True

        return yaml.dump(
            sim_dict, sort_keys=False, default_flow_style=False, Dumper=NoAliasDumper
        )

    @staticmethod
    def load_sim_from_yaml(txt: str, machine_conf: Any = None) -> "dales_simulation":
        """
        Load a YAML file into a dales_simulation instance.

        Args:
            txt: str containing YAML content
            machine_conf: machine configuration object

        Returns:
            sim: populated dales_simulation
        """

        data = yaml.safe_load(txt)

        case_name = data.get("case_name", None)

        if machine_conf is None:
            raise ValueError(
                "machine_conf is required to create a new dales_simulation"
            )
        sim = dales_simulation(case_name, machine_conf, timesteps=data.get("timesteps"))

        # Add modules
        for item in data.get("modules", []):
            # Supported shapes (no backwards compatibility):
            # 1) Concise form without config: "ModuleName"
            # 2) Concise form with config: {"ModuleName": { ... }}

            if isinstance(item, str):
                # Bare module name, e.g. "DefaultNamelistModule"
                mod_type = item
                mod_cfg: dict[str, Any] = {}
            elif isinstance(item, dict) and len(item) == 1:
                # Single-key mapping: {"GridModule": { ... }}
                mod_type, mod_cfg = next(iter(item.items()))
                if mod_cfg is None:
                    mod_cfg = {}
                elif not isinstance(mod_cfg, dict):
                    raise ValueError(
                        f"Module entry for {mod_type} must map to a dict, got {type(mod_cfg)}"
                    )
            else:
                raise ValueError(f"Unrecognized module specification: {item!r}")

            cls = module_registry.get(mod_type)
            if cls is None:
                raise ValueError(f"Unknown module type: {mod_type}")

            # Use recursive dataclass-aware deserialization when possible so
            # that nested structures like LandUseModifications and
            # LandUseModification objects are reconstructed correctly.
            if is_dataclass(cls):
                module = _deserialize_dataclass(cls, mod_cfg, sim=sim)
            else:
                module = cls(**mod_cfg)

            sim.add(module)

        return sim
