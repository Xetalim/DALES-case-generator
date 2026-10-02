"""Base classes for DALES simulation modules."""

import logging
import os
import pathlib
import subprocess
from dataclasses import is_dataclass
from pathlib import Path, PosixPath
from typing import Union

import f90nml
import yaml

from modular_dales.helpers import package_resource_path
from modular_dales.modular.forcing import ForcingSet, collect_forcings
from modular_dales.modular.simulation_module import set_nml_section, simulation_module
from modular_dales.MODULE_REGISTRY import MODULE_REGISTRY

from .serialize_deserialize import _deserialize_dataclass, asdict_user_set

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


def yaml_equivalent_of_path(dumper, data):
    return dumper.represent_str(str(data))


yaml.add_multi_representer(Path, yaml_equivalent_of_path)
yaml.add_multi_representer(PosixPath, yaml_equivalent_of_path)


def check_core_amounts(machine_conf, nml) -> tuple[int, int, int]:
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

    def valid_decomposition(nprocx, nprocy):
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


class dales_simulation:
    """Main simulation class that manages multiple simulation modules.

    Usage:
        sim = dales_simulation(case_name, machine_conf)
        sim += DefaultNamelistModule()
        sim += GridModule()
        sim += LSMModule()
        # ... add other modules ...
        sim._initialize_modules()  # Initialize grid and validation
        sim.do_config()             # Configure namelist
        sim.check_settings()        # Validate settings
        sim.prepare_all_calculations()  # Prepare calculations
        sim.write_module_files()       # Write output files
    """

    def __init__(self, case_name: str, machine_conf):
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
        self.nml = f90nml.namelist.Namelist()
        self.nml_docs = {}  # stores info about which module set which namelist variables for better error messages and debugging

        # These will be initialized by init_output_folder()
        self.output_path = None
        self.exp_id = None
        self.grid = None
        self.required_files = {}
        self.required_folder_list = []
        self._initialized = False
        self._forcings: ForcingSet | None = None
        self._forcings_in_progress = False

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

    def init_output_folder(self):
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
        self.output_path = output_path

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_path / "input", exist_ok=True)

        # Note: grid creation and additional validation are deferred to
        # the explicit do_config() and check_settings() phases so that
        # configuration modules can modify the namelist first.

        self._initialized = True

    def __add__(self, module: "simulation_module") -> "dales_simulation":
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

        # Initialize module with this simulation if not already initialized
        if module.sim is None:
            module._initialize_from_sim(self)
        else:
            logger.debug(
                "Module %s already has a simulation reference; skipping initialization",
                module,
            )
        self.modules.append(module)
        self._forcings = None
        return self

    def __iadd__(self, module: "simulation_module") -> "dales_simulation":
        """In-place addition of a module (+=).

        Args:
            module: A simulation_module instance to add

        Returns:
            self: Returns the simulation instance
        """
        self.__add__(module)
        return self

    def do_config(self):
        """Configure all modules.

        Call this explicitly after adding all modules but before check_settings().
        """
        for module in self.modules:
            module.do_config()
            module.do_config_done = True

    def check_settings(self):
        """Check settings and validate configuration for all modules.

        Call this explicitly after do_config() but before prepare_all_calculations().
        """
        # Global checks that depend on a configured namelist

        # Per-module checks

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

    def prepare_all_calculations(self):
        """Prepare all modules for calculations.

        Call this explicitly after check_settings().
        Modules that depend on others (e.g. forcing consumers) prepare their
        dependencies on demand via ``ensure_prepared()``.
        """
        for module in self.modules:
            module.ensure_prepared()
            module.apply_namelist_from_fields()
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

    def retrieve_module(
        self, module_name: Union[str, "simulation_module", type]
    ) -> "simulation_module":
        """Helper to retrieve another module by name."""
        for module in self.modules:
            if isinstance(module_name, str):
                if module.module_name == module_name:
                    return module
                else:
                    continue
            elif isinstance(module, module_name):
                return module
        raise KeyError(f"Module with name '{module_name}' not found!")

    def module_exists(self, module_name: Union[str, "simulation_module", type]) -> bool:
        """Check if a module with the given name exists."""
        return any(
            module.module_name == module_name or isinstance(module, module_name)
            for module in self.modules
        )

    def write_module_files(self):
        """Call write_files on all registered modules."""

        for module in self.modules:
            module.write_files()
            module.write_files_done = True

    def write_simulation_files(self):
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
        self.nml.write(
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

        yaml_str = self.save_sim_to_yaml()
        with open(
            self.output_path / f"input/simulation_config.{self.exp_id:03d}.yaml",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(yaml_str)

    def apply_job_configuration(self):
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
            for file in self.required_files.keys():
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

    def setup_module_links(self):
        pass

    def __repr__(self):
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
        modules_yaml = []
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

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        return yaml.dump(
            sim_dict, sort_keys=False, default_flow_style=False, Dumper=NoAliasDumper
        )

    @staticmethod
    def load_sim_from_yaml(txt, machine_conf=None) -> "dales_simulation":
        """
        Load a YAML file into a dales_simulation instance.

        Args:
            path: str or Path to YAML file
            machine_conf:

        Returns:
            sim: populated dales_simulation
        """

        data = yaml.safe_load(txt)

        case_name = data.get("case_name", None)

        if machine_conf is None:
            raise ValueError(
                "machine_conf is required to create a new dales_simulation"
            )
        sim = dales_simulation(case_name, machine_conf)

        # Add modules
        for item in data.get("modules", []):
            # Supported shapes (no backwards compatibility):
            # 1) Concise form without config: "ModuleName"
            # 2) Concise form with config: {"ModuleName": { ... }}

            if isinstance(item, str):
                # Bare module name, e.g. "DefaultNamelistModule"
                mod_type = item
                mod_cfg = {}
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

            cls = MODULE_REGISTRY.get(mod_type)
            if cls is None:
                raise ValueError(f"Unknown module type: {mod_type}")

            # Use recursive dataclass-aware deserialization when possible so
            # that nested structures like LandUseModifications and
            # LandUseModification objects are reconstructed correctly.
            if is_dataclass(cls):
                module = _deserialize_dataclass(cls, mod_cfg)
            else:
                module = cls(**mod_cfg)

            sim += module

        return sim
