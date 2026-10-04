"""Open Boundary Conditions module for lateral boundary forcing."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import dask
import xarray as xr

from modular_dales.Atmosphere import (
    AtmosphericProfile,
    InterpolatedProfile,
    TimedAtmosphereProfile,
)
from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.LBC.nest_dales_in_dales import (
    boundary_fields_fine,
    initial_fields_fine,
)
from modular_dales.LBC.nest_dales_in_HARMONIE import (
    boundary as harmonie_boundary,
)
from modular_dales.LBC.nest_dales_in_HARMONIE import (
    initfields,
    prep_harmonie,
)
from modular_dales.LBC.openbc_atmosphere_worker import OpenBCAtmosphereWorker
from modular_dales.LBC.openbc_knmi_worker import OpenBCKNMIWorker
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.modular.forcing import (
    EXTERNAL_FORCING_PRIORITY,
    ForcingSet,
    sounding_forcings,
)
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)
### FOR DEBUGGING, SET THE DASK SCHEDULER TO SINGLE-THREADED
### ELSE LEAVE COMMENTED OUT
# dask.config.set(
#     scheduler="single-threaded"
# )  # overwrite default with single-threaded scheduler

# if you're on a large cluster, you can for example set up a dask SLURM cluster which has a lot of memory so you can do any open BC calculations better.
# for example, here is how you can setup a SLURM cluster at ATOS. (set your own account), with this you can set tchunk=50 in OPENBC.
# from dask_jobqueue import SLURMCluster
# from dask.distributed import Client
# cluster = SLURMCluster(
#     cores=16,
#     processes=8,
#     memory="128GB",
#     account=ACCOUNT,
#     walltime="00:20:00",
#     job_script_prologue=[
#         'export LANG="en_US.utf8"',
#         'export LANGUAGE="en_US.utf8"',
#         'export LC_ALL="en_US.utf8"',
#     ],
#     job_extra_directives=['--qos="nf"'],
# )
# client = Client(cluster)


@dataclass
class Nest_in_Harmonie:
    """Nest a DALES simulation inside a HARMONIE model environment.

    This class configures and manages the coupling between a DALES
    simulation and an enclosing HARMONIE model, handling the exchange
    of necessary meteorological and surface fields.

    Args:
        ml_glob : Optional[str]
            Glob pattern or path template for HARMONIE model-level fields
            (e.g., 3D atmospheric state) used to drive or constrain DALES.
        sfc_glob : Optional[str]
            Glob pattern or path template for HARMONIE surface fields
            (e.g., surface fluxes, skin temperature) required for the nesting.
    """

    ml_glob: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    sfc_glob: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )


@dataclass
class Nest_in_KNMI:
    """Nest DALES inside HARMONIE using KNMI operational GRIB output.

    This adapter reads pre-converted NetCDF files from KNMI's operational
    HARMONIE archives (N55ML for 3D model-level data, N55/N20 for surface),
    remaps variables, fills missing fields (w=0, clwc=0, derives 2m q from RH),
    and feeds the result through the standard ``harmoniePrepper`` pipeline.

    Args:
        ml_glob : Optional[str]
            Glob pattern for pre-converted N55ML NetCDF files (3D hybrid-level).
        sfc_glob : Optional[str]
            Glob pattern for pre-converted N55 or N20 NetCDF files (surface).
        w_from_continuity : bool
            If True, derive vertical velocity from horizontal wind divergence
            via mass continuity instead of setting w=0. Default False.
    """

    ml_glob: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    sfc_glob: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    w_from_continuity: bool = field(
        default=False, repr=True, metadata={"serialize": True}, init=True
    )
    use_grib: bool | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_std: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_seed: int | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_boundaries: list[str] | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_variables: list[str] | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_minzt: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_maxzt: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )


@dataclass
class Nest_in_Dales:
    """Nest DALES inside another DALES simulation

    Args:
        outpath_coarse : Optional[str]
            Path to the output directory of the coarse parent DALES simulation
            used to source boundary fields for all time steps except t=0.
        outpath_coarse_old : Optional[str]
            Path to the output directory of the previous DALES simulation,
            used to source initial boundary fields at t=0
            from the last time step in the output of the previous simulation
        inpath_coarse : Optional[str]
            Path to the input directory of the previous DALES simulation,
            used to source initial boundary fields at t=0
            from the initfields.nc input of the previous simulation
        inpath : Optional[str]
            Path to the input directory of the current DALES simulation,
            used to source initial fields for the current simulation
    """

    outpath_coarse: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    outpath_coarse_old: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    inpath_coarse: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    inpath: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )


@dataclass
class Nest_in_AtmosphereProfiles:
    """Explicit homogeneous boundary and initialization profiles.

    All six fields are required: u (ua), v (va), w (w), thl (thetal),
    qt (qt), e12 (tke). Each accepts an AtmosphericProfile,
    InterpolatedProfile, or a list of TimedAtmosphereProfile entries including
    time zero. Scalar profile parameters may use independent time axes.
    No simulation providers, nudging targets or default profiles are consulted.
    """

    u: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    v: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    w: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    thl: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    qt: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    e12: AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    tracers: dict[
        str, AtmosphericProfile | InterpolatedProfile | list[TimedAtmosphereProfile]
    ] = field(default_factory=dict)

    def __post_init__(self):
        expected = {
            "u": "ua",
            "v": "va",
            "w": "w",
            "thl": "thetal",
            "qt": "qt",
            "e12": "tke",
        }
        if set(self.tracers) & set(expected):
            raise ValueError("Tracer names must not replace u, v, w, thl, qt or e12")
        inputs = {name: getattr(self, name) for name in expected}
        inputs.update(self.tracers)
        for name, source in inputs.items():
            if not isinstance(source, (AtmosphericProfile, InterpolatedProfile, list)):
                raise TypeError(
                    f"Boundary profile '{name}' requires a profile or a list of timed profiles"
                )
            entries = source if isinstance(source, list) else [source]
            if not entries:
                raise ValueError(f"Boundary profile '{name}' must not be empty")
            if isinstance(source, list) and not all(
                isinstance(entry, TimedAtmosphereProfile) for entry in entries
            ):
                raise TypeError(
                    f"Boundary profile '{name}' lists require TimedAtmosphereProfile entries"
                )
            for entry in entries:
                profile = (
                    entry.profile
                    if isinstance(entry, TimedAtmosphereProfile)
                    else entry
                )
                if not isinstance(profile, (AtmosphericProfile, InterpolatedProfile)):
                    raise TypeError(
                        f"Boundary profile '{name}' requires AtmosphericProfile or InterpolatedProfile"
                    )
                required = expected.get(name, name)
                if profile.variable.name != required:
                    raise ValueError(
                        f"Boundary field '{name}' requires variable '{required}', got '{profile.variable.name}'"
                    )

    add_to_top_thl: float | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )
    noise_std: float | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )
    noise_seed: int | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )
    noise_boundaries: list[str] | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )
    noise_variables: list[str] | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )
    noise_minzt: float | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )
    noise_maxzt: float | None = field(
        default=None,
        repr=True,
        metadata={"serialize": True},
        init=True,
    )


@register_module
@dataclass
class do_openboundary(simulation_module):
    forcing_priority = EXTERNAL_FORCING_PRIORITY

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    openBCgrid: GridDalesOpenBC = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    indices: dict | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    harmonieprepper: prep_harmonie.harmoniePrepper | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    boundaries: xr.Dataset | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    initfields: xr.Dataset | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    nest_in_harmonie: Nest_in_Harmonie | None = field(
        default=None, repr=False, metadata={"serialize": True}, init=True
    )
    nest_in_dales: Nest_in_Dales | None = field(
        default=None, repr=False, metadata={"serialize": True}, init=True
    )
    nest_in_atmosphere: Nest_in_AtmosphereProfiles | None = field(
        default=None, repr=False, metadata={"serialize": True}, init=True
    )
    nest_in_knmi: Nest_in_KNMI | None = field(
        default=None, repr=False, metadata={"serialize": True}, init=True
    )

    e12: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    tracernames: list[str] | None = field(
        default_factory=list, repr=True, metadata={"serialize": True}, init=True
    )
    tchunk: int | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    start: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    time0: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    end: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    lopenbc: bool = field(
        default=True,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lopenbc", "serialize": True},
        init=True,
    )
    linithetero: bool = field(
        default=True,
        repr=True,
        metadata={"nml": "OPENBC", "key": "linithetero", "serialize": True},
        init=True,
    )
    lper: bool = field(
        default=False,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lper", "serialize": True},
        init=True,
    )
    lbuoytop: bool = field(
        default=True,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lbuoytop", "serialize": True},
        init=True,
    )
    dxint: int = field(
        default=100,
        repr=True,
        metadata={"nml": "OPENBC", "key": "dxint", "serialize": True},
        init=True,
    )
    dyint: int = field(
        default=100,
        repr=True,
        metadata={"nml": "OPENBC", "key": "dyint", "serialize": True},
        init=True,
    )
    dzint: int = field(
        default=-1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "dzint", "serialize": True},
        init=True,
    )
    dxturb: int = field(
        default=-1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "dxturb", "serialize": True},
        init=True,
    )
    dyturb: int = field(
        default=-1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "dyturb", "serialize": True},
        init=True,
    )
    taum: int = field(
        default=0,
        repr=True,
        metadata={"nml": "OPENBC", "key": "taum", "serialize": True},
        init=True,
    )
    tauh: int = field(
        default=20,
        repr=True,
        metadata={"nml": "OPENBC", "key": "tauh", "serialize": True},
        init=True,
    )
    pbc: int = field(
        default=3,
        repr=True,
        metadata={"nml": "OPENBC", "key": "pbc", "serialize": True},
        init=True,
    )
    lsynturb: bool = field(
        default=False,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lsynturb", "serialize": True},
        init=True,
    )
    iturb: int = field(
        default=0,
        repr=True,
        metadata={"nml": "OPENBC", "key": "iturb", "serialize": True},
        init=True,
    )
    tau: int = field(
        default=180,
        repr=True,
        metadata={"nml": "OPENBC", "key": "tau", "serialize": True},
        init=True,
    )
    lambda_: int = field(
        default=1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lambda", "serialize": True},
        init=True,
    )
    nmodes: int = field(
        default=100,
        repr=True,
        metadata={"nml": "OPENBC", "key": "nmodes", "serialize": True},
        init=True,
    )
    lambdas: int = field(
        default=1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lambdas", "serialize": True},
        init=True,
    )
    lambdas_x: int = field(
        default=1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lambdas_x", "serialize": True},
        init=True,
    )
    lambdas_y: int = field(
        default=1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lambdas_y", "serialize": True},
        init=True,
    )
    lambdas_z: int = field(
        default=1,
        repr=True,
        metadata={"nml": "OPENBC", "key": "lambdas_z", "serialize": True},
        init=True,
    )
    igrw_damp: int = field(
        default=0,
        repr=True,
        metadata={"nml": "PHYSICS", "key": "igrw_damp", "serialize": True},
        init=True,
    )
    lconstexner: bool = field(
        default=True,
        repr=True,
        metadata={
            "nml": "thermodynamics",
            "key": "lconstexner",
            "serialize": True,
            "raise_conflict": True,
        },
        init=True,
    )
    lbaseexner: bool = field(
        default=True,
        repr=True,
        metadata={
            "nml": "thermodynamics",
            "key": "lbaseexner",
            "serialize": True,
            "raise_conflict": True,
        },
        init=True,
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "do_openboundary"

    def provide_forcings(self) -> ForcingSet:
        """Radiation sounding derived from Harmonie when nesting in Harmonie."""
        if self.nest_in_harmonie is None:
            return ForcingSet()
        self.ensure_prepared()
        return ForcingSet(
            initial=sounding_forcings(self.harmonieprepper.backrad_profile)
        )

    def __add__(self, obj) -> "do_openboundary":
        """Add configurations to open boundary module.

        Args:
            obj: Nest_in_Harmonie, Nest_in_Dales,
                Nest_in_AtmosphereProfiles

        Returns:
            self for chaining
        """
        if isinstance(obj, Nest_in_Harmonie):
            self.nest_in_harmonie = obj
        elif isinstance(obj, Nest_in_KNMI):
            self.nest_in_knmi = obj
        elif isinstance(obj, Nest_in_Dales):
            self.nest_in_dales = obj
        elif isinstance(obj, Nest_in_AtmosphereProfiles):
            self.nest_in_atmosphere = obj

        else:
            raise TypeError(
                "Expected "
                "Nest_in_Harmonie/Nest_in_KNMI/Nest_in_Dales/"
                "Nest_in_AtmosphereProfiles, "
                f"got {type(obj)}"
            )
        return self

    def __iadd__(self, modification) -> "do_openboundary":
        """In-place addition of modification."""
        return self.__add__(modification)

    def prepare_calculation(self):
        if not type(self.grid) is GridDalesOpenBC:
            logger.warning(
                "Passing wrong grid type to openboundary, converting to GridDalesOpenBC with as_openBC."
            )
            self.openBCgrid = self.grid.as_openbc()
        else:
            self.openBCgrid = self.grid
        if self.nest_in_harmonie is not None:
            self._prepare_from_harmonie()
        elif self.nest_in_knmi is not None:
            self._prepare_from_knmi()
        elif self.nest_in_dales is not None:
            self._prepare_from_dales()
        elif self.nest_in_atmosphere is not None:
            self._prepare_from_atmosphere()
        else:
            raise ValueError("Unknown source for open boundary conditions")

    def _prepare_from_harmonie(self) -> None:
        config = OpenBoundaryConfig(
            e12=self.e12,
            tracernames=list(self.tracernames or []),
            tchunk=self.tchunk,
            start=self.start,
            author="author",
            time0=self.time0,
            end=self.end,
            HARMONIE_ml_glob=self.nest_in_harmonie.ml_glob,
            HARMONIE_sfc_glob=self.nest_in_harmonie.sfc_glob,
        )
        self.harmonieprepper = prep_harmonie.harmoniePrepper(config, self.openBCgrid)
        self.harmonieprepper.load_data()
        data, transform = self.harmonieprepper.prep_harmonie()
        # we need to use the right surface pressure as calculated from the input data
        logger.info("Setting namelist NAMSURFACE:ps to %f", self.harmonieprepper.ps)

        self.set_nml_section(
            "NAMSURFACE",
            "ps",
            self.harmonieprepper.ps,
        )
        # we need to use the right skin liq. pot. temperature from the input data
        logger.info("Setting namelist NAMSURFACE:thls to %f", self.harmonieprepper.thls)
        self.set_nml_section(
            "NAMSURFACE",
            "thls",
            self.harmonieprepper.thls,
        )
        (data,) = dask.optimize(data)

        logger.debug("Setting up boundary fields")
        self.boundaries = harmonie_boundary.boundary_fields(
            config,
            self.openBCgrid,
            data,
            output_path=self.output_path,
        )
        logger.debug("Setting up initial fields")
        self.initfields = initfields.initial_fields(
            config,
            self.openBCgrid,
            data,
            transform,
            output_path=self.output_path,
        )
        logger.debug("Setup all openBC fields, optimizing fields now..")
        self.boundaries, self.initfields = dask.optimize(
            self.boundaries, self.initfields
        )
        logger.debug("Optimized fields")

    def _prepare_from_knmi(self) -> None:
        worker = OpenBCKNMIWorker(self)
        self.boundaries, self.initfields = worker.prepare()

    def _prepare_from_dales(self) -> None:
        config = OpenBoundaryConfig(
            e12=self.e12,
            tracernames=list(self.tracernames or []),
            tchunk=self.tchunk,
            lsynturb=self.lsynturb,
            start=self.start,
            time0=self.time0,
            author="author",
            end=self.end,
            outpath_coarse=self.nest_in_dales.outpath_coarse,
            outpath_coarse_old=self.nest_in_dales.outpath_coarse_old,
            inpath_coarse=self.nest_in_dales.inpath_coarse,
            inpath=self.nest_in_dales.inpath,
        )
        crosssection_chunks = None
        if self.tchunk is not None:
            crosssection_chunks = {"time": int(self.tchunk)}

        if self.nest_in_dales.inpath is not None:
            self.initfields = initial_fields_fine.initial_fields_fine(
                config,
                grid=self.openBCgrid,
                output_path=self.output_path,
            )
        else:
            logger.warning(
                "Nest_in_Dales configured without 'inpath': initfields.inp.*.nc "
                "will not be created from interpolated parent initial fields. "
                "A minimal placeholder initfields dataset will be written instead."
            )

            self.initfields = None
            self.linithetero = False

        self.boundaries = boundary_fields_fine.boundary_fields_fine(
            config,
            grid=self.openBCgrid,
            output_path=self.output_path,
            grid_indices=self.indices,
            chunks=crosssection_chunks,
        )

    def _prepare_from_atmosphere(self) -> None:
        worker = OpenBCAtmosphereWorker(self)
        self.boundaries, self.initfields = worker.prepare()

    def write_files(self):
        # Save data
        logger.debug("Optimizing initial fields")
        if self.initfields:
            self.initfields = self.initfields.assign_attrs(
                {
                    "title": f"initfields.inp.{self.exp_id:03d}.nc",
                }
            )
            initfields_writer = self.initfields.to_netcdf(
                path=self.output_path / "input" / self.initfields.attrs["title"],
                mode="w",
                format="netcdf4",
                compute=False,
            )
            (initfields_writer,) = dask.optimize(initfields_writer)
            logger.debug("Writing initial fields")

            initfields_writer.compute()

        self.boundaries = self.boundaries.assign_attrs(
            {
                "title": f"openboundaries.inp.{self.exp_id:03d}.nc",
            }
        )
        logger.debug("Optimizing boundaries")
        openboundaries_writer = self.boundaries.to_netcdf(
            path=self.output_path / "input" / self.boundaries.attrs["title"],
            mode="w",
            format="netcdf4",
            compute=False,
        )
        (openboundaries_writer,) = dask.optimize(openboundaries_writer)
        logger.debug("Writing boundaries")

        openboundaries_writer.compute()
