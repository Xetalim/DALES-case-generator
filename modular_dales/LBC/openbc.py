"""Open Boundary Conditions module for lateral boundary forcing."""

import logging
from dataclasses import dataclass, field

import dask
import xarray as xr

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
from modular_dales.LBC.nesting_idx import NestingIndices
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.modular.forcing import (
    EXTERNAL_FORCING_PRIORITY,
    ForcingSet,
    sounding_forcings,
)
from modular_dales.modular.namelist import nml_field
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


@dataclass(kw_only=True)
class _HarmonieOpenBCOptions:
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


@dataclass(kw_only=True)
class _DalesOpenBCOptions:
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


@register_module
@dataclass
class do_openboundary(simulation_module):
    forcing_priority = EXTERNAL_FORCING_PRIORITY
    e12: float = field(repr=True, metadata={"serialize": True}, init=True)
    openBCgrid_: GridDalesOpenBC | None = field(
        default=None, repr=False, metadata={"serialize": False}, init=False
    )
    indices: NestingIndices | None = field(
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
    lopenbc: bool = nml_field(
        default=True,
        repr=True,
        section="OPENBC",
        key="lopenbc",
        metadata={"serialize": True},
        init=True,
    )
    linithetero: bool = nml_field(
        default=True,
        repr=True,
        section="OPENBC",
        key="linithetero",
        metadata={"serialize": True},
        init=True,
    )
    lper: bool = nml_field(
        default=False,
        repr=True,
        section="OPENBC",
        key="lper",
        metadata={"serialize": True},
        init=True,
    )
    lbuoytop: bool = nml_field(
        default=True,
        repr=True,
        section="OPENBC",
        key="lbuoytop",
        metadata={"serialize": True},
        init=True,
    )
    dxint: int = nml_field(
        default=100,
        repr=True,
        section="OPENBC",
        key="dxint",
        metadata={"serialize": True},
        init=True,
    )
    dyint: int = nml_field(
        default=100,
        repr=True,
        section="OPENBC",
        key="dyint",
        metadata={"serialize": True},
        init=True,
    )
    dzint: int = nml_field(
        default=-1,
        repr=True,
        section="OPENBC",
        key="dzint",
        metadata={"serialize": True},
        init=True,
    )
    dxturb: int = nml_field(
        default=-1,
        repr=True,
        section="OPENBC",
        key="dxturb",
        metadata={"serialize": True},
        init=True,
    )
    dyturb: int = nml_field(
        default=-1,
        repr=True,
        section="OPENBC",
        key="dyturb",
        metadata={"serialize": True},
        init=True,
    )
    taum: int = nml_field(
        default=0,
        repr=True,
        section="OPENBC",
        key="taum",
        metadata={"serialize": True},
        init=True,
    )
    tauh: int = nml_field(
        default=20,
        repr=True,
        section="OPENBC",
        key="tauh",
        metadata={"serialize": True},
        init=True,
    )
    pbc: int = nml_field(
        default=3,
        repr=True,
        section="OPENBC",
        key="pbc",
        metadata={"serialize": True},
        init=True,
    )
    lsynturb: bool = nml_field(
        default=False,
        repr=True,
        section="OPENBC",
        key="lsynturb",
        metadata={"serialize": True},
        init=True,
    )
    iturb: int = nml_field(
        default=0,
        repr=True,
        section="OPENBC",
        key="iturb",
        metadata={"serialize": True},
        init=True,
    )
    tau: int = nml_field(
        default=180,
        repr=True,
        section="OPENBC",
        key="tau",
        metadata={"serialize": True},
        init=True,
    )
    lambda_: int = nml_field(
        default=1,
        repr=True,
        section="OPENBC",
        key="lambda",
        metadata={"serialize": True},
        init=True,
    )
    nmodes: int = nml_field(
        default=100,
        repr=True,
        section="OPENBC",
        key="nmodes",
        metadata={"serialize": True},
        init=True,
    )
    lambdas: int = nml_field(
        default=1,
        repr=True,
        section="OPENBC",
        key="lambdas",
        metadata={"serialize": True},
        init=True,
    )
    lambdas_x: int = nml_field(
        default=1,
        repr=True,
        section="OPENBC",
        key="lambdas_x",
        metadata={"serialize": True},
        init=True,
    )
    lambdas_y: int = nml_field(
        default=1,
        repr=True,
        section="OPENBC",
        key="lambdas_y",
        metadata={"serialize": True},
        init=True,
    )
    lambdas_z: int = nml_field(
        default=1,
        repr=True,
        section="OPENBC",
        key="lambdas_z",
        metadata={"serialize": True},
        init=True,
    )
    igrw_damp: int = nml_field(
        default=0,
        repr=True,
        section="PHYSICS",
        key="igrw_damp",
        metadata={"serialize": True},
        init=True,
    )
    # lconstexner: bool = nml_field(
    #     default=True,
    #     repr=True,
    #     section="thermodynamics",
    #     key="lconstexner",
    #     raise_conflict=True,
    #     metadata={"serialize": True},
    #     init=True,
    # )
    # lbaseexner: bool = nml_field(
    #     default=True,
    #     repr=True,
    #     section="thermodynamics",
    #     key="lbaseexner",
    #     raise_conflict=True,
    #     metadata={"serialize": True},
    #     init=True,
    # )

    module_name: str = field(
        default="do_openboundary", init=False, metadata={"serialize": False}
    )

    @property
    def openBCgrid(self) -> GridDalesOpenBC:
        if self.openBCgrid_ is None:
            self.openBCgrid_ = self.grid.as_openbc()
        return self.openBCgrid_

    def provide_forcings(self) -> ForcingSet:
        """Return no source-derived forcings for the abstract base module."""
        return ForcingSet()

    def prepare_calculation(self) -> None:
        raise NotImplementedError(
            "Instantiate a source-specific open-boundary module instead"
        )

    def write_files(self) -> None:
        if self.initfields is None:
            raise RuntimeError("Initial fields not generated yet")
        if self.boundaries is None:
            raise RuntimeError("Boundary fields not generated yet")
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


@register_module
@dataclass
class Nest_in_Harmonie(do_openboundary, _HarmonieOpenBCOptions):
    module_name: str = field(default="Nest_in_Harmonie", init=False)

    def prepare_calculation(self) -> None:
        config = OpenBoundaryConfig(
            e12=self.e12,
            tracernames=list(self.tracernames or []),
            tchunk=self.tchunk,
            start=self.start,
            author="author",
            time0=self.time0,
            end=self.end,
            HARMONIE_ml_glob=self.ml_glob,
            HARMONIE_sfc_glob=self.sfc_glob,
        )
        self.harmonieprepper = prep_harmonie.harmoniePrepper(config, self.openBCgrid)
        self.harmonieprepper.load_data()
        data, transform = self.harmonieprepper.prep_harmonie()
        logger.info("Setting namelist NAMSURFACE:ps to %f", self.harmonieprepper.ps)
        self.set_nml_section("NAMSURFACE", "ps", self.harmonieprepper.ps)
        logger.info("Setting namelist NAMSURFACE:thls to %f", self.harmonieprepper.thls)
        self.set_nml_section("NAMSURFACE", "thls", self.harmonieprepper.thls)
        (data,) = dask.optimize(data)

        self.boundaries = harmonie_boundary.boundary_fields(
            config, self.openBCgrid, data
        )
        self.initfields = initfields.initial_fields(
            config,
            self.openBCgrid,
            data,
            transform,
            output_path=self.output_path,
        )
        self.boundaries, self.initfields = dask.optimize(
            self.boundaries, self.initfields
        )

    def provide_forcings(self) -> ForcingSet:
        self.ensure_prepared()
        if self.harmonieprepper is None:
            raise RuntimeError("HARMONIE open-boundary preparation did not complete")
        if self.harmonieprepper.backrad_profile is None:
            raise RuntimeError("HARMONIE backrad profile not generated yet")
        return ForcingSet(
            initial=sounding_forcings(self.harmonieprepper.backrad_profile)
        )


@register_module
@dataclass
class Nest_in_Dales(do_openboundary, _DalesOpenBCOptions):
    module_name: str = field(default="Nest_in_Dales", init=False)

    def prepare_calculation(self) -> None:
        if self.indices is None:
            raise ValueError(
                "Nesting indices must be provided for Nest_in_Dales. Use a NestingTopology for this."
            )
        config = OpenBoundaryConfig(
            e12=self.e12,
            tracernames=list(self.tracernames or []),
            tchunk=self.tchunk,
            lsynturb=self.lsynturb,
            start=self.start,
            time0=self.time0,
            author="author",
            end=self.end,
            outpath_coarse=self.outpath_coarse,
            outpath_coarse_old=self.outpath_coarse_old,
            inpath_coarse=self.inpath_coarse,
            inpath=self.inpath,
        )
        crosssection_chunks = (
            {"time": int(self.tchunk)} if self.tchunk is not None else None
        )

        if self.inpath is not None:
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
            grid_indices=self.indices,
            chunks=crosssection_chunks,
        )
