# Crops the HARMONIE data to the required time and spatial extends.
# Transforms pressure coordinates into height levels.
# Transforms HARMONIE prognostic variables to DALES prognostic variables
import logging

import dask.base
import numpy as np
import xarray as xr
from dask.delayed import delayed

from MoDa.Geometry import GridDalesOpenBC
from MoDa.IO_helpers.raster import fix_lambert_offsets
from MoDa.LBC.nest_dales_in_HARMONIE import hybrid_levels
from MoDa.LBC.nest_dales_in_HARMONIE.helper import (
    calcBaseprof,
    differentiate,
)
from MoDa.LBC.nest_dales_in_HARMONIE.Transform import Transform
from MoDa.LBC.openboundary_config import OpenBoundaryConfig
from MoDa.logging_wrapper import logwrap

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


xr.set_options(use_new_combine_kwarg_defaults=True)  # type: ignore[no-untyped-call]
DATA_LAT_VAR = "latitude"
DATA_LON_VAR = "longitude"
# Constants
# Simulation constants (modglobal)
p0 = 1e5  # Reference pressure
Rd = 287.04  # Gas constant for dry air
Rv = 461.5  # Gas constant for water vapor
cp = 1004.0  # Specific heat at constant pressure (dry air)
Lv = 2.53e6  # Latent heat for vaporisation
grav = 9.81  # Gravitational constant
kappa = 0.4  # Von Karman constant
omega = 7.2921e-5  # Earth's angular velocity


class harmoniePrepper:
    """
    A class for preparing HARMONIE data for DALES (Dutch Atmospheric Large Eddy Simulation) nesting.
    This class handles the complete preprocessing pipeline for HARMONIE atmospheric data,
    including loading, interpolation, thermodynamic calculations, and optional synthetic
    turbulence generation.
    Attributes:
        input_json (OpenBoundaryConfig | Mapping[str, Any]):
            Configuration object containing input parameters and settings.
        grid (GridDalesOpenBC): DALES grid object defining the target domain.
        ps (float): Surface pressure value used for base profile calculations.
        thls (float): Liquid potential temperature at surface.
        data (xr.Dataset): xarray Dataset containing atmospheric variables.
        transform (rasterio.transform.Affine): Coordinate transformation parameters.
        p (xr.DataArray): Pressure field calculated from input data.
        z (xr.DataArray): 3D height levels.
        z_int (xr.DataArray): Reference height levels interpolated to DALES grid.
        exnrs (float): Exner pressure function at surface.
        exnr (xr.DataArray): 3D Exner pressure function field.
        thl (xr.DataArray): Liquid potential temperature field.
    Methods:
        load_data(): Loads HARMONIE data and creates xarray dataset with required variables.
        prep_harmonie(): Executes the complete preprocessing pipeline including pressure
            calculation, height interpolation, thermodynamic variable calculations, and
            optional synthetic turbulence parameterization.
    Returns:
        tuple: (processed_data, transform) - xarray Dataset with DALES-ready variables
            and coordinate transformation parameters.
    """

    def __init__(
        self,
        input_json: OpenBoundaryConfig,
        grid: GridDalesOpenBC,
    ):
        self.input_json = input_json
        self.grid = grid
        self.ps: float | None = None
        self.thls: float | None = None
        self.data: xr.Dataset | None = None
        self.transform: Transform | None = None
        self.p: xr.DataArray | None = None
        self.z: xr.DataArray | None = None
        self.z_int: xr.DataArray | None = None
        self.exnrs: float | None = None
        self.exnr: np.ndarray | xr.DataArray | None = None
        self.thl: xr.DataArray | None = None
        self.backrad_profile: xr.Dataset | None = None
        self.ug: xr.DataArray | None = None
        self.vg: xr.DataArray | None = None

    def load_data(self) -> None:
        variables = ["ua", "va", "wa", "ta", "hus", "clw", "ps", "tas", "huss"]
        if self.input_json.synturb is not None:
            variables = variables + ["tke", "tauu", "tauv", "cb", "hfss"]
        self.data, self.transform, x_sw, y_sw = create_xarray_dataset(
            self.input_json, self.grid, variables
        )

        (self.data,) = dask.base.optimize(self.data)
        # Change transform parameters to new DALES origin and update transform
        self.transform = update_transform(self.transform, x_sw, y_sw)

    @logwrap
    def prep_harmonie(self) -> tuple[xr.Dataset, Transform]:
        if self.data is None:
            raise ValueError(
                "Data not loaded. Call load_data() before prep_harmonie()."
            )
        if self.transform is None:
            raise ValueError("Coordinate transform was not initialized")
        # Calculate pressure levels
        self.p = calculate_pressure(self.data)

        self.data = self.data.assign({"p": self.p})

        # Add missing surface fields to 3d fields
        variables = ["uas", "vas", "was", "clws"]
        if self.input_json.synturb is not None:
            variables.append("tkes")
        self.data = self.data.assign(
            {var: xr.zeros_like(self.data["ps"]) for var in variables}
        )

        if self.input_json.synturb is not None:
            turbulence_data_dic = {
                "tauu": self.data["tauu"],
                "tauv": self.data["tauv"],
                "hfss": self.data["hfss"],
                "cb": self.data["cb"],
            }

        # Concatenate surface and 3D fields
        variables = ["ua", "va", "wa", "ta", "hus", "clw", "p"]
        if self.input_json.synturb is not None:
            variables.append("tke")
        self.data = merge_steps(self.data, variables)
        # Calculate 3D height levels

        self.z = calculate_3d_height_levels(self.data).persist()
        self.data = self.data.assign({"z3d": self.z})
        # Get reference height levels (mean of height field first time step) and crop to grid.zsize
        self.z_int = get_ref_height_crop(
            self.input_json, self.grid, self.data
        ).compute()

        # self.z_int = self.z_int.persist()

        print(self.z_int)

        # Interpolate data to reference height levels
        self.data = interpolate_ref_height(self.input_json, self.data, self.z_int)

        self.backrad_profile = _build_backrad_profile_from_harmonie(self.data)

        # make sure z_int is also dimension z now..
        self.z_int = self.z_int.rename({"lev": "z"})

        # Calculate qt
        self.data = self.data.assign({"qt": self.data["clwc"] + self.data["q"]})

        self.ug, self.vg = calculate_geostrophic_wind(self.data, self.grid)

        # Calculate base profiles and exnr function
        self.ps, self.exnrs, self.thls, self.exnr = calc_base_exner(
            self.input_json, self.grid, self.data, self.z_int
        )
        logger.debug("Getting surface pressure and temperature")
        self.ps = float(self.ps)  # save the ps value used for the profile
        self.thls = float(self.thls)  # and thls
        logger.debug("Creating exnr dataarray")
        ps, exnrs, thls, exnr = self.ps, self.exnrs, self.thls, self.exnr
        if ps is None or exnrs is None or thls is None or exnr is None:
            raise RuntimeError("Base Exner profile was not calculated")
        self.exnr = xr.DataArray(
            np.concatenate([[exnrs], np.asarray(exnr)]),
            dims=["z"],
            coords={"z": self.z_int},
            name="exnr",
            attrs={"thls": thls, "ps": ps},
        )
        logger.debug("Assigning exnr dataarray")
        self.data = self.data.assign({"exnr": self.exnr})
        logger.debug("Calculating liquid potential temperature")
        # Calculate liquid potential temperature and total specific humidity
        self.thl = (
            self.data["t"] / self.exnr
            - Lv * self.data["clwc"] / (cp * self.exnr).persist()
        )
        logger.debug("Assigning liquid potential temperature")
        self.data = self.data.assign({"thl": self.thl})

        logger.debug("Calculating synthetic turbulence parameters")
        # Calculate turbulence parameters
        if self.input_json.synturb is not None:
            calculate_turbulence_vars(
                self.grid,
                self.data,
                turbulence_data_dic,
                self.z_int,
                ps,
                exnrs,
                thls,
                self.thl,
            )
        logger.debug(
            "Organizing data, renaming variables, dropping non-DALES variables and setting transform"
        )
        # Organize data, rename variables and drop non DALES prognostic variables
        self.data = (
            self.data.rename({"wz": "w"})
            .drop(["t", "p", "clwc", "q"])
            .assign(
                {
                    "transform": xr.DataArray(
                        [], name="Lambert_Conformal", attrs=self.transform.parameters
                    )
                }
            )
        )
        logger.debug("Optimizing data")
        (self.data,) = dask.base.optimize(self.data)

        return self.data, self.transform


def _build_backrad_profile_from_harmonie(data: xr.Dataset) -> xr.Dataset:
    """Build a domain-mean pressure sounding for DALES backrad input."""
    for key in ("p", "t", "q"):
        if key not in data:
            raise KeyError(
                f"HARMONIE dataset does not contain required variable '{key}'"
            )

    p = data["p"].isel(time=0, drop=True).mean(dim=["x", "y"], skipna=True)
    t = data["t"].isel(time=0, drop=True).mean(dim=["x", "y"], skipna=True)
    q = data["q"].isel(time=0, drop=True).mean(dim=["x", "y"], skipna=True)

    p_vals = np.asarray(p.values, dtype=float)
    t_vals = np.asarray(t.values, dtype=float)
    q_vals = np.asarray(q.values, dtype=float)

    order = np.argsort(p_vals)[::-1]
    p_vals = p_vals[order]
    t_vals = t_vals[order]
    q_vals = q_vals[order]

    o3_vals = np.zeros_like(p_vals)

    return xr.Dataset(
        data_vars={
            "T": ("lev", t_vals),
            "q": ("lev", q_vals),
            "o3": ("lev", o3_vals),
        },
        coords={"lev": ("lev", p_vals)},
    )


@logwrap
def calculate_geostrophic_wind(
    data: xr.Dataset, grid: GridDalesOpenBC
) -> tuple[xr.DataArray, xr.DataArray]:
    """Geostrophic wind on ``(time, z, y, x)`` from horizontal pressure gradients on height levels."""
    if DATA_LAT_VAR in data.coords:
        lat_values: xr.DataArray | float = data[DATA_LAT_VAR]
    elif getattr(grid, "xlat", None) is not None:
        lat_values = float(grid.xlat)
    else:
        raise ValueError(
            "Cannot compute geostrophic wind: no latitude in HARMONIE data and grid.xlat is not set"
        )
    fc = 2.0 * omega * np.sin(np.deg2rad(lat_values))
    rho = data["p"] / (
        Rd
        * data["t"]
        * (1 + (Rv / Rd - 1) * (data["q"] + data["clwc"]) - Rv / Rd * data["clwc"])
    )
    ug = -data["p"].differentiate("y") / (rho * fc)
    vg = data["p"].differentiate("x") / (rho * fc)
    return ug.rename("ug"), vg.rename("vg")


@logwrap
def calculate_pressure(data: xr.Dataset) -> xr.DataArray:
    # right now this function uses 90 hybrid model levels.
    # hybrid_coeff = np.loadtxt("<inpath>/H43_65lev.txt")
    hybrid_A = xr.DataArray(
        hybrid_levels.ahalf, dims=["lev"], coords={"lev": np.arange(1, 92)}
    )
    hybrid_B = xr.DataArray(
        hybrid_levels.bhalfs, dims=["lev"], coords={"lev": np.arange(1, 92)}
    )
    ph = (hybrid_A + hybrid_B * data["ps"]).transpose(*data["u"].dims)
    # calculate on pressure levels
    p = 0.5 * (ph.assign_coords({"lev": ph["lev"] - 1}) + ph)
    return p.sel(lev=data["lev"]).chunk({"time": 1, "lev": -1})


@logwrap
def merge_steps(data: xr.Dataset, variables: list[str]) -> xr.Dataset:
    dic = {
        "ua": "u",
        "va": "v",
        "wa": "wz",
        "ta": "t",
        "hus": "q",
        "clw": "clwc",
        "ps": "ps",
        "tas": "2t",
        "huss": "2sh",
        "p": "p",
    }
    data = data[[dic[var] for var in variables]]

    return data


@logwrap
def get_ref_height_crop(
    input_json: OpenBoundaryConfig, grid: GridDalesOpenBC, data: xr.Dataset
) -> xr.DataArray:
    if input_json.start == input_json.time0:  # Define reference height levels
        z_int = (
            data["z3d"]
            .isel({"time": 0}, drop=True)
            .sel(x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1]))
            .mean(dim=["x", "y"])
        )
    else:  # Take reference height levels from exnr.inp.xxx
        if input_json.exnr_file is None:
            raise ValueError("exnr_file is required when start differs from time0")
        try:
            exnr = np.loadtxt(input_json.exnr_file, skiprows=1)
        except FileNotFoundError:
            logger.critical(
                "No reference height levels found in exnr_file in configuration. This is required as the simulation start is not the same as the HARMONIE start."
            )
            raise
        z_int = xr.DataArray(exnr[:, 0], dims=("lev",))
    return z_int


@logwrap
def calculate_turbulence_vars(
    grid: GridDalesOpenBC,
    data: xr.Dataset,
    turbulence_data_dic: dict[str, xr.DataArray],
    z_int: xr.DataArray,
    ps_exnr: float,
    exnrs: float,
    thls_exnr: float,
    thl: xr.DataArray,
) -> None:

    # Calculate inversion height from maximum curvature and with cloud base as a backup
    zi_min = 200
    zi_max = 4000
    thlmean = thl.sel(
        x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1])
    ).mean(dim=["x", "y"])
    cbmean = (
        xr.where(turbulence_data_dic["cb"] > 0.0, turbulence_data_dic["cb"], np.nan)
        .sel(x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1]))
        .mean(dim=["x", "y"])
    )
    its = 0
    d2thl = []
    d1thl = []
    rhobf = calcBaseprof(
        np.asarray(z_int.values, dtype=np.float64), thls_exnr, ps_exnr, pref0=p0
    )
    chunks = thlmean.chunks
    if chunks is None:
        raise ValueError("Synthetic turbulence requires chunked profile data")
    for tchunk in chunks[0]:
        ite = its + tchunk
        d1thl.append(
            delayed(differentiate)(
                thlmean.isel({"time": np.arange(its, ite)}), "z", 1, acc=6
            )
        )
        d2thl.append(
            delayed(differentiate)(
                thlmean.isel({"time": np.arange(its, ite)}), "z", 2, acc=6
            )
        )
        its = ite
    d1thl = delayed(xr.concat)(d2thl, dim="time").compute()
    d2thl = delayed(xr.concat)(d1thl, dim="time").compute()
    zi = d2thl.where(d1thl > 0).sel(z=slice(zi_min, zi_max)).idxmax("z").fillna(cbmean)
    rhobs = rhobf[0] - z_int[0] * (rhobf[1] - rhobf[0]) / (z_int[1] - z_int[0])
    ustar = np.sqrt(np.maximum(turbulence_data_dic["tauu"], 0) / rhobs).rename("ustar")
    vstar = np.sqrt(np.maximum(turbulence_data_dic["tauv"], 0) / rhobs).rename("vstar")
    wthls = (turbulence_data_dic["hfss"] / (exnrs * rhobs * cp)).rename("wthls")
    data = data.assign({"ustar": ustar, "vstar": vstar, "wthls": wthls, "zi": zi})


@logwrap
def calc_base_exner(
    input_json: OpenBoundaryConfig,
    grid: GridDalesOpenBC,
    data: xr.Dataset,
    z_int: xr.DataArray,
) -> tuple[float, float, float, np.ndarray]:
    if input_json.start == input_json.time0:  # Calculate exnr function
        z_min = data.z.argmin(dim="z")

        logger.debug("Calculating profile means")
        t0_values = (
            data[["t", "p", "qt", "clwc"]]
            .isel({"time": 0}, drop=True)
            .sel(x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1]))
            .mean(dim=["x", "y"])
        ).persist()

        surface_values = t0_values.isel({"z": z_min}, drop=True)
        air_values = t0_values.isel(z=slice(1, None, None))

        logger.debug("Calculating surface temperature")
        tas_exnr = (surface_values["t"]).compute()
        logger.debug("Calculating surface pressure")
        ps_exnr = (surface_values["p"]).compute()

        exnrs = (ps_exnr / p0) ** (Rd / cp)
        thls_exnr = tas_exnr / exnrs
        # somehow this function doesn't like being dasked. max size of array going in is (lev,) or (1,) so shouldn't be too big of a problem?
        # if ever you have performance issues, look at among others this function..
        rhobf = calcBaseprof(
            np.asarray(z_int.values, dtype=np.float64),
            float(thls_exnr),
            float(ps_exnr),
            pref0=p0,
        )
        p_exnr = (
            rhobf[1:]
            * Rd
            * air_values["t"]
            * (1 + (Rv / Rd - 1) * air_values["qt"] - Rv / Rd * air_values["clwc"])
        )  # Ideal gas law
        exnr = np.asarray((p_exnr / p0) ** (Rd / cp), dtype=np.float64)
        ps_value = float(ps_exnr)
        exnrs_value = float(exnrs)
        thls_value = float(thls_exnr)
    else:  # Read exnr.inp.xxx
        if input_json.exnr_file is None:
            raise ValueError("exnr_file is required when start differs from time0")
        try:
            with open(input_json.exnr_file, "r") as file:
                line0 = file.readline()
        except FileNotFoundError:
            logger.critical(
                "No reference height levels found in exnr_file in configuration. This is required as the simulation start is not the same as the HARMONIE start."
            )
            raise
        thls_value = float(line0.split(",")[1].split("thls = ")[-1])
        ps_value = float(line0.split(",")[2].split("ps = ")[-1])
        exnr_values = np.loadtxt(input_json.exnr_file, skiprows=1)
        exnrs_value = float(exnr_values[0, 1])
        exnr = np.asarray(exnr_values[1:, 1], dtype=np.float64)
    return ps_value, exnrs_value, thls_value, exnr


@logwrap
def vertical_interp_all(
    ds: xr.Dataset, z3d: xr.DataArray, z_new: xr.DataArray
) -> xr.Dataset:
    """
    Interpolate all variables in ds along a 4D vertical coordinate z3d
    onto target heights z_new.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dims (time, x, y, lev)
    z3d : xarray.DataArray
        4D array with actual heights (time, x, y, lev)
    z_new : 1D array
        Target vertical levels (physical heights)

    Returns
    -------
    ds_interp : xarray.Dataset
        Dataset with all variables interpolated to z_new
    """
    z_target = z_new

    def _interp_func(
        var_col: np.ndarray,
        z_col: np.ndarray,
        z_target: xr.DataArray = z_target,
    ) -> np.ndarray:
        # var_col, z_col: 1D arrays of shape (lev,)
        return np.asarray(np.interp(z_target, z_col, var_col))

    interp_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        # Apply vertical interpolation along lev dimension
        da_interp = xr.apply_ufunc(
            _interp_func,
            da,
            z3d,
            input_core_dims=[["lev"], ["lev"]],
            output_core_dims=[["lev"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[da.dtype],
        )
        da_interp = da_interp.assign_coords(lev=z_target)
        interp_vars[var] = da_interp

    # Build new dataset
    ds_interp = xr.Dataset(
        interp_vars, coords={k: v for k, v in ds.coords.items() if k != "lev"}
    )
    ds_interp = ds_interp.assign_coords(lev=z_target)

    return ds_interp.rename({"lev": "z"})


@logwrap
def interpolate_ref_height(
    input_json: OpenBoundaryConfig, data: xr.Dataset, z_int: xr.DataArray
) -> xr.Dataset:
    # data_intz = []
    # its = 0
    # new_shape = [data.sizes["time"], len(z_int), data.sizes["y"], data.sizes["x"]]
    # new_coords = {
    #     "time": data.coords["time"],
    #     "z": z_int,
    #     "y": data.coords["y"],
    #     "x": data.coords["x"],
    # }
    # new_dims = ["time", "z", "y", "x"]
    variables = ["ua", "va", "wa", "ta", "p", "hus", "clw"]
    # dic = {
    #     "ua": "u",
    #     "va": "v",
    #     "wa": "wz",
    #     "ta": "t",
    #     "hus": "q",
    #     "clw": "clwc",
    #     "ps": "ps",
    #     "tas": "2t",
    #     "huss": "2sh",
    #     "p": "p",
    # }
    if input_json.synturb is not None:
        variables.append("tke")
    logger.debug("Checking if data is ascending..")
    z_col = data["z3d"].isel(time=0, x=0, y=0, lev=slice(0, 2))
    z0 = z_col.isel(lev=0)
    z1 = z_col.isel(lev=1)
    logger.debug("Checking if data is ascending.. SUCCEEDED")
    # descending = float(z1) < float(z0)
    if float(z1) < float(z0):
        data = data.isel(lev=slice(None, None, -1))
        logger.debug("Successfully inverted data!")
    else:
        pass

    data = vertical_interp_all(data, data["z3d"], z_int)
    return data


@logwrap
def calculate_3d_height_levels(data: xr.Dataset) -> xr.DataArray:
    rho = data["p"] / (
        Rd
        * data["t"]
        * (1 + (Rv / Rd - 1) * (data["q"] + data["clwc"]) - Rv / Rd * data["clwc"])
    )
    rhoh = 0.5 * (rho + rho.shift(lev=-1))
    pdiff = data["p"].diff(dim="lev", label="lower")
    dz = pdiff / (rhoh * grav)

    zeros = xr.zeros_like(rho.isel(lev=data.lev.argmax(dim="lev")))

    z = (
        (
            xr.concat([dz, zeros], dim="lev")
            .sortby("lev", ascending=False)
            .cumsum(dim="lev")
            .sortby("lev", ascending=True)
        )
        .transpose(*data["u"].dims)
        .chunk({"time": 1, "lev": -1})
    )
    return z


# @logwrap
# def calculate_pressure(input_json, data):
#     # right now this function uses 90 hybrid model levels.
#     # hybrid_coeff = np.loadtxt("<inpath>/H43_65lev.txt")
#     hybrid_A = xr.DataArray(
#         hybrid_levels.ahalf, dims=["lev"], coords={"lev": np.arange(1, 92)}
#     )
#     hybrid_B = xr.DataArray(
#         hybrid_levels.bhalfs, dims=["lev"], coords={"lev": np.arange(1, 92)}
#     )
#     ph = (hybrid_A + hybrid_B * data["ps"]).transpose(*data["u"].dims)
#     # calculate on pressure levels
#     p = 0.5 * (ph.assign_coords({"lev": ph["lev"] - 1}) + ph)
#     return p.sel(lev=data["lev"]).chunk(data["u"].chunks)


@logwrap
def update_transform(transform: Transform, x_sw: float, y_sw: float) -> Transform:
    logger.debug("Skipping transform update, we are keeping the original transform.")
    return transform
    transform.parameters["false_easting"] = transform.parameters["false_easting"] - x_sw
    transform.parameters["false_northing"] = (
        transform.parameters["false_northing"] - y_sw
    )
    proj4 = ""
    for param in transform.parameters["proj4"][1:].split("+"):
        line = "+" + param
        if "x_0" in param:
            line = f"+x_0={transform.parameters['false_easting']} "
        if "y_0" in param:
            line = f"+y_0={transform.parameters['false_northing']} "
        proj4 = proj4 + line
    transform.parameters["proj4"] = proj4.rstrip()
    transform = Transform(transform.parameters)
    return transform


@logwrap
def create_xarray_dataset(
    input_json: OpenBoundaryConfig, grid: GridDalesOpenBC, variables: list[str]
) -> tuple[xr.Dataset, Transform, float, float]:
    if input_json.HARMONIE_ml_glob is None or input_json.HARMONIE_sfc_glob is None:
        raise ValueError("Both HARMONIE model-level and surface paths are required")
    time_chunk = input_json.tchunk if input_json.tchunk is not None else -1
    preprocess = True

    data: list[xr.DataArray] = []
    # Get time epochs

    # Open data and crop data

    # first get the time and transform data....
    x_sw, y_sw = grid.x0, grid.y0
    var = variables[0]
    ds_ml = fix_lambert_offsets(
        xr.open_mfdataset(
            input_json.HARMONIE_ml_glob,
            decode_coords="all",
            engine="netcdf4",
            # parallel=False,
            chunks={"time": time_chunk, "lev": -1},
            # chunks={"x": "auto", "y": "auto", "time": "auto", "lev": -1},
        )
    )
    transform, _, _, time = get_transform_time(var, ds_ml)

    ds_sfc = fix_lambert_offsets(
        xr.open_mfdataset(
            input_json.HARMONIE_sfc_glob,
            decode_coords="all",
            # parallel=False,
            engine="netcdf4",
            chunks={"time": time_chunk},
        )
    )

    logger.debug("Succesfully read in HARMONIE_ml_glob")

    logger.debug("Interpolating sfc time dimension to ml time dimension")
    if preprocess:
        # interpolate surface fluxes to higher time resolution, without assuming correct sorting as I've had problems with this before.
        ds_sfc = ds_sfc.interp(
            time=time,
            assume_sorted=True,
            kwargs={"fill_value": "extrapolate"},
        )
    logger.debug("Succesfully read in and interpolated HARMONIE_sfc_glob")
    if preprocess:
        for var_raw in variables:
            var = {
                "ua": "u",
                "va": "v",
                "wa": "wz",
                "ta": "t",
                "hus": "q",
                "clw": "clwc",
                "ps": "ps",
                "tas": "2t",
                "huss": "2sh",
            }[var_raw]
            logger.debug(f"Reading in variable {var}")

            # Crop data to time and spatial range, using harmonie spatial resolution or filter as buffer
            dx = ds_ml["x"][1] - ds_ml["x"][0]
            # dy = ds_ml["y"][1] - ds_ml["y"][0]

            if (
                input_json.filter is not None
            ):  # add some extra width for gaussian filtering
                buffer = 4 * input_json.filter["sigma"]
            else:
                buffer = dx

            # Interpolate fluxes and surface levels to same time
            if var in ["tauu", "tauv", "hfss", "ps", "2t", "2sh"]:
                data.append(
                    ds_sfc[var].sel(
                        time=time.sortby("time").sel(
                            time=slice(input_json.start, input_json.end)
                        ),
                        x=slice(int(x_sw - buffer), int(x_sw + grid.xsize + buffer)),
                        y=slice(
                            int(y_sw - buffer), int(y_sw + grid.ysize + buffer)
                        ),  # TODO INT
                    )
                )
            else:
                data.append(
                    ds_ml[var].sel(
                        time=time.sortby("time").sel(
                            time=slice(input_json.start, input_json.end)
                        ),
                        x=slice(int(x_sw - buffer), int(x_sw + grid.xsize + buffer)),
                        y=slice(
                            int(y_sw - buffer), int(y_sw + grid.ysize + buffer)
                        ),  # TODO INT
                    )
                )
    # Merge into xarray dataset
    logger.debug("Succesfully read in vars, merging now... also saving to netcdf")
    if preprocess:
        data_merged = xr.merge(data, compat="override", join="outer")
        # data.to_netcdf("/ec/res4/scratch/nld4411/dales_nest_harmonie/netcdfs_newnew4/data.nc",engine="netcdf4")
        ds_ml.close()
        ds_sfc.close()
        # del data
    else:
        logger.debug("Reading in saved dataset")
        data_merged = xr.open_dataset(
            "/ec/res4/scratch/nld4411/dales_nest_harmonie/netcdfs_newnew4/data.nc",
            engine="netcdf4",
            chunks={"time": input_json.tchunk, "lev": -1},
        )
        ds_ml.close()
        ds_sfc.close()
        logger.debug("Read in saved dataset")
    return data_merged, transform, x_sw, y_sw


@logwrap
def get_transform_time(
    var: str, ds: xr.Dataset
) -> tuple[Transform, None, None, xr.DataArray]:
    # sometimes dask doesn't recognise that rioxarray has been imported. We import rioxarray to make sure we can acess rio.crs..

    # Read transform information and transform lat/lon of southwest corner to harmonie x/y
    proj = ds.rio.crs.to_proj4()
    transform = Transform({"proj4": proj})
    # x_sw, y_sw = transform.latlon_to_xy(lat_sw, lon_sw)
    # Round to 5 meters to avoid numerical error in coordinates
    # x_sw = np.round(x_sw, 0)
    # y_sw = np.round(y_sw, 0)
    time = ds["time"]
    return transform, None, None, time
