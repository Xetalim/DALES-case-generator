# Crops the HARMONIE data to the required time and spatial extends.
# Transforms pressure coordinates into height levels.
# Transforms HARMONIE prognostic variables to DALES prognostic variables
import logging

import dask
import numpy as np
import xarray as xr
from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.IO_helpers.raster import fix_lambert_offsets
from modular_dales.LBC.nest_dales_in_HARMONIE import hybrid_levels
from modular_dales.LBC.nest_dales_in_HARMONIE.helper import (
    calcBaseprof,
    differentiate,
)
from modular_dales.LBC.nest_dales_in_HARMONIE.Transform import Transform
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


xr.set_options(use_new_combine_kwarg_defaults=True)
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
        self.ps = None
        self.thls = None
        self.data = None
        self.transform = None
        self.p = None
        self.z = None
        self.z_int = None
        self.exnrs = None
        self.exnr = None
        self.thl = None
        self.backrad_profile = None

    def load_data(self):
        variables = ["ua", "va", "wa", "ta", "hus", "clw", "ps", "tas", "huss"]
        if self.input_json.synturb is not None:
            variables = variables + ["tke", "tauu", "tauv", "cb", "hfss"]
        self.data, self.transform, x_sw, y_sw = create_xarray_dataset(
            self.input_json, self.grid, variables
        )

        (self.data,) = dask.optimize(self.data)
        # Change transform parameters to new DALES origin and update transform
        self.transform = update_transform(self.transform, x_sw, y_sw)

    @logwrap
    def prep_harmonie(self):

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

        self.z = calculate_3d_height_levels(self.input_json, self.data).persist()
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

        # Calculate base profiles and exnr function
        self.ps, self.exnrs, self.thls, self.exnr = calc_base_exner(
            self.input_json, self.grid, self.data, self.z_int
        )
        logger.debug("Getting surface pressure and temperature")
        self.ps = float(self.ps)  # save the ps value used for the profile
        self.thls = float(self.thls)  # and thls
        logger.debug("Creating exnr dataarray")
        self.exnr = xr.DataArray(
            np.concatenate([[self.exnrs], self.exnr]),
            dims=["z"],
            coords={"z": self.z_int},
            name="exnr",
            attrs={"thls": float(self.thls), "ps": float(self.ps)},
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
                self.ps,
                self.exnrs,
                self.thls,
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
        (self.data,) = dask.optimize(self.data)

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
def calculate_pressure(data):
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
    return p.sel(lev=data["lev"]).chunk(data["u"].chunks)


@logwrap
def merge_steps(data, variables):
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
def get_ref_height_crop(input_json, grid: GridDalesOpenBC, data):
    if input_json.start == input_json.time0:  # Define reference height levels
        z_int = (
            data["z3d"]
            .isel({"time": 0}, drop=True)
            .sel(x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1]))
            .mean(dim=["x", "y"])
        )
    else:  # Take reference height levels from exnr.inp.xxx
        try:
            exnr = np.loadtxt(input_json.exnr_file, skiprows=1)
        except FileNotFoundError as e:
            logger.critical(
                "No reference height levels found in exnr_file in configuration. This is required as the simulation start is not the same as the HARMONIE start."
            )
            raise e
        z_int = exnr[:, 0]
    return z_int


@logwrap
def calculate_turbulence_vars(
    grid: GridDalesOpenBC,
    data,
    turbulence_data_dic,
    z_int,
    ps_exnr,
    exnrs,
    thls_exnr,
    thl,
):

    # Calculate inversion height from maximum curvature and with cloud base as a backup
    zi_min = 200
    zi_max = 4000
    thlmean = thl.sel(
        x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1])
    ).mean(dim=["x", "y"])
    cbmean = (
        xr.where(turbulence_data_dic["cb"] > 0.0, turbulence_data_dic["cb"], np.NaN)
        .sel(x=slice(grid.xt[0], grid.xt[-1]), y=slice(grid.yt[0], grid.yt[-1]))
        .mean(dim=["x", "y"])
    )
    its = 0
    d2thl = []
    d1thl = []
    rhobf = calcBaseprof(z_int, thls_exnr, ps_exnr, pref0=p0)
    for tchunk in thlmean.chunks[0]:
        ite = its + tchunk
        d1thl.append(
            dask.delayed(differentiate)(
                thlmean.isel({"time": np.arange(its, ite)}), "z", 1, acc=6
            )
        )
        d2thl.append(
            dask.delayed(differentiate)(
                thlmean.isel({"time": np.arange(its, ite)}), "z", 2, acc=6
            )
        )
        its = ite
    d1thl = dask.delayed(xr.concat)(d2thl, dim="time").compute()
    d2thl = dask.delayed(xr.concat)(d1thl, dim="time").compute()
    zi = d2thl.where(d1thl > 0).sel(z=slice(zi_min, zi_max)).idxmax("z").fillna(cbmean)
    rhobs = rhobf[0] - z_int[0] * (rhobf[1] - rhobf[0]) / (z_int[1] - z_int[0])
    ustar = np.sqrt(np.maximum(turbulence_data_dic["tauu"], 0) / rhobs).rename("ustar")
    vstar = np.sqrt(np.maximum(turbulence_data_dic["tauv"], 0) / rhobs).rename("vstar")
    wthls = (turbulence_data_dic["hfss"] / (exnrs * rhobs * cp)).rename("wthls")
    data = data.assign({"ustar": ustar, "vstar": vstar, "wthls": wthls, "zi": zi})


@logwrap
def calc_base_exner(input_json, grid: GridDalesOpenBC, data, z_int):
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
        rhobf = calcBaseprof(z_int, thls_exnr, ps_exnr, pref0=p0)
        p_exnr = (
            rhobf[1:]
            * Rd
            * air_values["t"]
            * (1 + (Rv / Rd - 1) * air_values["qt"] - Rv / Rd * air_values["clwc"])
        )  # Ideal gas law
        exnr = (p_exnr / p0) ** (Rd / cp)
    else:  # Read exnr.inp.xxx
        try:
            with open(input_json.exnr_file, "r") as file:
                line0 = file.readline()
        except FileNotFoundError as e:
            logger.critical(
                "No reference height levels found in exnr_file in configuration. This is required as the simulation start is not the same as the HARMONIE start."
            )
            raise e
        thls_exnr = float(line0.split(",")[1].split("thls = ")[-1])
        ps_exnr = float(line0.split(",")[2].split("ps = ")[-1])
        exnr = np.loadtxt(input_json.exnr_file, skiprows=1)
        exnrs = exnr[0, 1]
        exnr = exnr[1:, 1]
    return ps_exnr, exnrs, thls_exnr, exnr


@logwrap
def vertical_interp_all(ds, z3d, z_new):
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

    def _interp_func(var_col, z_col, z_target=z_target):
        # var_col, z_col: 1D arrays of shape (lev,)
        return np.interp(z_target, z_col, var_col)

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
def interpolate_ref_height(input_json, data, z_int):
    data_intz = []
    its = 0
    new_shape = [data.sizes["time"], len(z_int), data.sizes["y"], data.sizes["x"]]
    new_coords = {
        "time": data.coords["time"],
        "z": z_int,
        "y": data.coords["y"],
        "x": data.coords["x"],
    }
    new_dims = ["time", "z", "y", "x"]
    variables = ["ua", "va", "wa", "ta", "p", "hus", "clw"]
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
def calculate_3d_height_levels(input_json, data):
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
        .chunk(data["u"].chunks)
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
def update_transform(transform, x_sw, y_sw):
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
def create_xarray_dataset(input_json, grid: GridDalesOpenBC, variables):
    preprocess = True

    data = []
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
            chunks={"time": input_json.tchunk, "lev": -1},
            # chunks={"x": "auto", "y": "auto", "time": "auto", "lev": -1},
        )
    )
    transform, _, _, time = get_transform_time(input_json, var, ds_ml)

    ds_sfc = fix_lambert_offsets(
        xr.open_mfdataset(
            input_json.HARMONIE_sfc_glob,
            decode_coords="all",
            # parallel=False,
            engine="netcdf4",
            chunks={"time": input_json.tchunk},
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
        data = xr.merge(data, compat="override", join="outer")
        # data.to_netcdf("/ec/res4/scratch/nld4411/dales_nest_harmonie/netcdfs_newnew4/data.nc",engine="netcdf4")
        ds_ml.close()
        ds_sfc.close()
        # del data
    else:
        logger.debug("Reading in saved dataset")
        data = xr.open_dataset(
            "/ec/res4/scratch/nld4411/dales_nest_harmonie/netcdfs_newnew4/data.nc",
            engine="netcdf4",
            chunks={"time": input_json.tchunk, "lev": -1},
        )
        ds_ml.close()
        ds_sfc.close()
        logger.debug("Read in saved dataset")
    return data, transform, x_sw, y_sw


@logwrap
def get_transform_time(input_json, var, ds):
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
