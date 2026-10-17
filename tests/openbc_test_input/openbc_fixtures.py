import pathlib
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


@pytest.fixture
def atmo_netcdf_file(tmp_path: Path):
    """Create an atmospheric NetCDF file matching the atmo.txt schema."""

    def _atmo_netcdf_file(xlen, ylen, levlen, timesteps) -> Path:
        """Create an atmospheric NetCDF file matching the atmo.txt schema."""

        path = pathlib.Path(tmp_path) / "atmo_new.nc"
        with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
            # Dimensions (sizes taken from atmo.txt)
            ds.createDimension("time", None)  # unlimited, we'll write 2 records
            ds.createDimension("x", xlen)
            ds.createDimension("y", ylen)
            ds.createDimension("lev", levlen)
            ds.createDimension("nhyi", 91)
            ds.createDimension("nhym", 90)

            # Coordinate variables
            time = ds.createVariable("time", "f8", ("time",), compression="zlib")
            time.standard_name = "time"
            time.units = "seconds since 2023-8-20 00:00:00"
            time.calendar = "proleptic_gregorian"
            time.axis = "T"
            time[:] = np.arange(timesteps, dtype="f8")

            x = ds.createVariable("x", "f8", ("x",), compression="zlib")
            x.standard_name = "projection_x_coordinate"
            x.units = "m"
            x.axis = "X"
            x[:] = np.arange(xlen, dtype="f8")

            y = ds.createVariable("y", "f8", ("y",), compression="zlib")
            y.standard_name = "projection_y_coordinate"
            y.units = "m"
            y.axis = "Y"
            y[:] = np.arange(ylen, dtype="f8")

            # Grid mapping variable
            lambert = ds.createVariable("Lambert_Conformal", "i4", compression="zlib")
            lambert.grid_mapping_name = "lambert_conformal_conic"
            lambert.standard_parallel = 48.85
            lambert.longitude_of_central_meridian = 2.35
            lambert.latitude_of_projection_origin = 48.85
            lambert.earth_radius = 6371229.0
            lambert.longitudeOfFirstGridPointInDegrees = 359.119746
            lambert.latitudeOfFirstGridPointInDegrees = 46.582186
            lambert.longitudeOfSouthernPoleInDegrees = 0.0
            lambert.latitudeOfSouthernPoleInDegrees = -90.0
            lambert[...] = 1

            lev = ds.createVariable("lev", "f8", ("lev",), compression="zlib")
            lev.standard_name = "hybrid_sigma_pressure"
            lev.long_name = "hybrid level at layer midpoints"
            lev.formula = "hyam hybm (mlev=hyam+hybm*aps)"
            lev.formula_terms = "ap: hyam b: hybm ps: aps"
            lev.units = "level"
            lev.positive = "down"
            lev[:] = np.arange(levlen, dtype="f8")[::-1]  # 20, 19, ..., 0

            hyai = ds.createVariable("hyai", "f8", ("nhyi",), compression="zlib")
            hyai.long_name = "hybrid A coefficient at layer interfaces"
            hyai.units = "Pa"
            hyai[:] = np.linspace(0.0, 1.0, 91, dtype="f8")

            hybi = ds.createVariable("hybi", "f8", ("nhyi",), compression="zlib")
            hybi.long_name = "hybrid B coefficient at layer interfaces"
            hybi.units = "1"
            hybi[:] = np.linspace(0.0, 1.0, 91, dtype="f8")

            hyam = ds.createVariable("hyam", "f8", ("nhym",), compression="zlib")
            hyam.long_name = "hybrid A coefficient at layer midpoints"
            hyam.units = "Pa"
            hyam[:] = np.linspace(0.0, 1.0, 90, dtype="f8")

            hybm = ds.createVariable("hybm", "f8", ("nhym",), compression="zlib")
            hybm.long_name = "hybrid B coefficient at layer midpoints"
            hybm.units = "1"
            hybm[:] = np.linspace(0.0, 1.0, 90, dtype="f8")

            crwc = ds.createVariable(
                "crwc", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            crwc.long_name = "Specific rain water content"
            crwc.units = "kg kg**-1"
            crwc.param = "85.1.0"
            crwc.grid_mapping = "Lambert_Conformal"
            crwc[:, :, :, :] = 0

            cswc = ds.createVariable(
                "cswc", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            cswc.long_name = "Specific snow water content"
            cswc.units = "kg kg**-1"
            cswc.param = "86.1.0"
            cswc.grid_mapping = "Lambert_Conformal"
            cswc[:, :, :, :] = 0

            z = ds.createVariable(
                "z", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            z.standard_name = "geopotential"
            z.long_name = "Geopotential"
            z.units = "m**2 s**-2"
            z.param = "4.3.0"
            z.grid_mapping = "Lambert_Conformal"
            z[:, :, :, :] = np.arange(len(lev), dtype="f4")[::-1].reshape(
                1, len(lev), 1, 1
            )

            t = ds.createVariable(
                "t", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            t.standard_name = "air_temperature"
            t.long_name = "Temperature"
            t.units = "K"
            t.param = "0.0.0"
            t.grid_mapping = "Lambert_Conformal"
            t[:, :, :, :] = 273.15

            u = ds.createVariable(
                "u", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            u.standard_name = "eastward_wind"
            u.long_name = "U component of wind"
            u.units = "m s**-1"
            u.param = "2.2.0"
            u.grid_mapping = "Lambert_Conformal"
            u[:, :, :, :] = 0.5

            v = ds.createVariable(
                "v", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            v.standard_name = "northward_wind"
            v.long_name = "V component of wind"
            v.units = "m s**-1"
            v.param = "3.2.0"
            v.grid_mapping = "Lambert_Conformal"
            v[:, :, :, :] = 0.5

            q = ds.createVariable(
                "q", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            q.standard_name = "specific_humidity"
            q.long_name = "Specific humidity"
            q.units = "kg kg**-1"
            q.param = "0.1.0"
            q.grid_mapping = "Lambert_Conformal"
            q[:, :, :, :] = 1e-3

            clwc = ds.createVariable(
                "clwc", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            clwc.long_name = "Specific cloud liquid water content"
            clwc.units = "kg kg**-1"
            clwc.param = "83.1.0"
            clwc.grid_mapping = "Lambert_Conformal"
            clwc[:, :, :, :] = 0

            ciwc = ds.createVariable(
                "ciwc", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            ciwc.long_name = "Specific cloud ice water content"
            ciwc.units = "kg kg**-1"
            ciwc.param = "84.1.0"
            ciwc.grid_mapping = "Lambert_Conformal"
            ciwc[:, :, :, :] = 0

            cc = ds.createVariable(
                "cc", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            cc.long_name = "Fraction of cloud cover"
            cc.units = "(0 - 1)"
            cc.param = "32.6.0"
            cc.grid_mapping = "Lambert_Conformal"
            cc[:, :, :, :] = 0

            grle = ds.createVariable(
                "grle", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            grle.long_name = "Graupel (snow pellets)"
            grle.units = "kg kg**-1"
            grle.param = "32.1.0"
            grle.grid_mapping = "Lambert_Conformal"
            grle[:, :, :, :] = 0

            tke = ds.createVariable(
                "tke", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            tke.long_name = "Turbulent kinetic energy"
            tke.units = "J kg**-1"
            tke.param = "11.19.0"
            tke.grid_mapping = "Lambert_Conformal"
            tke[:, :, :, :] = 0

            wz = ds.createVariable(
                "wz", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            wz.long_name = "Geometric vertical velocity"
            wz.units = "m s**-1"
            wz.param = "9.2.0"
            wz.grid_mapping = "Lambert_Conformal"
            wz[:, :, :, :] = 0
            # Global attributes (minimal subset)
            ds.CDI = "Climate Data Interface"
            ds.Conventions = "CF-1.6"
            ds.CDO = "Climate Data Operators"

        return path

    return _atmo_netcdf_file


@pytest.fixture
def surface_netcdf_file(tmp_path: Path):
    """Create a surface NetCDF file matching the surface.txt schema."""

    def _surface_netcdf_file(xlen, ylen, levlen, timesteps) -> Path:
        """Create a surface NetCDF file matching the surface.txt schema."""
        path = pathlib.Path(tmp_path) / "surface_new.nc"
        with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
            # Dimensions (sizes taken from surface.txt)
            ds.createDimension("time", None)  # unlimited, we'll write 3 records
            ds.createDimension("x", xlen)
            ds.createDimension("y", ylen)
            ds.createDimension("height", 1)
            ds.createDimension("height_2", 1)
            ds.createDimension("plev", 1)
            ds.createDimension("bnds", 2)
            ds.createDimension("plev_2", 1)
            ds.createDimension("lev", 1)
            ds.createDimension("lev_2", 1)
            ds.createDimension("lev_3", 1)
            ds.createDimension("lev_4", 1)
            ds.createDimension("lev_5", 1)
            ds.createDimension("lev_6", 1)
            ds.createDimension("lev_7", 1)
            ds.createDimension("lev_8", 1)

            # Coordinate variables
            time = ds.createVariable("time", "f8", ("time",), compression="zlib")
            time.standard_name = "time"
            time.units = "seconds since 2023-8-20 00:00:00"
            time.calendar = "proleptic_gregorian"
            time.axis = "T"
            time[:] = np.arange(timesteps, dtype="f8")

            x = ds.createVariable("x", "f8", ("x",), compression="zlib")
            x.standard_name = "projection_x_coordinate"
            x.units = "m"
            x.axis = "X"
            x[:] = np.arange(xlen, dtype="f8")

            y = ds.createVariable("y", "f8", ("y",), compression="zlib")
            y.standard_name = "projection_y_coordinate"
            y.units = "m"
            y.axis = "Y"
            y[:] = np.arange(ylen, dtype="f8")

            lambert = ds.createVariable("Lambert_Conformal", "i4", compression="zlib")
            lambert.grid_mapping_name = "lambert_conformal_conic"
            lambert.standard_parallel = 48.85
            lambert.longitude_of_central_meridian = 2.35
            lambert.latitude_of_projection_origin = 48.85
            lambert.earth_radius = 6371229.0
            lambert.longitudeOfFirstGridPointInDegrees = 359.119746
            lambert.latitudeOfFirstGridPointInDegrees = 46.582186
            lambert.longitudeOfSouthernPoleInDegrees = 0.0
            lambert.latitudeOfSouthernPoleInDegrees = -90.0
            lambert[...] = 1

            height = ds.createVariable("height", "f8", ("height",), compression="zlib")
            height.standard_name = "height"
            height.long_name = "height"
            height.units = "m"
            height.positive = "up"
            height.axis = "Z"
            height[:] = np.array([10.0])

            height_2 = ds.createVariable(
                "height_2", "f8", ("height_2",), compression="zlib"
            )
            height_2.standard_name = "height"
            height_2.long_name = "height"
            height_2.units = "m"
            height_2.positive = "up"
            height_2.axis = "Z"
            height_2[:] = np.array([2.0])

            plev = ds.createVariable("plev", "f8", ("plev",), compression="zlib")
            plev.standard_name = "air_pressure"
            plev.long_name = "pressure"
            plev.units = "Pa"
            plev.positive = "down"
            plev.axis = "Z"
            plev.bounds = "plev_bnds"
            plev[:] = np.array([100000.0])

            plev_bnds = ds.createVariable(
                "plev_bnds", "f8", ("plev", "bnds"), compression="zlib"
            )
            plev_bnds[:] = np.array([[95000.0, 105000.0]])

            plev_2 = ds.createVariable("plev_2", "f8", ("plev_2",), compression="zlib")
            plev_2.standard_name = "air_pressure"
            plev_2.long_name = "pressure"
            plev_2.units = "Pa"
            plev_2.positive = "down"
            plev_2.axis = "Z"
            plev_2.bounds = "plev_2_bnds"
            plev_2[:] = np.array([30000.0])

            plev_2_bnds = ds.createVariable(
                "plev_2_bnds", "f8", ("plev_2", "bnds"), compression="zlib"
            )
            plev_2_bnds[:] = np.array([[25000.0, 35000.0]])

            lev = ds.createVariable("lev", "f8", ("lev",), compression="zlib")
            lev.axis = "Z"
            lev[:] = np.array([0.0])

            lev_2 = ds.createVariable("lev_2", "f8", ("lev_2",), compression="zlib")
            lev_2.axis = "Z"
            lev_2[:] = np.array([0.0])

            lev_3 = ds.createVariable("lev_3", "f8", ("lev_3",), compression="zlib")
            lev_3.axis = "Z"
            lev_3[:] = np.array([0.0])

            lev_4 = ds.createVariable("lev_4", "f8", ("lev_4",), compression="zlib")
            lev_4.axis = "Z"
            lev_4[:] = np.array([0.0])

            lev_5 = ds.createVariable("lev_5", "f8", ("lev_5",), compression="zlib")
            lev_5.axis = "Z"
            lev_5[:] = np.array([0.0])

            lev_6 = ds.createVariable("lev_6", "f8", ("lev_6",), compression="zlib")
            lev_6.axis = "Z"
            lev_6[:] = np.array([0.0])

            lev_7 = ds.createVariable("lev_7", "f8", ("lev_7",), compression="zlib")
            lev_7.axis = "Z"
            lev_7[:] = np.array([0.0])

            lev_8 = ds.createVariable("lev_8", "f8", ("lev_8",), compression="zlib")
            lev_8.long_name = "isentropic"
            lev_8.units = "K"
            lev_8.axis = "Z"
            lev_8[:] = np.array([300.0])

            z = ds.createVariable("z", "f8", ("y", "x"), compression="zlib")
            z.standard_name = "geopotential"
            z.long_name = "Geopotential"
            z.units = "m**2 s**-2"
            z.param = "4.3.0"
            z.grid_mapping = "Lambert_Conformal"
            z[:, :] = 0

            t = ds.createVariable("t", "f4", ("time", "y", "x"), compression="zlib")
            t.standard_name = "air_temperature"
            t.long_name = "Temperature"
            t.units = "K"
            t.param = "0.0.0"
            t.grid_mapping = "Lambert_Conformal"
            t[:, :, :] = 273.15

            sp = ds.createVariable("sp", "f4", ("time", "y", "x"), compression="zlib")
            sp.standard_name = "surface_air_pressure"
            sp.long_name = "Surface pressure"
            sp.units = "Pa"
            sp.param = "0.3.0"
            sp.grid_mapping = "Lambert_Conformal"
            sp[
                :,
                :,
            ] = 100000.0

            tcwv = ds.createVariable("tcwv", "f4", ("y", "x"), compression="zlib")
            tcwv.standard_name = (
                "lwe_thickness_of_atmosphere_mass_content_of_water_vapor"
            )
            tcwv.long_name = "Total column vertically-integrated water vapour"
            tcwv.units = "kg m**-2"
            tcwv.param = "64.1.0"
            tcwv.grid_mapping = "Lambert_Conformal"
            tcwv[:, :] = 0

            msl = ds.createVariable("msl", "f4", ("time", "y", "x"), compression="zlib")
            msl.standard_name = "air_pressure_at_mean_sea_level"
            msl.long_name = "Mean sea level pressure"
            msl.units = "Pa"
            msl.param = "0.3.0"
            msl.grid_mapping = "Lambert_Conformal"
            msl[:, :, :] = 100000.0

            blh = ds.createVariable("blh", "f4", ("time", "y", "x"), compression="zlib")
            blh.long_name = "Boundary layer height"
            blh.units = "m"
            blh.param = "18.3.0"
            blh.grid_mapping = "Lambert_Conformal"
            blh[:, :, :] = 0

            u10 = ds.createVariable(
                "10u", "f4", ("time", "height", "y", "x"), compression="zlib"
            )
            u10.standard_name = "eastward_wind"
            u10.long_name = "10 metre U wind component"
            u10.units = "m s**-1"
            u10.param = "2.2.0"
            u10.grid_mapping = "Lambert_Conformal"
            u10[:, :, :, :] = 0.5

            v10 = ds.createVariable(
                "10v", "f4", ("time", "height", "y", "x"), compression="zlib"
            )
            v10.standard_name = "northward_wind"
            v10.long_name = "10 metre V wind component"
            v10.units = "m s**-1"
            v10.param = "3.2.0"
            v10.grid_mapping = "Lambert_Conformal"
            v10[:, :, :, :] = 0.5

            t2 = ds.createVariable(
                "2t", "f4", ("time", "height_2", "y", "x"), compression="zlib"
            )
            t2.standard_name = "air_temperature"
            t2.long_name = "2 metre temperature"
            t2.units = "K"
            t2.param = "0.0.0"
            t2.grid_mapping = "Lambert_Conformal"
            t2[:, :, :, :] = 273.15

            lsm = ds.createVariable("lsm", "f4", ("y", "x"), compression="zlib")
            lsm.standard_name = "land_binary_mask"
            lsm.long_name = "Land-sea mask"
            lsm.units = "(0 - 1)"
            lsm.param = "0.0.2"
            lsm.grid_mapping = "Lambert_Conformal"
            lsm[:, :] = 1

            h = ds.createVariable("h", "f8", ("y", "x"), compression="zlib")
            h.long_name = "Geometrical height above ground"
            h.units = "m"
            h.param = "6.3.0"
            h.grid_mapping = "Lambert_Conformal"
            h[:, :] = 0

            lcc = ds.createVariable("lcc", "f4", ("time", "y", "x"), compression="zlib")
            lcc.long_name = "Low cloud cover"
            lcc.units = "%"
            lcc.param = "3.6.0"
            lcc.grid_mapping = "Lambert_Conformal"
            lcc[:, :, :] = 0

            mcc = ds.createVariable(
                "mcc", "f4", ("time", "plev", "y", "x"), compression="zlib"
            )
            mcc.long_name = "Medium cloud cover"
            mcc.units = "%"
            mcc.param = "4.6.0"
            mcc.grid_mapping = "Lambert_Conformal"
            mcc[:, :, :, :] = 0

            hcc = ds.createVariable(
                "hcc", "f4", ("time", "plev_2", "y", "x"), compression="zlib"
            )
            hcc.long_name = "High cloud cover"
            hcc.units = "%"
            hcc.param = "5.6.0"
            hcc.grid_mapping = "Lambert_Conformal"
            hcc[:, :, :, :] = 0

            q2 = ds.createVariable(
                "2sh", "f4", ("time", "height_2", "y", "x"), compression="zlib"
            )
            q2.standard_name = "specific_humidity"
            q2.long_name = "2 metre specific humidity"
            q2.units = "kg kg**-1"
            q2.param = "0.1.0"
            q2.grid_mapping = "Lambert_Conformal"
            q2[:, :, :, :] = 1e-3

            cbh = ds.createVariable("cbh", "f4", ("time", "y", "x"), compression="zlib")
            cbh.long_name = "Cloud base height"
            cbh.units = "m"
            cbh.param = "11.6.0"
            cbh.grid_mapping = "Lambert_Conformal"
            cbh.level_type = "cloudbase"
            cbh[:, :, :] = 0

            deg0l = ds.createVariable(
                "deg0l", "f4", ("time", "lev", "y", "x"), compression="zlib"
            )
            deg0l.long_name = "Geometric height of 0 degrees C atmospheric isothermal level above ground"
            deg0l.units = "m"
            deg0l.param = "34.3.0"
            deg0l.grid_mapping = "Lambert_Conformal"
            deg0l[:, :, :, :] = 0

            sd = ds.createVariable("sd", "f4", ("time", "y", "x"), compression="zlib")
            sd.long_name = "Snow depth water equivalent"
            sd.units = "kg m**-2"
            sd.param = "60.1.0"
            sd.grid_mapping = "Lambert_Conformal"
            sd[:, :, :] = 0

            tcc = ds.createVariable("tcc", "f4", ("time", "y", "x"), compression="zlib")
            tcc.long_name = "Total Cloud Cover"
            tcc.units = "%"
            tcc.param = "1.6.0"
            tcc.grid_mapping = "Lambert_Conformal"
            tcc[:, :, :] = 0

            mucape = ds.createVariable(
                "mucape", "f4", ("time", "lev_2", "y", "x"), compression="zlib"
            )
            mucape.long_name = "Most-unstable CAPE"
            mucape.units = "J kg**-1"
            mucape.param = "6.7.0"
            mucape.grid_mapping = "Lambert_Conformal"
            mucape[:, :, :, :] = 0

            mucin = ds.createVariable(
                "mucin", "f4", ("time", "lev_2", "y", "x"), compression="zlib"
            )
            mucin.long_name = "Most-unstable CIN"
            mucin.units = "J kg**-1"
            mucin.param = "7.7.0"
            mucin.grid_mapping = "Lambert_Conformal"
            mucin[:, :, :, :] = 0

            pcdb = ds.createVariable(
                "pcdb", "f4", ("time", "y", "x"), compression="zlib"
            )
            pcdb.long_name = "Pressure at cloud base"
            pcdb.units = "Pa"
            pcdb.param = "0.3.0"
            pcdb.grid_mapping = "Lambert_Conformal"
            pcdb.level_type = "cloudbase"
            pcdb[:, :, :] = 100000.0

            hacg = ds.createVariable(
                "hacg", "f4", ("time", "lev_3", "y", "x"), compression="zlib"
            )
            hacg.long_name = (
                "Geometric height of adiabatic condensation level above ground"
            )
            hacg.units = "m"
            hacg.param = "34.3.0"
            hacg.grid_mapping = "Lambert_Conformal"
            hacg[:, :, :, :] = 0

            hfcg = ds.createVariable(
                "hfcg", "f4", ("time", "lev_4", "y", "x"), compression="zlib"
            )
            hfcg.long_name = "Geometric height of free convection level above ground"
            hfcg.units = "m"
            hfcg.param = "34.3.0"
            hfcg.grid_mapping = "Lambert_Conformal"
            hfcg[:, :, :, :] = 0

            hnbg = ds.createVariable(
                "hnbg", "f4", ("time", "lev_5", "y", "x"), compression="zlib"
            )
            hnbg.long_name = "Geometric height of neutral buoyancy level above ground"
            hnbg.units = "m"
            hnbg.param = "34.3.0"
            hnbg.grid_mapping = "Lambert_Conformal"
            hnbg[:, :, :, :] = 0

            deg3l = ds.createVariable(
                "deg3l", "f4", ("time", "lev_6", "y", "x"), compression="zlib"
            )
            deg3l.long_name = "Geometric height of 3 degrees C atmospheric isothermal level above ground"
            deg3l.units = "m"
            deg3l.param = "34.3.0"
            deg3l.grid_mapping = "Lambert_Conformal"
            deg3l[:, :, :, :] = 0

            pcdc = ds.createVariable(
                "pcdc", "f4", ("time", "lev_7", "y", "x"), compression="zlib"
            )
            pcdc.long_name = "Pressure at cloud ceiling"
            pcdc.units = "Pa"
            pcdc.param = "0.3.0"
            pcdc.grid_mapping = "Lambert_Conformal"
            pcdc[:, :, :, :] = 0

            h1thg = ds.createVariable(
                "h1thg", "f4", ("time", "lev_8", "y", "x"), compression="zlib"
            )
            h1thg.long_name = "Geometric height of 1 degree C theta level above ground"
            h1thg.units = "m"
            h1thg.param = "34.3.0"
            h1thg.grid_mapping = "Lambert_Conformal"
            h1thg[:, :, :, :] = 0

            tisemf = ds.createVariable("tisemf", "f4", ("y", "x"), compression="zlib")
            tisemf.long_name = "Time integral of surface eastward momentum flux"
            tisemf.units = "N m**-2 s"
            tisemf.param = "17.2.0"
            tisemf.grid_mapping = "Lambert_Conformal"
            tisemf[:, :] = 0

            tisnmf = ds.createVariable("tisnmf", "f4", ("y", "x"), compression="zlib")
            tisnmf.long_name = "Time integral of surface northward momentum flux"
            tisnmf.units = "N m**-2 s"
            tisnmf.param = "18.2.0"
            tisnmf.grid_mapping = "Lambert_Conformal"
            tisnmf[:, :] = 0

            rprate = ds.createVariable(
                "rprate", "f4", ("time", "y", "x"), compression="zlib"
            )
            rprate.long_name = "Rain precipitation rate"
            rprate.units = "kg m**-2 s**-1"
            rprate.param = "65.1.0"
            rprate.grid_mapping = "Lambert_Conformal"
            rprate[:, :, :] = 0

            ceil = ds.createVariable(
                "ceil", "f4", ("time", "y", "x"), compression="zlib"
            )
            ceil.long_name = "Ceiling"
            ceil.units = "m"
            ceil.param = "13.6.0"
            ceil.grid_mapping = "Lambert_Conformal"
            ceil[:, :, :] = 0

            r2 = ds.createVariable(
                "2r", "f4", ("time", "height_2", "y", "x"), compression="zlib"
            )
            r2.standard_name = "relative_humidity"
            r2.long_name = "2 metre relative humidity"
            r2.units = "%"
            r2.param = "1.1.0"
            r2.grid_mapping = "Lambert_Conformal"
            r2[:, :, :, :] = 0

            mrt = ds.createVariable(
                "mrt", "f4", ("time", "height_2", "y", "x"), compression="zlib"
            )
            mrt.long_name = "Mean radiant temperature"
            mrt.units = "K"
            mrt.param = "1.0.20"
            mrt.grid_mapping = "Lambert_Conformal"
            mrt[:, :, :, :] = 273.15

            # Global attributes (minimal subset)
            ds.CDI = "Climate Data Interface"
            ds.Conventions = "CF-1.6"
            ds.CDO = "Climate Data Operators"

        return path

    return _surface_netcdf_file
