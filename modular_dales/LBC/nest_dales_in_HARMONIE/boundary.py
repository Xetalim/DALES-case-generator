# Interpolate fields to DALES domain boundary
# Creates openboundaries.inp.xxx.nc
import logging
from datetime import datetime

import numpy as np
import xarray as xr
from modular_dales.Geometry.GridDales import GridDalesOpenBC
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@logwrap
def boundary_fields(
    input_json: OpenBoundaryConfig, grid: GridDalesOpenBC, data, output_path
):
    # data = data.drop(["lat", "lon"])
    # West boundary
    openboundaries = get_boundaries(input_json, grid, data)
    # Adjust time variable to seconds since initial field
    openboundaries = set_time_attrs(input_json, openboundaries)
    # Add variable attributes
    set_variable_attributes(input_json, openboundaries)
    # Add global attributes
    openboundaries = openboundaries.assign_attrs(
        {
            # "title": "openboundaries.inp.XXX.nc",
            "history": f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "author": input_json.author,
            "time0": input_json.time0,
        }
    )

    return openboundaries


@logwrap
def set_variable_attributes(input_json: OpenBoundaryConfig, openboundaries):
    openboundaries["time"] = openboundaries["time"].assign_attrs(
        {"longname": "Time", "units": f"seconds since {input_json.time0}"}
    )
    openboundaries["xt"] = openboundaries["xt"].assign_attrs(
        {"longname": "West-East displacement of cell centers", "units": "m"}
    )
    openboundaries["xm"] = openboundaries["xm"].assign_attrs(
        {"longname": "West-East displacement of cell edges", "units": "m"}
    )
    openboundaries["yt"] = openboundaries["yt"].assign_attrs(
        {"longname": "South-North displacement of cell centers", "units": "m"}
    )
    openboundaries["ym"] = openboundaries["ym"].assign_attrs(
        {"longname": "South-North displacement of cell edges", "units": "m"}
    )
    openboundaries["zt"] = openboundaries["zt"].assign_attrs(
        {"longname": "Vertical displacement of cell centers", "units": "m"}
    )
    openboundaries["zm"] = openboundaries["zm"].assign_attrs(
        {"longname": "Vertical displacement of cell edges", "units": "m"}
    )
    variables = ["u", "v", "w", "thl", "qt", "e12"]
    units = ["m/s", "m/s", "m/s", "K", "kg/kg", "m/s"]
    long_names = [
        "West-East velocity at ",
        "South-North velocity at ",
        "Vertical velocity at ",
        "Liquid water potential temperature at ",
        "Total water specific humidity at ",
        "Square root of turbulent kinetic energy at ",
    ]
    for ivar, var in enumerate(variables):
        unit = units[ivar]
        long_name = long_names[ivar]
        for boundary in ["West", "East", "South", "North", "top"]:
            openboundaries[var + boundary.lower()] = openboundaries[
                var + boundary.lower()
            ].assign_attrs(
                {"longname": long_name + boundary + " boundary", "units": unit}
            )


@logwrap
def set_time_attrs(input_json: OpenBoundaryConfig, openboundaries):
    ts = openboundaries["time"].values.astype("datetime64[s]")
    dts = (ts - np.datetime64(input_json.time0, "s")) / np.timedelta64(1, "s")
    openboundaries = openboundaries.assign_coords({"time": ("time", dts)})
    openboundaries["time"].attrs.clear()
    return openboundaries


@logwrap
def get_boundaries(input_json: OpenBoundaryConfig, grid: GridDalesOpenBC, data):
    north = data.sel(
        y=slice(grid.yt[-1] - grid.dy * 16, grid.yt[-1] + grid.dy * 16),
        x=slice(grid.xt[0] - grid.dx * 16, grid.xt[-1] + grid.dx * 16),
    )
    south = data.sel(
        y=slice(grid.yt[0] - grid.dy * 16, grid.yt[0] + grid.dy * 16),
        x=slice(grid.xt[0] - grid.dx * 16, grid.xt[-1] + grid.dx * 16),
    )

    east = data.sel(
        x=slice(grid.xt[-1] - grid.dx * 16, grid.xt[-1] + grid.dx * 16),
        y=slice(grid.yt[0] - grid.dy * 16, grid.yt[-1] + grid.dy * 16),
    )
    west = data.sel(
        x=slice(grid.xt[0] - grid.dx * 16, grid.xt[0] + grid.dx * 16),
        y=slice(grid.yt[0] - grid.dy * 16, grid.yt[-1] + grid.dy * 16),
    )

    top = data.sel(
        y=slice(grid.yt[0] - grid.dy * 16, grid.yt[-1] + grid.dy * 16),
        x=slice(grid.xt[0] - grid.dx * 16, grid.xt[-1] + grid.dx * 16),
    )

    uwest = (
        west["u"]
        .interp(z=grid.zt)
        .interp(y=grid.yt, x=grid.xm[0], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "yt"})
        .rename("uwest")
        .drop(["x"])
    )
    vwest = (
        west["v"]
        .interp(z=grid.zt)
        .interp(y=grid.ym, x=grid.xm[0], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "ym"})
        .rename("vwest")
        .drop(["x"])
    )
    wwest = (
        west["w"]
        .interp(z=grid.zm)
        .interp(y=grid.yt, x=grid.xm[0], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zm", "y": "yt"})
        .rename("wwest")
        .drop(["x"])
    )
    thlwest = (
        west["thl"]
        .interp(z=grid.zt)
        .interp(y=grid.yt, x=grid.xm[0], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "yt"})
        .rename("thlwest")
        .drop(["x"])
    )
    qtwest = (
        west["qt"]
        .interp(z=grid.zt)
        .interp(y=grid.yt, x=grid.xm[0], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "yt"})
        .rename("qtwest")
        .drop(["x"])
    )
    e12west = (xr.ones_like(thlwest) * input_json.e12).rename("e12west")
    uwest.attrs.clear()
    vwest.attrs.clear()
    wwest.attrs.clear()
    thlwest.attrs.clear()
    qtwest.attrs.clear()
    # East boundary
    ueast = (
        east["u"]
        .interp(z=grid.zt)
        .interp(y=grid.yt, x=grid.xm[-1], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "yt"})
        .rename("ueast")
        .drop(["x"])
    )
    veast = (
        east["v"]
        .interp(z=grid.zt)
        .interp(y=grid.ym, x=grid.xm[-1], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "ym"})
        .rename("veast")
        .drop(["x"])
    )
    weast = (
        east["w"]
        .interp(z=grid.zm)
        .interp(y=grid.yt, x=grid.xm[-1], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zm", "y": "yt"})
        .rename("weast")
        .drop(["x"])
    )
    thleast = (
        east["thl"]
        .interp(z=grid.zt)
        .interp(y=grid.yt, x=grid.xm[-1], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "yt"})
        .rename("thleast")
        .drop(["x"])
    )
    qteast = (
        east["qt"]
        .interp(z=grid.zt)
        .interp(y=grid.yt, x=grid.xm[-1], assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "y": "yt"})
        .rename("qteast")
        .drop(["x"])
    )
    e12east = (xr.ones_like(thleast) * input_json.e12).rename("e12east")
    ueast.attrs.clear()
    veast.attrs.clear()
    weast.attrs.clear()
    thleast.attrs.clear()
    qteast.attrs.clear()
    # South boundary
    usouth = (
        south["u"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[0], x=grid.xm, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xm"})
        .rename("usouth")
        .drop(["y"])
    )
    vsouth = (
        south["v"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[0], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xt"})
        .rename("vsouth")
        .drop(["y"])
    )
    wsouth = (
        south["w"]
        .interp(z=grid.zm)
        .interp(y=grid.ym[0], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zm", "x": "xt"})
        .rename("wsouth")
        .drop(["y"])
    )
    thlsouth = (
        south["thl"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[0], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xt"})
        .rename("thlsouth")
        .drop(["y"])
    )
    qtsouth = (
        south["qt"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[0], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xt"})
        .rename("qtsouth")
        .drop(["y"])
    )
    e12south = (xr.ones_like(thlsouth) * input_json.e12).rename("e12south")
    usouth.attrs.clear()
    vsouth.attrs.clear()
    wsouth.attrs.clear()
    thlsouth.attrs.clear()
    qtsouth.attrs.clear()
    # North boundary
    unorth = (
        north["u"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[-1], x=grid.xm, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xm"})
        .rename("unorth")
        .drop(["y"])
    )
    vnorth = (
        north["v"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[-1], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xt"})
        .rename("vnorth")
        .drop(["y"])
    )
    wnorth = (
        north["w"]
        .interp(z=grid.zm)
        .interp(y=grid.ym[-1], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zm", "x": "xt"})
        .rename("wnorth")
        .drop(["y"])
    )
    thlnorth = (
        north["thl"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[-1], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xt"})
        .rename("thlnorth")
        .drop(["y"])
    )
    qtnorth = (
        north["qt"]
        .interp(z=grid.zt)
        .interp(y=grid.ym[-1], x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"z": "zt", "x": "xt"})
        .rename("qtnorth")
        .drop(["y"])
    )
    e12north = (xr.ones_like(thlnorth) * input_json.e12).rename("e12north")
    unorth.attrs.clear()
    vnorth.attrs.clear()
    wnorth.attrs.clear()
    thlnorth.attrs.clear()
    qtnorth.attrs.clear()
    # Top boundary
    utop = (
        top["u"]
        .interp(z=grid.zm[-1])
        .interp(y=grid.yt, x=grid.xm, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"y": "yt", "x": "xm"})
        .rename("utop")
        .drop(["z"])
    )
    vtop = (
        top["v"]
        .interp(z=grid.zm[-1])
        .interp(y=grid.ym, x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"y": "ym", "x": "xt"})
        .rename("vtop")
        .drop(["z"])
    )
    wtop = (
        top["w"]
        .interp(z=grid.zm[-1])
        .interp(y=grid.yt, x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"y": "yt", "x": "xt"})
        .rename("wtop")
        .drop(["z"])
    )
    thltop = (
        top["thl"]
        .interp(z=grid.zm[-1])
        .interp(y=grid.yt, x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"y": "yt", "x": "xt"})
        .rename("thltop")
        .drop(["z"])
    )
    qttop = (
        top["qt"]
        .interp(z=grid.zm[-1])
        .interp(y=grid.yt, x=grid.xt, assume_sorted=True)
        .transpose("time", "z", "y", "x", ..., missing_dims="warn")
        .rename({"y": "yt", "x": "xt"})
        .rename("qttop")
        .drop(["z"])
    )
    e12top = (xr.ones_like(thltop) * input_json.e12).rename("e12top")
    utop.attrs.clear()
    vtop.attrs.clear()
    wtop.attrs.clear()
    thltop.attrs.clear()
    qttop.attrs.clear()
    # Add fields to dataset
    openboundaries = xr.merge(
        [
            uwest,
            vwest,
            wwest,
            thlwest,
            qtwest,
            e12west,
            ueast,
            veast,
            weast,
            thleast,
            qteast,
            e12east,
            usouth,
            vsouth,
            wsouth,
            thlsouth,
            qtsouth,
            e12south,
            unorth,
            vnorth,
            wnorth,
            thlnorth,
            qtnorth,
            e12north,
            utop,
            vtop,
            wtop,
            thltop,
            qttop,
            e12top,
        ],
        combine_attrs="drop",
    )

    return openboundaries
