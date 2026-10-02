import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from modular_dales.Geometry import GridDalesOpenBC
from modular_dales.LBC.nest_dales_in_dales.get_all_dales_boundaries import (
    get_all_dales_boundaries,
)
from modular_dales.LBC.nest_dales_in_dales.synturb import add_synthetic_turbulence
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.logging_wrapper import logwrap

if TYPE_CHECKING:
    from ..nesting_idx import NestingIndices

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@logwrap
def boundary_fields_fine(
    input_json: OpenBoundaryConfig,
    grid: GridDalesOpenBC,
    output_path,
    grid_indices: "NestingIndices",
    chunks=None,
):
    if not grid_indices:
        raise ValueError("No nesting indices provided!")

    # Get initial boundary fields from initial fields
    openboundaries = get_all_dales_boundaries(
        input_json, grid, grid_indices, chunks=chunks
    )
    if input_json.lsynturb:
        openboundaries = add_synthetic_turbulence(
            input_json, openboundaries, chunks=chunks
        )
    openboundaries = set_openboundary_attrs(input_json, openboundaries)
    # Add global attributes
    openboundaries = openboundaries.assign_attrs(
        {
            "history": f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "author": input_json.author,
            "time0": input_json.time0,
        }
    )
    return openboundaries


@logwrap
def set_openboundary_attrs(input_json: OpenBoundaryConfig, openboundaries):
    dts = (
        openboundaries.time.values.astype("datetime64[s]")
        - np.datetime64(input_json.time0, "s")
    ) / np.timedelta64(1, "s")
    openboundaries = openboundaries.assign_coords({"time": ("time", dts)})
    # # Adjust time variable to seconds since initial field
    # ts = openboundaries['time'].values.astype('datetime64[s]')
    # dts = (ts-np.datetime64(time0,'s'))/np.timedelta64(1, 's')
    # openboundaries = openboundaries.assign_coords({'time':('time', dts)})
    # Add variable attributes
    openboundaries["time"] = openboundaries["time"].assign_attrs(
        {"longname": "Time"}
    )  # , 'units': f"seconds since {time0}"})
    openboundaries.time.encoding["units"] = f"seconds since {input_json.time0}"
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
    if len(input_json.tracernames) > 0:
        for tracername in input_json.tracernames:
            variables.append(tracername)
    units = ["m/s", "m/s", "m/s", "K", "kg/kg", "m/s"]
    if len(input_json.tracernames) > 0:
        for tracername in input_json.tracernames:
            if len(input_json.tracernames) > 2:
                logger.warning(
                    "Unit applied to tracer %s might not be correct!", tracername
                )
            units.append("kg/kg")
    long_names = [
        "West-East velocity at ",
        "South-North velocity at ",
        "Vertical velocity at ",
        "Liquid water potential temperature at ",
        "Total water specific humidity at ",
        "Square root of turbulent kinetic energy at ",
    ]
    if len(input_json.tracernames) > 0:
        for tracername in input_json.tracernames:
            long_names.append(f"scalar field {tracername} at ")
    for ivar, var in enumerate(variables):
        unit = units[ivar]
        long_name = long_names[ivar]
        for boundary in ["West", "East", "South", "North", "top"]:
            openboundaries[var + boundary.lower()] = openboundaries[
                var + boundary.lower()
            ].assign_attrs(
                {"longname": long_name + boundary + " boundary", "units": unit}
            )

    if input_json.lsynturb:
        turb_variables = [
            "u2",
            "v2",
            "w2",
            "uv",
            "uw",
            "vw",
            "thl2",
            "qt2",
            "wthl",
            "wqt",
        ]
        turb_units = [
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "m2/s2",
            "K2",
            "(kg/kg)2",
            "K m/s",
            "kg/kg m/s",
        ]
        turb_long_names = [
            "Variance of u at ",
            "Variance of v at ",
            "Variance of w at ",
            "Covariance of u and v at ",
            "Covariance of u and w at ",
            "Covariance of v and w at ",
            "Variance of thl at ",
            "Variance of qt at ",
            "Covariance of w and thl at ",
            "Covariance of w and qt at ",
        ]
        for ivar, var in enumerate(turb_variables):
            unit = turb_units[ivar]
            long_name = turb_long_names[ivar]
            for boundary in ["West", "East", "South", "North", "top"]:
                key = var + boundary.lower()
                if key in openboundaries:
                    openboundaries[key] = openboundaries[key].assign_attrs(
                        {"longname": long_name + boundary + " boundary", "units": unit}
                    )

    return openboundaries
