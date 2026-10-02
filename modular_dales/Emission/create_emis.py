import datetime
import logging
import os
from dataclasses import dataclass

import netCDF4 as nc
import numpy as np

from modular_dales.Geometry import GridDales
from modular_dales.logging_wrapper import logwrap

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@dataclass
class TracerInfo:
    name: str
    long_name: str
    unit: str
    molar_mass: float
    lemis: bool = False
    lreact: bool = False
    ldep: bool = False
    lags: bool = False
    profile: np.ndarray | None = None


@dataclass
class PointSource:
    x_idx: int
    y_idx: int
    height: float
    temperature: float
    volume: float
    emission: float
    stack_exit_area: float


@dataclass
class Tracer:
    info: TracerInfo
    profile: list | np.ndarray
    pointsources: list[PointSource]


class emissions:
    @logwrap
    def __init__(self, output_path, grid: GridDales):
        self.output_path = output_path
        self.tracers = {}
        self.tracer_profiles = {}
        self.grid = grid

    def add_tracer(self, info: TracerInfo):
        self.tracers[info.name] = Tracer(info, np.zeros_like(self.grid.zt), [])

    def add_pointsource(self, tracername, source_info: PointSource):
        self.tracers[tracername].pointsources.append(source_info)

    def add_pre_existing_tracer(self, tracername):
        self.add_tracer(pre_existing_tracers[tracername])

    def set_profile(self, tracername, profile):
        self.tracers[tracername].profile = profile

    def set_tracer_property(self, tracername, property_name, property_value):
        # check if the property is valid
        if property_name == "name":
            raise KeyError("Can't change name of tracer!")
        if not (
            property_name
            in ["long_name", "unit", "molar_mass", "lemis", "lreact", "ldep", "lags"]
        ):
            raise KeyError(f"Unknown property {property_name} to set to tracer.")
        setattr(self.tracers[tracername].info, property_name, property_value)

    @logwrap
    def tracerinfo_to_netcdf(self, file, tracername):
        info = self.tracers[tracername].info

        new_tracer = file.createVariable(info.name, float, "z")
        new_tracer.long_name = info.long_name
        new_tracer.unit = info.unit
        new_tracer.molar_mass = info.molar_mass
        new_tracer.lemis = int(info.lemis)
        new_tracer.lreact = int(info.lreact)
        new_tracer.ldep = int(info.ldep)
        new_tracer.lags = int(info.lags)
        new_tracer[:] = self.tracers[tracername].profile

    @logwrap
    def emissionmap_to_netcdf(self, datetime_str, tracername):
        with nc.Dataset(
            self.output_path
            / "input"
            / f"emissions/{tracername}_emis_{datetime_str}_3d.nc",
            "w",
        ) as file:
            file.createDimension("x", len(self.grid.xt))
            file.createDimension("y", len(self.grid.yt))
            file.createDimension("z", len(self.grid.zt))
            # file.createDimension("t", 1)
            file.createVariable("x", "f8", ("x",))[:] = self.grid.xt
            file.createVariable("y", "f8", ("y",))[:] = self.grid.yt
            file.createVariable("z", "f8", ("z",))[:] = self.grid.zt

            tracer_var = file.createVariable(tracername, float, ("z", "y", "x"))
            tracer_var.units = "kg hour-1"
            tracer_var[:, :, :] = 0

    @logwrap
    def pointsources_to_netcdf(self, datetime_str, tracername):
        num_pointsources = len(self.tracers[tracername].pointsources)
        source_list = self.tracers[tracername].pointsources

        with nc.Dataset(
            self.output_path
            / "input"
            / f"emissions/pointsources.{datetime_str}.{tracername}.nc",
            "w",
        ) as file:
            file.createDimension("n", num_pointsources)
            nc_x_var = file.createVariable("x_idx", "f8", ("n",))
            nc_y_var = file.createVariable("y_idx", "f8", ("n",))

            nc_height_var = file.createVariable("height", "f8", ("n",))
            nc_temperature_var = file.createVariable("temperature", "f8", ("n",))
            nc_volume_var = file.createVariable("volume", "f8", ("n",))
            nc_emission_var = file.createVariable("emission", "f8", ("n",))
            nc_stack_exit_area_var = file.createVariable(
                "stack_exit_area", "f8", ("n",)
            )

            nc_x_var[:] = [source.x_idx for source in source_list]
            nc_y_var[:] = [source.y_idx for source in source_list]
            nc_height_var[:] = [source.height for source in source_list]

            nc_temperature_var[:] = [source.temperature for source in source_list]
            nc_volume_var[:] = [source.volume for source in source_list]
            nc_emission_var[:] = [source.emission for source in source_list]
            nc_stack_exit_area_var[:] = [
                source.stack_exit_area for source in source_list
            ]

    @logwrap
    def write_tracers_file(self):
        os.makedirs(self.output_path / "input" / "emissions", exist_ok=True)
        with nc.Dataset(self.output_path / "input" / "tracers.001.nc", "w") as file:
            file.createDimension("z", len(self.grid.zt))
            for tracername in self.tracers:
                self.tracerinfo_to_netcdf(file, tracername)

    def write_hourly_emission_data(self, datetime_str):
        for tracername in self.tracers:
            self.emissionmap_to_netcdf(datetime_str, tracername)
            self.pointsources_to_netcdf(datetime_str, tracername)


co2 = TracerInfo(
    name="co2", long_name="Carbon Dioxide (CO2)", unit="ppm", molar_mass=44.009
)
pre_existing_tracers = {"co2": co2}


def get_emis_sim_hours(nml):
    sim_start_time = datetime.datetime(
        year=int(nml["NAMDATETIME"]["startyear"]),
        month=int(nml["NAMDATETIME"]["startmonth"]),
        day=int(nml["NAMDATETIME"]["startday"]),
        hour=int(nml["DOMAIN"]["xtime"]),
    )
    end_hour = sim_start_time + datetime.timedelta(seconds=nml["RUN"]["runtime"])

    required_datetimes = []
    cur_sim_time = sim_start_time - datetime.timedelta(hours=1)
    while cur_sim_time < end_hour + datetime.timedelta(hours=1):
        required_datetimes.append(cur_sim_time)
        cur_sim_time = cur_sim_time + datetime.timedelta(hours=1)

    return [dt.strftime("%Y%m%d%H00") for dt in required_datetimes]


def parse_tracer_type(tracer_conf) -> TracerInfo:
    return TracerInfo(**tracer_conf)
