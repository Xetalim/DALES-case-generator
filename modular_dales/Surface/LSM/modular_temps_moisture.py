from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from modular_dales.Geometry import GridDales
from modular_dales.IO_helpers.raster import fix_lambert_offsets
from modular_dales.MODULE_REGISTRY import register_module, register_special_serializing


@register_special_serializing
@register_module
@dataclass
class VaryingSkinTemperature:
    skin_temperature: np.ndarray
    aquatic_skin_temperature: float | None = None
    """Spatially varying skin temperature over the horizontal domain.

    Init: provide values as a list or array-like object.
    """


@register_special_serializing
@register_module
@dataclass
class VaryingSoilTemperature:
    soil_temperature: np.ndarray
    """Spatially varying soil temperature for the full domain.

    Init: provide values as a list or array-like object.
    """


@register_special_serializing
@register_module
@dataclass
class VaryingSoilMoisture:
    soil_moisture: np.ndarray
    """Spatially varying soil moisture for the full domain.

    Init: provide values as a list or array-like object.
    """


@register_special_serializing
@register_module
@dataclass
class UniformSkinTemperature:
    skin_temperature: float
    aquatic_skin_temperature: float | None = None
    """Uniform skin temperature (K) across the horizontal domain."""


@register_special_serializing
@register_module
@dataclass
class UniformSoilTemperature:
    soil_temperature: list[float]
    """Uniform soil temperature profile across soil levels.

    Init: provide a sequence of length kmax_soil containing temperature values.
    """


@register_special_serializing
@register_module
@dataclass
class UniformSoilMoisture:
    soil_moisture: list[float]
    """Uniform soil moisture profile across soil levels.

    Init: provide a sequence of length kmax_soil containing moisture values.
    """


@register_module
@dataclass
class SoilTemperatureMoistureFromHarmonie:
    harmonie_soil_file: str = field(
        default=None, init=True, repr=True, metadata={"serialize": True}
    )
    harmonie_soil_height_levels: list[float] = field(
        default_factory=list, init=True, repr=True, metadata={"serialize": True}
    )
    harmonie_soil_valid_time: str = field(
        default=None, init=True, repr=True, metadata={"serialize": True}
    )
    use_as_tskin: bool = field(
        default=False, init=True, repr=True, metadata={"serialize": True}
    )
    aquatic_skin_temperature: float | None = field(
        default=None, init=True, repr=True, metadata={"serialize": True}
    )
    data: xr.Dataset | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    grid: GridDales | None = field(
        default=None, init=False, repr=False, metadata={"serialize": False}
    )
    """Gridded interpolated temperatures and moistures from HARMONIE soil data.

    Init: provide the path to the HARMONIE soil file and the corresponding height levels, and valid time for the soil data.
    The soil moisture array will be extracted from the file and made available for the LSM output module.
    """

    def __post_init__(self):
        if not isinstance(self.harmonie_soil_file, str):
            raise ValueError(
                "harmonie_soil_file must be a string path to the HARMONIE soil file"
            )
        if not self.harmonie_soil_file.endswith(".nc"):
            raise ValueError(
                "harmonie_soil_file must be a NetCDF file with .nc extension"
            )

    def get_soil_temp_moisture_arrays(
        self, grid: GridDales, dz_soil: np.ndarray | list
    ):
        # Placeholder for any processing needed to read the HARMONIE soil file and extract soil moisture profiles
        ds_soil = fix_lambert_offsets(
            xr.open_dataset(
                self.harmonie_soil_file,
                decode_coords="all",
                engine="netcdf4",
            )
        )
        if isinstance(dz_soil, list):
            dz_soil = np.array(dz_soil)

        # Crop data to time and spatial range, using harmonie spatial resolution or filter as buffer
        dx = ds_soil["x"][1] - ds_soil["x"][0]
        dy = ds_soil["y"][1] - ds_soil["y"][0]

        buffer = dx
        # Merge into xarray dataset
        self.data = (
            ds_soil.sel(
                time=self.harmonie_soil_valid_time,
                x=slice(
                    int(grid.x0 - buffer),
                    int(grid.x0 + grid.xsize + buffer),
                ),
                y=slice(
                    int(grid.y0 - buffer),
                    int(grid.y0 + grid.ysize + buffer),
                ),
            )
            .interp(
                y=grid.yt,
                x=grid.xt,
                assume_sorted=False,
            )
            .rename({"lev": "k_soil", "sot": "t_soil", "liqvsm": "theta_soil"})
        )
        if self.use_as_tskin:
            tskin = self.data.t_soil.isel(k_soil=0)
        self.data = (
            self.data.assign_coords(k_soil=self.harmonie_soil_height_levels)
            .rename({"k_soil": "z_soil"})
            .interp(z_soil=dz_soil[::-1].cumsum()[::-1], assume_sorted=False)
            .drop_dims("bnds")
        )
        if self.use_as_tskin:
            self.data["tskin"] = tskin

    def get_soil_moisture_array(
        self, grid: GridDales, dz_soil: np.ndarray | list
    ) -> np.ndarray:
        if self.data is None:
            self.get_soil_temp_moisture_arrays(grid, dz_soil)
        return self.data.theta_soil.values

    def get_tskin_array(self) -> np.ndarray:
        if self.data is None:
            raise ValueError(
                "Data not loaded. Call get_soil_temp_moisture_arrays first to load the data from the HARMONIE soil file."
            )
        return self.data.tskin.values

    def get_soil_temperature_array(
        self, grid: GridDales, dz_soil: np.ndarray | list
    ) -> np.ndarray:
        if self.data is None:
            self.get_soil_temp_moisture_arrays(grid, dz_soil)
        return self.data.t_soil.values
