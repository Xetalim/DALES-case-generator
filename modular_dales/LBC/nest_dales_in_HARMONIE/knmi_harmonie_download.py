"""Simulation module for downloading KNMI HARMONIE forecast archives."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modular_dales.IO_helpers.external_data_cache import cache_root
from modular_dales.LBC.nest_dales_in_HARMONIE.knmi_api import (
    ForecastDownloadResult,
    download_harmonie_forecast_pair,
)
from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


def resolve_knmi_harmonie_download_module(
    sim,
) -> Optional["KNMIHarmonieForecastDownloadModule"]:
    if sim is None:
        return None
    if not sim.module_exists(KNMIHarmonieForecastDownloadModule):
        return None
    module = sim.retrieve_module(KNMIHarmonieForecastDownloadModule)
    module.ensure_prepared()
    return module


@register_module
@dataclass
class KNMIHarmonieForecastDownloadModule(simulation_module):
    """Download and convert a KNMI HARMONIE forecast pair before preprocessing.

    The module downloads the operational P5 and P3 tar archives for a
    specific forecast run, unpacks them into a cache folder and converts
    each GRIB member to NetCDF using CDO. The resulting NetCDF globs are
    exposed as ``ml_glob`` and ``sfc_glob`` so that the standard KNMI
    preprocessing pipeline can consume them.
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    forecast_datetime: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    download_root: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    api_key_file: str = field(
        default=".knmiapirc",
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    cdo_path: str = field(
        default="cdo",
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    check_disk_space: bool | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    minimum_free_space_gb: float = field(
        default=10.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    disk_space_safety_factor: float = field(
        default=2.0,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    delete_grib_files: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    skip_cdo: bool = field(
        default=False,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    convert_to_netcdf: bool = field(
        default=True,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )

    ml_glob: str | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    sfc_glob: str | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    p5_tar_path: str | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    p3_tar_path: str | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "KNMIHarmonieForecastDownloadModule"

    def _resolve_forecast_datetime(self) -> str:
        value = self.forecast_datetime
        if value in (None, ""):
            raise ValueError(
                "KNMIHarmonieForecastDownloadModule requires forecast_datetime to be set"
            )
        return str(value)

    def _resolve_output_root(self) -> Path:
        if self.download_root not in (None, ""):
            return Path(self.download_root)
        return cache_root("knmi_harmonie", self.sim)

    def _resolve_check_disk_space(self) -> bool:
        if self.check_disk_space is not None:
            return bool(self.check_disk_space)
        return True

    def check_settings(self):
        self._resolve_forecast_datetime()

    def prepare_calculation(self):
        forecast_datetime = self._resolve_forecast_datetime()
        output_root = self._resolve_output_root()
        check_disk_space = self._resolve_check_disk_space()
        minimum_free_space_gb = float(self.minimum_free_space_gb)
        disk_space_safety_factor = float(self.disk_space_safety_factor)
        delete_grib_files = bool(self.delete_grib_files)
        result: ForecastDownloadResult = download_harmonie_forecast_pair(
            forecast_datetime,
            output_root,
            api_key_file=self.api_key_file,
            cdo_path=self.cdo_path,
            check_disk_space=check_disk_space,
            minimum_free_space_gb=minimum_free_space_gb,
            disk_space_safety_factor=disk_space_safety_factor,
            delete_grib_files=delete_grib_files,
            skip_cdo=self.skip_cdo,
            convert_to_netcdf=self.convert_to_netcdf,
        )
        if self.skip_cdo or not self.convert_to_netcdf:
            self.ml_glob = result.p5_grib_glob
            self.sfc_glob = result.p3_grib_glob
        else:
            self.ml_glob = result.p5_nc_glob
            self.sfc_glob = result.p3_nc_glob
        self.p5_tar_path = str(result.p5_tar)
        self.p3_tar_path = str(result.p3_tar)
        logger.info(
            "Downloaded KNMI HARMONIE forecast %s to %s",
            forecast_datetime,
            output_root,
        )

    def write_files(self):
        return None
