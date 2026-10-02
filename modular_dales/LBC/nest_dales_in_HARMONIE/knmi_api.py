"""Utilities for downloading and converting KNMI HARMONIE forecast archives."""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.dataplatform.knmi.nl/open-data/v1"
P5_DATASET = ("harmonie_arome_cy43_p5", "1.0")
P3_DATASET = ("harmonie_arome_cy43_p3", "1.0")
DEFAULT_API_KEY_FILE = ".knmiapirc"
GB = 1024**3


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    version: str
    archive_prefix: str
    output_glob_name: str


@dataclass(frozen=True)
class DownloadedForecastPaths:
    forecast_datetime: dt.datetime
    p5_tar: Path
    p3_tar: Path
    p5_extract_dir: Path
    p3_extract_dir: Path
    p5_nc_glob: str
    p3_nc_glob: str


P5_ARCHIVE = DatasetConfig(
    name=P5_DATASET[0],
    version=P5_DATASET[1],
    archive_prefix="HARM43_V1_P5",
    output_glob_name="HARM43_V1_P5",
)
P3_ARCHIVE = DatasetConfig(
    name=P3_DATASET[0],
    version=P3_DATASET[1],
    archive_prefix="HARM43_V1_P3",
    output_glob_name="HARM43_V1_P3",
)


@dataclass(frozen=True)
class ForecastDownloadResult:
    forecast_datetime: dt.datetime
    p5_tar: Path
    p3_tar: Path
    p5_extract_dir: Path
    p3_extract_dir: Path
    p5_nc_glob: str | None = None
    p3_nc_glob: str | None = None
    p5_grib_glob: str | None = None
    p3_grib_glob: str | None = None


class OpenDataAPI:
    """Small wrapper around the KNMI Open Data REST API."""

    def __init__(self, api_token: str):
        self.base_url = API_URL
        self.headers = {"Authorization": api_token}

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        response = requests.get(url, headers=self.headers, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_file_url(
        self, dataset_name: str, dataset_version: str, file_name: str
    ) -> dict:
        return self._get_json(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        )


def _load_api_key_from_file(filename: str = DEFAULT_API_KEY_FILE) -> str:
    api_key = os.environ.get("KNMI_API_TOKEN")
    if api_key:
        return api_key.strip()

    requested_path = Path(filename).expanduser()
    candidate_paths = [requested_path]
    if not requested_path.is_absolute():
        candidate_paths.insert(0, Path.cwd() / requested_path)
        candidate_paths.append(Path.home() / requested_path)

    for path in candidate_paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"KNMI API key file '{path}' is empty")
        return content

    raise FileNotFoundError(
        f"KNMI API key file '{filename}' not found in {Path.cwd()} or {Path.home()} and KNMI_API_TOKEN is unset."
    )


def _parse_forecast_datetime(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value

    text = str(value).strip()
    for fmt in (
        "%Y%m%d%H",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            "forecast_datetime must be a datetime or an ISO-like string such as '2026-06-27T21:00:00' or '2026062721'"
        ) from exc


def _archive_filename(product: DatasetConfig, forecast_datetime: dt.datetime) -> str:
    return f"{product.archive_prefix}_{forecast_datetime.strftime('%Y%m%d%H')}.tar"


def _download_file(download_url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(download_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _download_temporary_url(download_url: str, target_path: Path) -> None:
    _download_file(download_url, target_path)


def _ensure_free_space(target_dir: Path, required_bytes: int) -> None:
    usage = shutil.disk_usage(target_dir)
    if usage.free < required_bytes:
        raise OSError(
            f"Not enough free disk space in {target_dir}: need {required_bytes / GB:.1f} GB, have {usage.free / GB:.1f} GB"
        )


def _tar_member_bytes(tar_path: Path) -> int:
    total = 0
    with tarfile.open(tar_path, "r") as archive:
        for member in archive.getmembers():
            if member.isfile():
                total += member.size
    return total


def _extract_tar(tar_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(extract_dir)
        return [
            extract_dir / member.name
            for member in archive.getmembers()
            if member.isfile()
        ]


def _run_cdo_convert(cdo_path: str, source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            cdo_path,
            "--eccodes",
            "-O",
            "-f",
            "nc4",
            "-z",
            "zip_6",
            "copy",
            str(source_path),
            str(target_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _convert_grib_files(
    grib_files: Iterable[Path],
    nc_dir: Path,
    cdo_path: str,
    delete_grib_files: bool,
) -> None:
    nc_dir.mkdir(parents=True, exist_ok=True)
    for grib_path in sorted(grib_files):
        if not grib_path.exists() or not grib_path.is_file():
            continue
        output_path = nc_dir / f"{grib_path.name}.nc"
        if output_path.exists():
            continue
        _run_cdo_convert(cdo_path, grib_path, output_path)
        if delete_grib_files:
            try:
                grib_path.unlink()
            except OSError:
                logger.warning("Unable to remove temporary GRIB file %s", grib_path)


def _download_archive(
    api: OpenDataAPI,
    product: DatasetConfig,
    forecast_datetime: dt.datetime,
    target_dir: Path,
) -> Path:
    archive_name = _archive_filename(product, forecast_datetime)
    target_path = target_dir / archive_name
    if target_path.exists():
        return target_path

    response = api.get_file_url(product.name, product.version, archive_name)
    if response.get("error"):
        raise FileNotFoundError(
            f"Unable to retrieve archive {archive_name} from dataset {product.name}: {response['error']}"
        )
    if response.get("message") == "Not Found":
        raise FileNotFoundError(
            f"Archive {archive_name} not found in dataset {product.name}"
        )

    _download_temporary_url(response["temporaryDownloadUrl"], target_path)
    logger.info("Downloaded KNMI archive %s", target_path.name)
    return target_path


def download_harmonie_forecast_pair(
    forecast_datetime: str | dt.datetime,
    target_root: str | os.PathLike[str],
    *,
    api_key_file: str = DEFAULT_API_KEY_FILE,
    cdo_path: str = "cdo",
    check_disk_space: bool = True,
    minimum_free_space_gb: float = 10.0,
    disk_space_safety_factor: float = 2.0,
    delete_grib_files: bool = True,
    skip_cdo: bool = False,
    convert_to_netcdf: bool = True,
) -> ForecastDownloadResult:
    """Download and convert the KNMI HARMONIE P5/P3 forecast pair.

    The archives are downloaded as ``HARM43_V1_P5_YYYYMMDDHH.tar`` and
    ``HARM43_V1_P3_YYYYMMDDHH.tar``. Each tar is unpacked into its own
    directory and every GRIB member is optionally converted to NetCDF via CDO.
    """

    forecast_dt = _parse_forecast_datetime(forecast_datetime)
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)

    api = OpenDataAPI(api_token=_load_api_key_from_file(api_key_file))

    p5_tar = _download_archive(api, P5_ARCHIVE, forecast_dt, root)
    p3_tar = _download_archive(api, P3_ARCHIVE, forecast_dt, root)

    p5_extract_dir = root / p5_tar.stem
    p3_extract_dir = root / p3_tar.stem
    p5_nc_dir = p5_extract_dir / "netcdf"
    p3_nc_dir = p3_extract_dir / "netcdf"

    if check_disk_space:
        required_p5 = int(_tar_member_bytes(p5_tar) * disk_space_safety_factor)
        required_p3 = int(_tar_member_bytes(p3_tar) * disk_space_safety_factor)
        minimum_free_bytes = int(minimum_free_space_gb * GB)
        _ensure_free_space(root, max(required_p5, minimum_free_bytes))
        _ensure_free_space(root, max(required_p3, minimum_free_bytes))

    should_convert = convert_to_netcdf and not skip_cdo

    if should_convert:
        if not list(p5_nc_dir.glob("*.nc")):
            p5_grib_files = _extract_tar(p5_tar, p5_extract_dir)
            _convert_grib_files(p5_grib_files, p5_nc_dir, cdo_path, delete_grib_files)

        if not list(p3_nc_dir.glob("*.nc")):
            p3_grib_files = _extract_tar(p3_tar, p3_extract_dir)
            _convert_grib_files(p3_grib_files, p3_nc_dir, cdo_path, delete_grib_files)
    else:
        # Skip CDO conversion: just extract GRIB files if not already present
        p5_gribs = (
            [f for f in p5_extract_dir.iterdir() if f.is_file()]
            if p5_extract_dir.exists()
            else []
        )
        if not p5_gribs:
            _extract_tar(p5_tar, p5_extract_dir)

        p3_gribs = (
            [f for f in p3_extract_dir.iterdir() if f.is_file()]
            if p3_extract_dir.exists()
            else []
        )
        if not p3_gribs:
            _extract_tar(p3_tar, p3_extract_dir)

    p5_nc_glob = str(p5_nc_dir / "*.nc") if list(p5_nc_dir.glob("*.nc")) else None
    p3_nc_glob = str(p3_nc_dir / "*.nc") if list(p3_nc_dir.glob("*.nc")) else None
    p5_grib_glob = str(p5_extract_dir / "*_GB")
    p3_grib_glob = str(p3_extract_dir / "*_GB")

    return ForecastDownloadResult(
        forecast_datetime=forecast_dt,
        p5_tar=p5_tar,
        p3_tar=p3_tar,
        p5_extract_dir=p5_extract_dir,
        p3_extract_dir=p3_extract_dir,
        p5_nc_glob=p5_nc_glob,
        p3_nc_glob=p3_nc_glob,
        p5_grib_glob=p5_grib_glob,
        p3_grib_glob=p3_grib_glob,
    )
