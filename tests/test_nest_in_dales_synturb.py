from pathlib import Path

import numpy as np
import xarray as xr

from modular_dales.LBC.nest_dales_in_dales.synturb import (
    _load_synturb_profiles,
)
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig


def _write_profiles_dataset(path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "u2r": (("time", "zt"), [[1.0, 1.0], [1.0, 1.0]]),
            "v2r": (("time", "zt"), [[1.0, 1.0], [1.0, 1.0]]),
            "w2r": (("time", "zt"), [[0.2, 0.2], [0.2, 0.2]]),
            "w2s": (("time", "zt"), [[0.0, 0.0], [0.0, 0.0]]),
            "uwt": (("time", "zt"), [[2.0, 2.0], [2.0, 2.0]]),
            "vwt": (("time", "zt"), [[-1.0, -1.0], [-1.0, -1.0]]),
            "thl2r": (("time", "zt"), [[0.05, 0.05], [0.05, 0.05]]),
            "wthlt": (("time", "zt"), [[1.0, 1.0], [1.0, 1.0]]),
            "qt2r": (("time", "zt"), [[0.02, 0.02], [0.02, 0.02]]),
            "wqtt": (("time", "zt"), [[1.0, 1.0], [1.0, 1.0]]),
        },
        coords={"time": [0, 10.0], "zt": [50.0, 150.0]},
    )
    ds.to_netcdf(path)


def test_load_synturb_profiles_repairs_covariances(tmp_path: Path) -> None:
    _write_profiles_dataset(tmp_path / "profiles.001.nc")

    synturb = _load_synturb_profiles(
        OpenBoundaryConfig(outpath_coarse=tmp_path.as_posix()),
        xr.DataArray([5.0], dims=["time"]),
        xr.DataArray([100.0], dims=["zt"]),
    )

    u2 = float(synturb["u2"].isel(time=0, zt=0))
    v2 = float(synturb["v2"].isel(time=0, zt=0))
    w2 = float(synturb["w2"].isel(time=0, zt=0))
    uw = float(synturb["uw"].isel(time=0, zt=0))
    vw = float(synturb["vw"].isel(time=0, zt=0))
    uv = float(synturb["uv"].isel(time=0, zt=0))
    thl2 = float(synturb["thl2"].isel(time=0, zt=0))
    wthl = float(synturb["wthl"].isel(time=0, zt=0))
    qt2 = float(synturb["qt2"].isel(time=0, zt=0))
    wqt = float(synturb["wqt"].isel(time=0, zt=0))

    assert abs(uw) <= np.sqrt(u2 * w2) + 1e-12
    assert abs(vw) <= np.sqrt(v2 * w2) + 1e-12
    assert abs(uv) <= np.sqrt(u2 * v2) + 1e-12
    assert abs(wthl) <= np.sqrt(w2 * thl2) + 1e-12
    assert abs(wqt) <= np.sqrt(w2 * qt2) + 1e-12

    reynolds = np.array([[u2, uv, uw], [uv, v2, vw], [uw, vw, w2]])
    eigvals = np.linalg.eigvalsh(reynolds)
    assert np.all(eigvals >= -1e-10)
