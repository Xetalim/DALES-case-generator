import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Same names as TIME_DIM/Z_DIM in modular.time_dependent_scalars (not imported to avoid a cycle).
TIME_DIM = "time"
Z_DIM = "zt"
FILE_Z_DIM = "zh"
"""Name of the vertical dimension in DALES input files."""


def _to_netcdf(ds: xr.Dataset, path) -> None:
    ds = ds.rename({Z_DIM: FILE_Z_DIM}) if Z_DIM in ds.dims else ds
    encoding = {name: {"_FillValue": None, "dtype": "f8"} for name in ds.variables}
    ds.to_netcdf(path, format="NETCDF3_CLASSIC", encoding=encoding)


class AtmosphereProfileWriter:
    """Helper class for writing atmosphere profiles and time-dependent forcings."""

    def _plot_time_series(
        self,
        time_values: np.ndarray,
        series: np.ndarray,
        label: str,
        out_path,
        exp_id: int,
    ) -> None:
        plt.ioff()
        fig, ax = plt.subplots()
        ax.plot(time_values, series)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(label)
        ax.set_title(f"exp {exp_id:03d} forcing {label}")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        plt.ion()

    def _plot_time_height_heatmap(
        self,
        time_values: np.ndarray,
        z_values: np.ndarray,
        data_2d: np.ndarray,
        label: str,
        out_path,
        exp_id: int,
    ) -> None:
        plt.ioff()
        fig, ax = plt.subplots()
        if np.max(data_2d) < 0:
            cmap = "viridis_r"
        else:
            cmap = "viridis"
        if np.max(data_2d) > 0 and np.min(data_2d) < 0:
            cmap = "RdBu_r"
        mesh = ax.pcolormesh(
            time_values,
            z_values,
            data_2d,
            shading="nearest",
            cmap=cmap,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("z (m)")
        ax.set_title(f"exp {exp_id:03d} heatmap {label}")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        plt.ion()

    def _plot_forcing_file_variables(
        self,
        file_path,
        profiles_path,
        exp_id: int,
        prefix: str,
    ) -> None:
        if not os.path.exists(file_path):
            return

        with xr.open_dataset(
            file_path, decode_times=False, decode_timedelta=False
        ) as ds:
            for var_name, da in ds.data_vars.items():
                if da.dims == (TIME_DIM,):
                    self._plot_time_series(
                        ds[TIME_DIM].values,
                        da.values,
                        var_name,
                        profiles_path / f"{prefix}_timeseries_{var_name}.png",
                        exp_id,
                    )
                elif set(da.dims) == {TIME_DIM, FILE_Z_DIM}:
                    self._plot_time_height_heatmap(
                        ds[TIME_DIM].values,
                        ds[FILE_Z_DIM].values,
                        da.transpose(FILE_Z_DIM, TIME_DIM).values,
                        var_name,
                        profiles_path / f"{prefix}_heatmap_{var_name}.png",
                        exp_id,
                    )

    def write_init(self, ds: xr.Dataset, output_path, exp_id: int) -> None:
        """Write ``init.<exp_id>.nc``: profiles on ``zt`` and optional ``(time, zt)`` fields."""
        ds = ds.transpose(TIME_DIM, Z_DIM, missing_dims="ignore")
        ds[Z_DIM].attrs.update(long_name="Height of vertical levels", units="m")
        if TIME_DIM in ds.dims:
            ds[TIME_DIM].attrs.update(long_name="time", units="s")
        _to_netcdf(ds, output_path / f"init.{exp_id:03d}.nc")

    def write_forcings(self, ds: xr.Dataset, output_path, exp_id: int) -> None:
        """Write ``forcings.<exp_id>.nc`` from series on a common axis starting at t=0.

        The t=0 entry is dropped: DALES takes it from the initial profiles.
        """
        ds = ds.isel({TIME_DIM: slice(1, None)}).transpose(
            Z_DIM, TIME_DIM, missing_dims="ignore"
        )
        ds[Z_DIM].attrs.update(long_name="Height of vertical levels", units="m")
        ds[TIME_DIM].attrs.update(
            long_name="Time validity of time-dependent values", units="s"
        )
        _to_netcdf(ds, output_path / f"forcings.{exp_id:03d}.nc")

    def plot_profiles(self, profiles: xr.Dataset, output_path, exp_id: int) -> None:
        """Plot initial profiles and the written init/forcings fields for quick inspection."""
        profiles_path = output_path / ".." / "profiles"
        os.makedirs(profiles_path, exist_ok=True)
        for name, da in profiles.data_vars.items():
            plt.ioff()
            fig, ax = plt.subplots()
            ax.plot(da.values, da[Z_DIM].values)
            label = da.attrs.get("long_name", name)
            ax.set_xlabel(label)
            ax.set_ylabel("z (m)")
            ax.set_title(f"exp {exp_id:03d} {label}")
            fig.savefig(profiles_path / f"profile_{name}.png", dpi=300)
            plt.close(fig)
            plt.ion()

        for prefix in ("forcings", "init"):
            self._plot_forcing_file_variables(
                output_path / f"{prefix}.{exp_id:03d}.nc",
                profiles_path,
                exp_id,
                prefix=prefix,
            )
