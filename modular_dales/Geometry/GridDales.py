import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
from pyproj import CRS

from modular_dales.modular.simulation_module import simulation_module
from modular_dales.MODULE_REGISTRY import register_module

logger = logging.getLogger(__name__)


@register_module
@dataclass
class GridDales(simulation_module):
    """Grid configuration module for DALES simulation.

    Handles all grid-related namelist parameters in the DOMAIN section.
    Initialized with a complete domain_info dict containing all grid parameters.
    This module should be added after DefaultNamelistModule and before other modules,
    as it configures the grid that other modules depend on.

    Args:
        domain_info: Dictionary containing domain configuration with keys:
            - itot, jtot, kmax: Grid dimensions
            - xsize, ysize: Domain physical dimensions
            - kmax_soil: Number of soil levels
            - xlat, xlon: Latitude/longitude
            - x0, y0: Grid origin
            - alpha: Vertical stretch factor
            - dz0: Initial vertical spacing
            - proj4 (optional): Projection string
        sim: Parent dales_simulation instance
    """

    sim: Optional["simulation_module"] = field(default=None, repr=False)
    itot: int = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "itot", "required": True},
    )
    jtot: int = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "jtot", "required": True},
    )
    kmax: int = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "kmax", "required": True},
    )
    xsize: float = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "xsize", "required": True},
    )
    ysize: float = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "ysize", "required": True},
    )
    kmax_soil: int = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "serialize": True,
            "nml": "DOMAIN",
            "key": "kmax_soil",
            "required": True,
        },
    )
    xlat: float = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "xlat", "required": False},
    )
    xlon: float = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True, "nml": "DOMAIN", "key": "xlon", "required": False},
    )
    x0: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "serialize": True,
            "nml": "DOMAIN",
            "key": "x0",
            "required": False,
        },  # TODO add x0 if merged
    )
    y0: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "serialize": True,
            "nml": "DOMAIN",
            "key": "y0",
            "required": False,
        },  # TODO add y0 if merged
    )
    alpha: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "serialize": True,
        },
    )
    dz0: float | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={"serialize": True},
    )
    proj4: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "serialize": True,
        },
    )
    wkt: str | None = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "serialize": True,
        },
    )
    crs: str | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "serialize": False,
        },
    )

    def do_config(self):
        """Configure DOMAIN section from domain_info and create GridDales."""
        self.apply_namelist_from_fields()
        logger.info("GridModule: DOMAIN section configured from domain_info")

        if self.sim is not None:
            self.sim.grid = self
        else:
            logger.critical("GridModule: No parent simulation to attach grid to")
            raise RuntimeError(
                "GridModule must be added to a dales_simulation instance"
            )
        logger.info("GridModule: GridDales created and attached to simulation")

    def prepare_calculation(self):
        """No additional preparation needed."""
        return

    def check_settings(self):
        """Check grid settings validity."""
        return

    def write_files(self):
        """No files to write for grid module."""
        return

    def as_dic(self):
        dic = {}
        for f in fields(self):
            # skip field if it is not serialized
            if not f.metadata.get("serialize", False):
                continue
            dic[f.name] = getattr(self, f.name)
        return dic

    def __post_init__(self):
        super().__init__(self.sim)
        self.module_name = "GridModule"
        self.input_dic = self.as_dic()  # Store all fields for later use in as_openbc

        if self.proj4 and self.wkt:
            logger.warning(
                "Both proj4 and wkt are set on GridDales. proj4 will be used for CF grid mapping and wkt will be ignored."
            )
        if self.proj4:
            self.crs = self.proj4
        elif self.wkt:
            self.crs = self.wkt
        else:
            logger.debug(
                "Neither proj4 nor wkt is set on GridDales. CRS will be undefined and some functionality may not work."
            )
            self.crs = None
        # self.xsize = input_dic["xsize"]
        # self.ysize = input_dic["ysize"]

        # self.itot = input_dic["itot"]
        # self.jtot = input_dic["jtot"]
        # self.kmax = input_dic["kmax"]

        self.imax = self.itot
        self.jmax = self.jtot
        self.i1 = self.imax + 1
        self.j1 = self.jmax + 1
        self.k1 = self.kmax + 1
        self.k2 = self.kmax + 2
        self.i2 = self.imax + 2
        self.j2 = self.jmax + 2

        # self.x0 = input_dic["x0"]
        # self.y0 = input_dic["y0"]

        self.dx = self.xsize / self.itot
        self.dy = self.ysize / self.jtot

        i_arr = np.arange(self.itot)
        j_arr = np.arange(self.jtot)

        self.xt = self.x0 + (i_arr + 0.5) * self.dx
        self.yt = self.y0 + (j_arr + 0.5) * self.dy

        self.xm = self.x0 + i_arr * self.dx
        self.ym = self.y0 + j_arr * self.dy

        # self.xt = self.x0 + np.arange(0.5 * self.dx, self.xsize, self.dx)
        # self.yt = self.y0 + np.arange(0.5 * self.dy, self.ysize, self.dy)

        self.xt_openbc = self.xt
        self.yt_openbc = self.yt

        self.xm_openbc = self.x0 + np.arange(0, self.xsize + self.dx, self.dx)
        self.ym_openbc = self.y0 + np.arange(0, self.ysize + self.dy, self.dy)

        # self.xt_ghost = self.x0 + (i_arr + 0.5) * self.dx
        # self.yt_ghost = self.y0 + (j_arr + 0.5) * self.dy

        # self.xm_ghost = self.x0 + i_arr * self.dx
        # self.ym_ghost = self.y0 + j_arr * self.dy
        if self.alpha != 1.0:
            # Stretched height grid
            self.dz = np.zeros(self.kmax)
            self.zt = np.zeros(self.kmax)
            self.zm = np.zeros(self.kmax + 1)
            self.dz[:] = self.dz0 * (self.alpha) ** np.arange(self.kmax)
            self.zm[1:] = np.cumsum(self.dz)
            self.zt[:] = 0.5 * (self.zm[1:] + self.zm[:-1])
            self.zsize = self.zm[-1]
        else:
            # Equidistant height grid
            self.dz = np.ones(self.kmax) * self.dz0
            self.zsize = self.kmax * self.dz0
            self.zt = np.arange(0.5 * self.dz0, self.zsize, self.dz0)
            self.zm = np.arange(0, self.zsize + self.dz0, self.dz0)

    def as_openbc(self):
        openbc_grid = GridDalesOpenBC(**self.input_dic)
        openbc_grid.zt = self.zt
        openbc_grid.zm = self.zm
        return openbc_grid

    def set_cf_grid_mapping(
        self,
        ds,
        mapping_var_name: str = "crs",
        data_var_names: Sequence[str] | None = None,
    ):
        """Attach CF-compliant grid_mapping information to a NetCDF file.

        This helper uses the grid's ``proj4`` definition together with
        :class:`pyproj.CRS` to derive CF-compliant projection metadata and
        writes it to a dedicated *grid mapping* variable in the provided
        NetCDF dataset.

        The method operates in-place on an already opened dataset and can
        handle both ``netCDF4.Dataset`` and ``xarray.Dataset`` objects.

        Args:
            ds: Open NetCDF dataset (``netCDF4.Dataset`` or ``xarray.Dataset``)
                that will be modified in-place.
            mapping_var_name: Name of the grid mapping variable that will be
                created or updated (default: ``"crs"``).
            data_var_names: Optional list of data variable names that should
                receive a ``grid_mapping = mapping_var_name`` attribute.

        Returns:
            For ``xarray.Dataset`` inputs, the modified dataset is returned to
            allow method chaining. For ``netCDF4.Dataset`` inputs, ``None`` is
            returned as the dataset is modified in-place on disk.
        """

        if not self.crs:
            logger.debug(
                "GridDales.set_cf_grid_mapping called but no CRS (proj4 or wkt) is set on the grid"
            )
            return ds

        try:
            from netCDF4 import Dataset as NetCDF4Dataset  # type: ignore
        except (
            Exception
        ):  # pragma: no cover - netCDF4 may not be available in all contexts
            NetCDF4Dataset = None  # type: ignore

        try:
            import xarray as xr
        except (
            Exception
        ):  # pragma: no cover - xarray may not be available in all contexts
            xr = None  # type: ignore

        is_netcdf4 = NetCDF4Dataset is not None and isinstance(ds, NetCDF4Dataset)
        is_xarray = xr is not None and isinstance(ds, xr.Dataset)

        if not (is_netcdf4 or is_xarray):
            raise TypeError(
                "ds must be a netCDF4.Dataset or xarray.Dataset, got "
                f"{type(ds)!r} instead"
            )

        if self.crs:
            crs = CRS.from_user_input(self.crs)
        else:
            logger.warning(
                "GridDales.set_cf_grid_mapping called but neither proj4 nor wkt is set on the grid"
            )
            return ds
        cf_attributes = crs.to_cf()
        if cf_attributes.get("grid_mapping_name") is None:
            logger.warning(
                "CRS provided to GridDales has no CF grid_mapping_name; writing WKT-only CRS attributes"
            )

        # GDAL/QGIS look for one of these WKT keys on the grid mapping variable.
        crs_wkt = crs.to_wkt()
        cf_attributes["crs_wkt"] = crs_wkt
        cf_attributes["spatial_ref"] = crs_wkt

        mapping_var_name = "crs"

        def _find_spatial_var_names_netcdf4(dataset):
            horizontal_dims = {
                dim_name
                for dim_name in ("x", "y", "xt", "yt", "xm", "ym")
                if dim_name in dataset.dimensions
            }
            spatial_names = []
            for var_name, var in dataset.variables.items():
                if var_name == mapping_var_name:
                    continue
                if len(var.dimensions) < 2:
                    continue
                if any(dim in horizontal_dims for dim in var.dimensions):
                    spatial_names.append(var_name)
            return spatial_names

        def _find_spatial_var_names_xarray(dataset):
            horizontal_dims = {
                dim_name
                for dim_name in ("x", "y", "xt", "yt", "xm", "ym")
                if dim_name in dataset.dims
            }
            spatial_names = []
            for var_name, var in dataset.data_vars.items():
                if var_name == mapping_var_name:
                    continue
                if len(var.dims) < 2:
                    continue
                if any(dim in horizontal_dims for dim in var.dims):
                    spatial_names.append(var_name)
            return spatial_names

        if data_var_names is None:
            if is_netcdf4:
                data_var_names = _find_spatial_var_names_netcdf4(ds)
            else:
                data_var_names = _find_spatial_var_names_xarray(ds)

        # Ensure the grid mapping variable exists and carries CF attributes
        if is_netcdf4:
            if mapping_var_name in ds.variables:
                mapping_var = ds.variables[mapping_var_name]
            else:
                mapping_var = ds.createVariable(mapping_var_name, "i4")
                mapping_var[...] = 1

            for key, value in cf_attributes.items():
                setattr(mapping_var, key, value)

            if data_var_names is not None:
                for var_name in data_var_names:
                    if var_name in ds.variables:
                        ds.variables[var_name].grid_mapping = mapping_var_name
                    else:
                        logger.warning(
                            "Variable '%s' not found in NetCDF dataset when setting grid_mapping",
                            var_name,
                        )

            # Set CF-compliant metadata on x/y coordinate variables if present
            for coord_name, axis, std_name in (
                ("x", "X", "projection_x_coordinate"),
                ("y", "Y", "projection_y_coordinate"),
            ):
                if coord_name in ds.variables:
                    coord_var = ds.variables[coord_name]
                    coord_var.standard_name = std_name
                    coord_var.units = "m"
                    coord_var.axis = axis

            return None

        # xarray.Dataset case
        if mapping_var_name in ds:
            mapping_da = ds[mapping_var_name]
        else:
            mapping_da = xr.DataArray(1, name=mapping_var_name)
            ds[mapping_var_name] = mapping_da

        ds[mapping_var_name].attrs.update(cf_attributes)

        if data_var_names is not None:
            for var_name in data_var_names:
                if var_name in ds.data_vars:
                    ds[var_name].attrs["grid_mapping"] = mapping_var_name
                else:
                    logger.warning(
                        "Variable '%s' not found in xarray Dataset when setting grid_mapping",
                        var_name,
                    )

        # Set CF-compliant metadata on x/y coordinate variables if present
        for coord_name, axis, std_name in (
            ("x", "X", "projection_x_coordinate"),
            ("y", "Y", "projection_y_coordinate"),
        ):
            if coord_name in ds.coords:
                coord_da = ds.coords[coord_name]
            elif coord_name in ds:
                coord_da = ds[coord_name]
            else:
                continue

            coord_da.attrs.setdefault("standard_name", std_name)
            coord_da.attrs.setdefault("units", "m")
            coord_da.attrs.setdefault("axis", axis)

        return ds

    def __repr__(self):
        """Pretty print GridDales configuration with grid dimensions and spacing."""
        grid_type = "Stretched" if (self.alpha != 1.0) else "Equidistant"
        proj_info = f", proj4={self.proj4!r}" if self.proj4 else ""

        header = f"GridDales({grid_type} grid{proj_info})"
        separator = "=" * (len(header) + 4)

        # Dimensions
        dims = f"  Dimensions:   ({self.itot:5d} × {self.jtot:5d} × {self.kmax:4d}) [i × j × k]"

        # Physical sizes
        sizes = f"  Domain size:  ({self.xsize:10.2f} × {self.ysize:10.2f} × {self.zsize:10.2f}) [m]"

        # Origin
        origin = f"  Origin:       ({self.x0:10.2f}, {self.y0:10.2f}) [m]"

        # Spacing
        spacing_str = f"  Grid spacing: dx={self.dx:.4f}m, dy={self.dy:.4f}m"
        if self.alpha:
            spacing_str += f", dz₀={self.dz0:.4f}m (α={self.alpha:.4f})"
        else:
            spacing_str += f", dz={self.dz0:.4f}m"

        # Array bounds
        x_bounds = f"  X range:      [{self.xm.min():.2f}, {self.xm.max():.2f}] m"
        y_bounds = f"  Y range:      [{self.ym.min():.2f}, {self.ym.max():.2f}] m"
        z_bounds = f"  Z range:      [{self.zm.min():.2f}, {self.zm.max():.2f}] m"

        # Array info
        arrays = (
            f"  Arrays: xt[{len(self.xt)}], yt[{len(self.yt)}], zt[{len(self.zt)}], "
            f"xm[{len(self.xm)}], ym[{len(self.ym)}], zm[{len(self.zm)}]"
        )

        return (
            f"{separator}\n"
            f"  {header}\n"
            f"{separator}\n"
            f"{dims}\n"
            f"{sizes}\n"
            f"{origin}\n"
            f"{spacing_str}\n"
            f"{x_bounds}\n"
            f"{y_bounds}\n"
            f"{z_bounds}\n"
            f"{arrays}\n"
            f"{separator}"
        )


class GridDalesOpenBC(GridDales):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xt = self.x0 + np.arange(0.5 * self.dx, self.xsize, self.dx)
        self.yt = self.y0 + np.arange(0.5 * self.dy, self.ysize, self.dy)

        self.xm = self.x0 + np.arange(0, self.xsize + self.dx, self.dx)
        self.ym = self.y0 + np.arange(0, self.ysize + self.dy, self.dy)
