"""Worker for building KNMI-based open-boundary fields."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import dask
import numpy as np
import xarray as xr

from modular_dales.LBC.nest_dales_in_HARMONIE import (
    boundary as harmonie_boundary,
    initfields,
    nest_dales_in_KNMI,
)
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.LBC.nest_dales_in_HARMONIE.knmi_harmonie_download import (
    resolve_knmi_harmonie_download_module,
)

if TYPE_CHECKING:
    from modular_dales.LBC.openbc import do_openboundary


logger = logging.getLogger(__name__)


class OpenBCKNMIWorker:
    """Build open boundaries from KNMI operational HARMONIE output."""

    def __init__(self, module: "do_openboundary") -> None:
        self.module = module

    def prepare(self) -> Tuple[xr.Dataset, xr.Dataset]:
        config = self._build_config()
        prepper = nest_dales_in_KNMI.KNMIPrepper(
            config,
            self.module.openBCgrid,
        )
        prepper.load_data()
        data, transform = prepper.prep_harmonie()
        self.module.harmonieprepper = prepper

        logger.info("Setting namelist NAMSURFACE:ps to %f", prepper.ps)
        self.module.set_nml_section("NAMSURFACE", "ps", prepper.ps)
        logger.info("Setting namelist NAMSURFACE:thls to %f", prepper.thls)
        self.module.set_nml_section("NAMSURFACE", "thls", prepper.thls)

        (data,) = dask.optimize(data)

        logger.debug("Setting up boundary fields (KNMI source)")
        boundaries = harmonie_boundary.boundary_fields(
            config,
            self.module.openBCgrid,
            data,
            output_path=self.module.output_path,
        )
        logger.debug("Setting up initial fields (KNMI source)")
        initfields_ds = initfields.initial_fields(
            config,
            self.module.openBCgrid,
            data,
            transform,
            output_path=self.module.output_path,
        )

        boundaries = self._apply_noise(boundaries)
        boundaries, initfields_ds = dask.optimize(boundaries, initfields_ds)
        return boundaries, initfields_ds

    def _build_config(self) -> OpenBoundaryConfig:
        download_module = resolve_knmi_harmonie_download_module(self.module)
        ml_glob = self.module.nest_in_knmi.ml_glob
        sfc_glob = self.module.nest_in_knmi.sfc_glob
        use_grib = self.module.nest_in_knmi.use_grib
        if download_module is not None:
            if ml_glob in (None, ""):
                ml_glob = download_module.ml_glob
            if sfc_glob in (None, ""):
                sfc_glob = download_module.sfc_glob
            if use_grib is None and (
                getattr(download_module, "skip_cdo", False)
                or not getattr(download_module, "convert_to_netcdf", True)
            ):
                use_grib = True

        return OpenBoundaryConfig(
            e12=self.module.e12,
            tracernames=list(self.module.tracernames or []),
            tchunk=self.module.tchunk,
            start=self.module.start,
            author="author",
            time0=self.module.time0,
            end=self.module.end,
            KNMI_ml_glob=ml_glob,
            KNMI_sfc_glob=sfc_glob,
            w_from_continuity=self.module.nest_in_knmi.w_from_continuity,
            use_grib=use_grib,
        )

    def _resolve_noise_window(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        noise_std = self.module.nest_in_knmi.noise_std
        noise_minzt = self.module.nest_in_knmi.noise_minzt
        noise_maxzt = self.module.nest_in_knmi.noise_maxzt
        if noise_std is None or (noise_minzt is None and noise_maxzt is None):
            return noise_std, noise_minzt, noise_maxzt

        zt = self.module.openBCgrid.zt
        zm = self.module.openBCgrid.zm
        mask = np.ones_like(zt, dtype=bool)
        mask_zm = np.ones_like(zm, dtype=bool)
        if noise_minzt is not None:
            mask &= zt >= noise_minzt
            mask_zm &= zm >= noise_minzt
        if noise_maxzt is not None:
            mask &= zt <= noise_maxzt
            mask_zm &= zm <= noise_maxzt

        if (not np.any(mask)) or (not np.any(mask_zm)):
            logger.warning(
                "Noise std is set but no vertical levels are within the specified min/max zt bounds; skipping noise addition."
            )
            return None, noise_minzt, noise_maxzt

        logger.info(
            "Applying noise with std=%.3f to levels where zt is between %.2f and %.2f",
            noise_std,
            zt[mask][0],
            zt[mask][-1],
        )
        return noise_std, noise_minzt, noise_maxzt

    def _selected_noise_targets(self) -> Tuple[set[str], set[str]]:
        all_boundaries = {"west", "east", "south", "north", "top"}
        base_vars = {"u", "v", "w", "thl", "qt", "e12"}
        requested_bounds = self.module.nest_in_knmi.noise_boundaries
        requested_vars = self.module.nest_in_knmi.noise_variables

        active_bounds = (
            set(requested_bounds) & all_boundaries
            if requested_bounds is not None
            else all_boundaries
        )
        active_vars = (
            set(requested_vars) & base_vars if requested_vars is not None else base_vars
        )
        return active_bounds, active_vars

    def _apply_noise(self, boundaries: xr.Dataset) -> xr.Dataset:
        noise_std, noise_minzt, noise_maxzt = self._resolve_noise_window()
        if noise_std is None or noise_std <= 0.0:
            return boundaries

        rng = np.random.default_rng(self.module.nest_in_knmi.noise_seed)
        active_bounds, active_vars = self._selected_noise_targets()

        for var in active_vars:
            for bnd in active_bounds:
                name = f"{var}{bnd}"
                if name not in boundaries:
                    continue
                arr = boundaries[name]
                mask_3d = xr.ones_like(arr, dtype=bool)
                if "zt" in arr.dims:
                    if noise_minzt is not None:
                        mask_3d = mask_3d.where(
                            boundaries.zt >= noise_minzt, other=False
                        )
                    if noise_maxzt is not None:
                        mask_3d = mask_3d.where(
                            boundaries.zt <= noise_maxzt, other=False
                        )
                if "zm" in arr.dims:
                    if noise_minzt is not None:
                        mask_3d = mask_3d.where(
                            boundaries.zm >= noise_minzt, other=False
                        )
                    if noise_maxzt is not None:
                        mask_3d = mask_3d.where(
                            boundaries.zm <= noise_maxzt, other=False
                        )
                noise = rng.normal(loc=0.0, scale=noise_std, size=arr.shape)
                boundaries[name] = arr + (mask_3d * noise)

        return boundaries
