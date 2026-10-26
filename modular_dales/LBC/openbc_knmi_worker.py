"""KNMI-backed open-boundary module."""

import logging
from dataclasses import dataclass, field

import dask
import numpy as np
import xarray as xr

from modular_dales.LBC.nest_dales_in_HARMONIE import (
    boundary as harmonie_boundary,
)
from modular_dales.LBC.nest_dales_in_HARMONIE import (
    initfields,
    nest_dales_in_KNMI,
)
from modular_dales.LBC.nest_dales_in_HARMONIE.knmi_harmonie_download import (
    resolve_knmi_harmonie_download_module,
)
from modular_dales.LBC.openbc import do_openboundary
from modular_dales.LBC.openboundary_config import OpenBoundaryConfig
from modular_dales.module_registry import register_module

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class _KNMIOpenBCOptions:
    """Options for nesting DALES inside KNMI HARMONIE output."""

    ml_glob: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    sfc_glob: str | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    w_from_continuity: bool = field(
        default=False, repr=True, metadata={"serialize": True}, init=True
    )
    use_grib: bool | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_std: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_seed: int | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_boundaries: list[str] | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_variables: list[str] | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_minzt: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )
    noise_maxzt: float | None = field(
        default=None, repr=True, metadata={"serialize": True}, init=True
    )


@register_module
@dataclass
class Nest_in_KNMI(do_openboundary, _KNMIOpenBCOptions):
    module_name: str = field(default="Nest_in_KNMI", init=False)

    def prepare_calculation(self) -> None:
        self._prepare_from_knmi()

    def _prepare_from_knmi(self) -> None:
        config = self._build_config()
        prepper = nest_dales_in_KNMI.KNMIPrepper(config, self.openBCgrid)
        prepper.load_data()
        data, transform = prepper.prep_harmonie()
        self.harmonieprepper = prepper

        logger.info("Setting namelist NAMSURFACE:ps to %f", prepper.ps)
        self.set_nml_section("NAMSURFACE", "ps", prepper.ps)
        logger.info("Setting namelist NAMSURFACE:thls to %f", prepper.thls)
        self.set_nml_section("NAMSURFACE", "thls", prepper.thls)

        (data,) = dask.base.optimize(data)

        logger.debug("Setting up boundary fields (KNMI source)")
        boundaries = harmonie_boundary.boundary_fields(config, self.openBCgrid, data)
        logger.debug("Setting up initial fields (KNMI source)")
        initfields_ds = initfields.initial_fields(
            config,
            self.openBCgrid,
            data,
            transform,
            output_path=self.output_path.as_posix(),
        )

        boundaries = self._apply_noise(boundaries)
        self.boundaries, self.initfields = dask.base.optimize(boundaries, initfields_ds)

    def _build_config(self) -> OpenBoundaryConfig:
        download_module = resolve_knmi_harmonie_download_module(self.sim)
        ml_glob = self.ml_glob
        sfc_glob = self.sfc_glob
        use_grib = self.use_grib
        if download_module is not None:
            if ml_glob in (None, ""):
                ml_glob = download_module.ml_glob
            if sfc_glob in (None, ""):
                sfc_glob = download_module.sfc_glob
            if use_grib is None and (
                getattr(download_module, "skip_cdo", False)
                or not getattr(download_module, "convert_to_netcdf", False)
            ):
                use_grib = True

        return OpenBoundaryConfig(
            e12=self.e12,
            tracernames=list(self.tracernames or []),
            tchunk=self.tchunk,
            start=self.start,
            author="author",
            time0=self.time0,
            end=self.end,
            HARMONIE_ml_glob=ml_glob,
            HARMONIE_sfc_glob=sfc_glob,
            w_from_continuity=self.w_from_continuity,
            use_grib=use_grib,
        )

    def _resolve_noise_window(
        self,
    ) -> tuple[float | None, float | None, float | None]:
        noise_std = self.noise_std
        noise_minzt = self.noise_minzt
        noise_maxzt = self.noise_maxzt
        if noise_std is None or (noise_minzt is None and noise_maxzt is None):
            return noise_std, noise_minzt, noise_maxzt

        zt = self.openBCgrid.zt
        zm = self.openBCgrid.zm
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

    def _selected_noise_targets(self) -> tuple[set[str], set[str]]:
        all_boundaries = {"west", "east", "south", "north", "top"}
        base_vars = {"u", "v", "w", "thl", "qt", "e12"}
        active_bounds = (
            set(self.noise_boundaries) & all_boundaries
            if self.noise_boundaries is not None
            else all_boundaries
        )
        active_vars = (
            set(self.noise_variables) & base_vars
            if self.noise_variables is not None
            else base_vars
        )
        return active_bounds, active_vars

    def _apply_noise(self, boundaries: xr.Dataset) -> xr.Dataset:
        noise_std, noise_minzt, noise_maxzt = self._resolve_noise_window()
        if noise_std is None or noise_std <= 0.0:
            return boundaries

        rng = np.random.default_rng(self.noise_seed)
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
