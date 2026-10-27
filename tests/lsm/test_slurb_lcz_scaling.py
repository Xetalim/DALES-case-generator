from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from MoDa.Geometry.grids import GridDales
from MoDa.Surface.LSM.lsm_output import LSM_output_dales
from MoDa.Surface.LSM.lsm_slurb import (
    apply_from_netcdf,
    apply_slurb_parameters_lcz,
)
from MoDa.Surface.LSM.SLuRB.slurb import slbCreatorClass
from MoDa.Surface.LSM.translation_tables.landuse_types import LandUseSet


@pytest.mark.parametrize("include_slurb", [False, True])
def test_apply_slurb_parameters_lcz_scales_f_bld_with_urban_cover(include_slurb):
    grid = GridDales(
        itot=2,
        jtot=2,
        kmax=1,
        xsize=200.0,
        ysize=200.0,
        kmax_soil=1,
        xlat=52.0,
        xlon=5.0,
        x0=0.0,
        y0=0.0,
        alpha=1.0,
        dz0=10.0,
    )

    lsm = LSM_output_dales(
        grid=grid,
        lu_types=LandUseSet.DEPAC,
        soil_levels=1,
        include_slurb=include_slurb,
    )
    urban_tile = "urb"
    lsm.land_use_fields[urban_tile].cover[:, :] = np.array([[0.0, 0.2], [0.5, 1.0]])

    lsm.LCZ_ds = xr.Dataset(
        {
            "building_surface_fraction": (
                ("y", "x"),
                np.array([[0.8, 0.5], [0.4, 0.3]]),
            ),
            "aspect_ratio": (("y", "x"), np.array([[1.0, 1.0], [1.0, 1.0]])),
            "height_roughness_element": (
                ("y", "x"),
                np.array([[10.0, 10.0], [10.0, 10.0]]),
            ),
            "roughness_length": (("y", "x"), np.array([[0.3, 0.4], [0.5, 0.6]])),
        }
    )

    slb_generator = slbCreatorClass(grid)
    apply_slurb_parameters_lcz(
        lsm.LCZ_ds, slb_generator, lsm.land_use_fields[urban_tile].cover
    )

    expected_f_bld = np.array([[0.0, 0.1], [0.2, 0.3]])
    assert np.allclose(slb_generator.f_bld, expected_f_bld)
    assert np.allclose(slb_generator.z0_urb, lsm.LCZ_ds["roughness_length"].values)


def test_lcz_without_urban_cover_preserves_unscaled_building_fraction(caplog):
    dataset = xr.Dataset(
        {
            name: (("y", "x"), np.full((2, 2), 0.4))
            for name in (
                "building_surface_fraction",
                "aspect_ratio",
                "height_roughness_element",
                "roughness_length",
            )
        }
    )
    generator = Mock(
        f_bld=np.zeros((2, 2)),
        hw_can=np.zeros((2, 2)),
        h_bld=np.zeros((2, 2)),
        z0_urb=np.zeros((2, 2)),
    )
    apply_slurb_parameters_lcz(dataset, generator, None)
    np.testing.assert_allclose(generator.f_bld, 0.4)
    assert "using unscaled f_bld" in caplog.text


def test_netcdf_morphology_applies_only_supplied_fields():
    dataset = xr.Dataset({"f_bld": (("y", "x"), np.full((2, 2), 0.4))})
    generator = Mock(f_bld=np.zeros((2, 2)), z0_urb=np.full((2, 2), 0.2))
    apply_from_netcdf(dataset, generator)
    np.testing.assert_allclose(generator.f_bld, 0.4)
    np.testing.assert_allclose(generator.z0_urb, 0.2)
    generator.apply_modification.assert_called_once()
