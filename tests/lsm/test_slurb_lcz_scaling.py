from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from moda.Geometry.grids import GridDales
from moda.Geometry.masks import AllMask
from moda.Surface.LSM.lsm_output import LSM_output_dales
from moda.Surface.LSM.lsm_sources.slurb_modifications import (
    SlurbModification,
    apply_from_netcdf,
    apply_slurb_parameters_lcz,
    slbCreatorClass,
)
from moda.Surface.LSM.translation_tables.landuse_types import LandUseSet


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
    calls = {
        call.args[0]: call.args[1] for call in generator.apply_field.call_args_list
    }
    np.testing.assert_allclose(calls["f_bld"], 0.4)
    generator.apply_modification.assert_not_called()
    assert "using unscaled f_bld" in caplog.text


def test_netcdf_morphology_applies_only_supplied_fields():
    dataset = xr.Dataset({"f_bld": (("y", "x"), np.full((2, 2), 0.4))})
    generator = Mock(f_bld=np.zeros((2, 2)), z0_urb=np.full((2, 2), 0.2))
    apply_from_netcdf(dataset, generator)
    generator.apply_field.assert_called_once()
    assert generator.apply_field.call_args.args[0] == "f_bld"
    np.testing.assert_allclose(generator.apply_field.call_args.args[1], 0.4)
    np.testing.assert_allclose(generator.z0_urb, 0.2)
    generator.apply_modification.assert_not_called()


@pytest.fixture
def field_generator():
    return slbCreatorClass(
        GridDales(
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
    )


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("dtype", [float, int])
def test_apply_field_copies_values_and_preserves_storage(
    field_generator, layered, dtype
):
    shape = (3, 2, 2) if layered else (2, 2)
    values = np.full(shape, 2, dtype=dtype)
    field_generator.apply_field("test_field", values, dtype=dtype)
    stored = field_generator.test_field
    values.fill(7)
    np.testing.assert_array_equal(stored, 2)
    field_generator.apply_field("test_field", values, dtype=dtype)
    assert field_generator.test_field is stored
    np.testing.assert_array_equal(stored, 7)
    assert field_generator.vars["test_field"].dtype is dtype
    assert field_generator.vars["test_field"].n_layers == (3 if layered else None)


@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 3), (0, 2, 2), (1, 1, 2, 2)])
def test_apply_field_rejects_invalid_shapes_without_initializing(
    field_generator, shape
):
    with pytest.raises(ValueError, match="Incompatible SLURB field shape"):
        field_generator.apply_field("test_field", np.zeros(shape))
    assert "test_field" not in field_generator.vars


def test_apply_field_and_geometric_modification_share_storage(field_generator):
    field_generator.apply_field("h_bld", np.full((2, 2), 4.0))
    field_generator.apply_modification(
        SlurbModification(mask=AllMask(), varname="h_bld", value=12.0)
    )
    np.testing.assert_allclose(field_generator.h_bld, 12.0)
    field_generator.apply_field("h_bld", np.full((2, 2), 8.0))
    np.testing.assert_allclose(field_generator.h_bld, 8.0)


@pytest.mark.parametrize(
    "values, dtype",
    [(np.zeros((3, 2, 2)), float), (np.zeros((2, 2)), int)],
)
def test_apply_field_rejects_incompatible_existing_metadata(
    field_generator, values, dtype
):
    field_generator.apply_field("h_bld", np.full((2, 2), 4.0))
    with pytest.raises(ValueError, match="Incompatible dtype or layer count"):
        field_generator.apply_field("h_bld", values, dtype=dtype)
    np.testing.assert_allclose(field_generator.h_bld, 4.0)
