import numpy as np
import xarray as xr

from modular_dales.Surface.LSM.LCZ.get_from_lcz import (
    LCZ_URBAN_NATURAL_TO_IFS,
    apply_urban_natural_lcz_overrides,
    lcz_dict,
    map_lcz_to_fields,
)


def test_map_lcz_to_fields_maps_values_for_valid_lcz_cells():
    lcz_da = xr.DataArray(
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float),
        dims=("y", "x"),
    )
    ifs_da = xr.DataArray(
        np.array([[20.0, 10.0], [20.0, 30.0]], dtype=float),
        dims=("y", "x"),
    )

    fields = map_lcz_to_fields(lcz_da, ifs_da, lcz_dict)
    f_bld = fields["building_surface_fraction"].values
    z0_urb = fields["roughness_length"].values

    assert np.isclose(f_bld[0, 0], lcz_dict[1].building_surface_fraction)
    assert np.isclose(f_bld[1, 0], lcz_dict[2].building_surface_fraction)
    assert np.isclose(f_bld[0, 1], lcz_dict[1].building_surface_fraction)
    assert np.isclose(f_bld[1, 1], lcz_dict[2].building_surface_fraction)

    assert np.isclose(z0_urb[0, 1], lcz_dict[1].roughness_length)
    assert np.isclose(z0_urb[1, 1], lcz_dict[2].roughness_length)


def test_apply_urban_natural_lcz_overrides_sets_lcz_to_10_for_urban_pixels():
    lcz_da = xr.DataArray(
        np.array([[11.0, 12.0], [10.0, 14.0]], dtype=float),
        dims=("y", "x"),
    )
    ifs_da = xr.DataArray(
        np.array([[20.0, 10.0], [20.0, 20.0]], dtype=float),
        dims=("y", "x"),
    )

    lcz_out, ifs_out = apply_urban_natural_lcz_overrides(
        lcz_da,
        ifs_da,
        urban_natural_lcz_to_10=True,
        urban_natural_lcz_to_natural_lsm=False,
    )

    assert np.array_equal(lcz_out.values, np.array([[10.0, 12.0], [10.0, 10.0]]))
    assert np.array_equal(ifs_out.values, ifs_da.values)


def test_apply_urban_natural_lcz_overrides_maps_ifs_for_urban_pixels():
    lcz_da = xr.DataArray(
        np.array([[11.0, 12.0], [16.0, 17.0]], dtype=float),
        dims=("y", "x"),
    )
    ifs_da = xr.DataArray(
        np.array([[20.0, 20.0], [20.0, 20.0]], dtype=float),
        dims=("y", "x"),
    )

    lcz_out, ifs_out = apply_urban_natural_lcz_overrides(
        lcz_da,
        ifs_da,
        urban_natural_lcz_to_10=False,
        urban_natural_lcz_to_natural_lsm=True,
    )

    expected = np.array(
        [
            [LCZ_URBAN_NATURAL_TO_IFS[11], LCZ_URBAN_NATURAL_TO_IFS[12]],
            [LCZ_URBAN_NATURAL_TO_IFS[16], LCZ_URBAN_NATURAL_TO_IFS[17]],
        ],
        dtype=float,
    )
    assert np.array_equal(lcz_out.values, lcz_da.values)
    assert np.array_equal(ifs_out.values, expected)
