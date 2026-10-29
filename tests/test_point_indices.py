import pytest

from moda.Configuration.output_modules import prepare_point_indices


@pytest.mark.parametrize("allow_idx_list", [True, False])
@pytest.mark.parametrize("z_idx", [None, [5, 6]])
def test_point_indices_preserve_list_mode_and_optional_z(allow_idx_list, z_idx):
    result = prepare_point_indices(
        "test",
        None,
        allow_idx_list=allow_idx_list,
        x_idx=[1, 2],
        y_idx=[3, 4],
        z_idx=z_idx,
    )
    if allow_idx_list:
        assert result == ([1, 2], [3, 4], z_idx)
    else:
        assert result == (1, 3, 5 if z_idx is not None else None)


@pytest.mark.parametrize("allow_idx_list", [True, False])
def test_point_indices_validate_vertical_lengths(allow_idx_list):
    with pytest.raises(ValueError, match="matching lengths"):
        prepare_point_indices(
            "test",
            None,
            allow_idx_list=allow_idx_list,
            x_idx=[1, 2],
            y_idx=[3, 4],
            z_idx=[5],
        )


def test_point_indices_require_grid_for_coordinates():
    with pytest.raises(RuntimeError, match="attached GridDales"):
        prepare_point_indices("test", None, x=1.0, y=2.0)
