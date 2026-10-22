from dataclasses import fields, replace

import numpy as np
import pytest

from modular_dales.Geometry.GridDales import GridDales, GridDalesOpenBC


@pytest.fixture
def grid() -> GridDales:
    return GridDales(
        itot=4,
        jtot=3,
        kmax=2,
        xsize=400.0,
        ysize=300.0,
        kmax_soil=4,
        xlat=52.0,
        xlon=5.0,
        dz0=10.0,
        x0=100.0,
        y0=200.0,
        alpha=1.1,
        proj4="EPSG:28992",
    )


def test_conversion_uses_current_dataclass_inputs(grid: GridDales) -> None:
    grid.x0 = 500.0
    grid.ysize = 600.0
    grid.proj4 = None
    grid.wkt = "EPSG:4326"
    openbc_grid = grid.as_openbc()

    assert isinstance(openbc_grid, GridDalesOpenBC)
    for grid_field in fields(GridDales):
        if grid_field.init:
            assert getattr(openbc_grid, grid_field.name) == getattr(
                grid, grid_field.name
            )
    assert openbc_grid.crs == grid.wkt
    np.testing.assert_array_equal(openbc_grid.xm, [500.0, 600.0, 700.0, 800.0, 900.0])
    np.testing.assert_array_equal(openbc_grid.ym, [200.0, 400.0, 600.0, 800.0])
    np.testing.assert_array_equal(openbc_grid.xt, [550.0, 650.0, 750.0, 850.0])
    np.testing.assert_array_equal(openbc_grid.yt, [300.0, 500.0, 700.0])


def test_conversion_preserves_custom_vertical_grid(grid: GridDales) -> None:
    grid.zm = np.array([0.0, 12.0, 30.0])
    grid.zt = np.array([6.0, 21.0])
    grid.dz = np.diff(grid.zm)
    grid.zsize = 30.0

    openbc_grid = GridDalesOpenBC.from_grid(grid)

    assert openbc_grid.zt is grid.zt
    assert openbc_grid.zm is grid.zm
    assert openbc_grid.dz is grid.dz
    assert openbc_grid.zsize == grid.zsize


def test_conversion_does_not_copy_module_lifecycle(grid: GridDales) -> None:
    grid.do_config_done = True
    grid.prepare_calculation_done = True

    openbc_grid = grid.as_openbc()

    assert not openbc_grid.do_config_done
    assert not openbc_grid.prepare_calculation_done
    with pytest.raises(RuntimeError, match="detached"):
        _ = openbc_grid.sim


def test_openbc_dataclass_replace_reinitializes_coordinates(grid: GridDales) -> None:
    openbc_grid = replace(grid.as_openbc(), itot=2, jtot=2)

    assert isinstance(openbc_grid, GridDalesOpenBC)
    assert openbc_grid.dx == 200.0
    assert openbc_grid.dy == 150.0
    np.testing.assert_array_equal(openbc_grid.xm, [100.0, 300.0, 500.0])
    np.testing.assert_array_equal(openbc_grid.ym, [200.0, 350.0, 500.0])
    np.testing.assert_array_equal(openbc_grid.xt, [200.0, 400.0])
    np.testing.assert_array_equal(openbc_grid.yt, [275.0, 425.0])
