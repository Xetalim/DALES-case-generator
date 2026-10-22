from unittest.mock import Mock, call

import numpy as np
import pytest
import xarray as xr

from modular_dales.Geometry.GridDales import GridDales
from modular_dales.Surface.LSM import lsm_netcdf, lsm_slurb, lsm_sources
from modular_dales.Surface.LSM.LSM import FromLCZ, FromNetCDF, LSMModule
from modular_dales.Surface.LSM.lsm_data import (
    LAND_USE_PARAMETERS,
    LandUseFields,
    SoilFields,
)
from modular_dales.Surface.LSM.LSM_output_dales import LSM_output_dales
from modular_dales.Surface.LSM.translation_tables.landuse_types import (
    LandUseSet,
    get_land_use_definitions,
)


def test_typed_arrays_are_independent():
    soil = SoilFields.zeros((2, 2, 3))
    tile = LandUseFields.zeros((2, 3))
    tile.cover[:] = 0.5
    np.testing.assert_array_equal(tile.c_veg, np.zeros((2, 3)))
    assert soil.index_soil.dtype == np.int32
    assert soil.t_soil.dtype == np.float64


@pytest.mark.parametrize(
    "method, field",
    [("set_uniform_temperature", "t_soil"), ("set_uniform_moisture", "theta_soil")],
)
def test_soil_profiles_broadcast_without_replacing_arrays(method, field):
    soil = SoilFields.zeros((2, 2, 3))
    target = getattr(soil, field)
    getattr(soil, method)([0.2, 0.4])
    assert getattr(soil, field) is target
    np.testing.assert_allclose(
        target, np.broadcast_to(np.array([0.2, 0.4])[:, None, None], target.shape)
    )


@pytest.mark.parametrize("method", ["set_uniform_temperature", "set_uniform_moisture"])
@pytest.mark.parametrize("profile", [[1.0], [1.0, 2.0, 3.0], [[1.0], [2.0]]])
def test_soil_profiles_reject_wrong_shapes(method, profile):
    soil = SoilFields.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="Expected 2 soil values"):
        getattr(soil, method)(profile)
    np.testing.assert_array_equal(soil.t_soil, 0.0)
    np.testing.assert_array_equal(soil.theta_soil, 0.0)


@pytest.mark.parametrize(
    "method, field", [("set_temperature", "t_soil"), ("set_moisture", "theta_soil")]
)
def test_soil_arrays_copy_values_without_aliasing(method, field):
    soil = SoilFields.zeros((2, 2, 3))
    target = getattr(soil, field)
    source = np.full(target.shape, 0.3)
    getattr(soil, method)(source)
    source.fill(0.9)
    assert getattr(soil, field) is target
    np.testing.assert_allclose(target, 0.3)


@pytest.mark.parametrize("method", ["set_temperature", "set_moisture"])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3), (2, 3, 2)])
def test_soil_arrays_reject_broadcastable_and_transposed_shapes(method, shape):
    soil = SoilFields.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="Incompatible array shape"):
        getattr(soil, method)(np.ones(shape))
    np.testing.assert_array_equal(soil.t_soil, 0.0)
    np.testing.assert_array_equal(soil.theta_soil, 0.0)


@pytest.mark.parametrize(
    "method",
    [
        "iter_output_fields",
        "from_lcz",
        "_distribute_fraction",
        "apply_from_netcdf",
        "apply_slurb_parameters_lcz",
        "save_netcdf",
        "_set_soil_profile",
        "_copy_array",
        "set_uniform_soil_temperature",
        "set_uniform_soil_moisture",
        "set_soil_temperature_array",
        "set_soil_moisture_array",
    ],
)
def test_writer_has_no_source_serialization_or_soil_forwarders(method):
    assert not hasattr(LSM_output_dales, method)


@pytest.mark.parametrize("choice", list(LandUseSet))
def test_catalogues_have_typed_metadata_and_independent_mappings(choice):
    definitions = get_land_use_definitions(choice)
    assert definitions
    assert all(name == definition.lu_short for name, definition in definitions.items())
    assert sum(definition.bare_soil for definition in definitions.values()) == 1
    definitions.clear()
    assert get_land_use_definitions(choice)


def test_unknown_land_use_set_has_clear_error():
    with pytest.raises(ValueError, match="choose one of"):
        get_land_use_definitions("unknown")


@pytest.fixture
def grid():
    from modular_dales.modular.dales_simulation import dales_simulation

    sim = dales_simulation("test", {})
    grid = GridDales(
        itot=2,
        jtot=2,
        kmax=1,
        xsize=200.0,
        ysize=200.0,
        kmax_soil=4,
        xlat=52.0,
        xlon=5.0,
        x0=0.0,
        y0=0.0,
        alpha=1.0,
        dz0=10.0,
    )
    grid._attach_to_sim(sim)
    sim.grid = grid
    return grid


def test_depac_slurb_has_only_one_urban_tile(grid):
    writer = LSM_output_dales(grid, LandUseSet.DEPAC, include_slurb=True)
    assert "urb" in writer.land_use
    assert "slb" not in writer.land_use
    assert sum(state.definition.ifs_id == 20 for state in writer.land_use.values()) == 1
    writer.standard_fill_geometry_modification()
    np.testing.assert_allclose(writer.cover_tot, 1.0)
    assert "slb" not in get_land_use_definitions(LandUseSet.DEPAC)


def test_shared_ifs_classes_do_not_duplicate_cover(grid):
    writer = LSM_output_dales(grid, LandUseSet.DEPAC, include_slurb=True)
    lsm_sources.distribute_fraction(writer.land_use, np.ones(writer.shape), ifs_id=17)
    writer.init_lutypes_ifs()
    writer.recalculate_remaining_cover()
    np.testing.assert_allclose(writer.land_use_fields["fce"].cover, 0.5)
    np.testing.assert_allclose(writer.land_use_fields["fbd"].cover, 0.5)
    np.testing.assert_allclose(writer.cover_tot, 1.0)


@pytest.mark.parametrize("include_slurb", [False, True])
def test_from_lcz_supports_depac_with_slurb(grid, monkeypatch, include_slurb):
    monkeypatch.setattr(grid, "crs", "EPSG:28992")
    dataset = xr.Dataset(
        {
            "ifs_land_cover": (("y", "x"), np.array([[20, 17], [1, 99]])),
            "index_soil": (("y", "x"), np.full((2, 2), 3, dtype=np.int32)),
        }
    )
    loader = Mock(return_value=dataset)
    monkeypatch.setattr(
        "modular_dales.Surface.LSM.LSM.get_from_LCZ.do_everything",
        loader,
    )
    writer = LSM_output_dales(grid, LandUseSet.DEPAC, include_slurb=include_slurb)
    module = LSMModule()
    module._attach_to_sim(grid.sim)
    generator = Mock() if include_slurb else None
    module._init_slurb_generator = Mock(return_value=generator)
    apply_slurb = Mock()
    monkeypatch.setattr(lsm_slurb, "apply_slurb_parameters_lcz", apply_slurb)
    module._apply_lcz_source(writer, FromLCZ(urban_natural_lcz_to_10=True))
    loader.assert_called_once_with(
        grid, urban_natural_lcz_to_10=True, urban_natural_lcz_to_natural_lsm=False
    )
    assert writer.LCZ_ds is dataset
    urban = "urb"
    np.testing.assert_allclose(writer.land_use_fields[urban].cover, [[1, 0], [0, 0]])
    if include_slurb:
        assert "urb" not in writer.land_use
        assert apply_slurb.call_args.args[0] is dataset
        assert apply_slurb.call_args.args[1] is generator
        assert apply_slurb.call_args.args[2] is writer.land_use_fields[urban].cover
    else:
        apply_slurb.assert_not_called()
    np.testing.assert_allclose(writer.land_use_fields["fce"].cover, [[0, 0.5], [0, 0]])
    np.testing.assert_allclose(writer.land_use_fields["fbd"].cover, [[0, 0.5], [0, 0]])
    np.testing.assert_allclose(writer.land_use_fields["grs"].cover, [[0, 0], [1, 0]])
    np.testing.assert_allclose(writer.land_use_fields["brn"].cover, [[0, 0], [0, 1]])
    np.testing.assert_allclose(writer.cover_tot, 1.0)
    np.testing.assert_array_equal(writer.soil.index_soil, np.full((4, 2, 2), 3))


def test_lcz_source_requires_projection_before_loading(grid, monkeypatch):
    monkeypatch.setattr(grid, "crs", None)
    loader = Mock()
    monkeypatch.setattr(
        "modular_dales.Surface.LSM.LSM.get_from_LCZ.do_everything", loader
    )
    module = LSMModule()
    module._attach_to_sim(grid.sim)
    with pytest.raises(ValueError, match="valid projection"):
        module._apply_lcz_source(LSM_output_dales(grid), FromLCZ())
    loader.assert_not_called()


@pytest.mark.parametrize("include_slurb", [False, True])
def test_netcdf_source_finalizes_depac_and_optional_slurb(
    grid, monkeypatch, include_slurb
):
    dataset = xr.Dataset(
        {
            "index_soil": (("y", "x"), np.full((2, 2), 3)),
            "ifs_land_cover": (("y", "x"), np.full((2, 2), 17)),
            "frac_water": (("y", "x"), np.full((2, 2), 0.1)),
            "frac_sea": (("y", "x"), np.full((2, 2), 0.1)),
            "frac_town": (("y", "x"), np.full((2, 2), 0.3)),
            "frac_nature": (("y", "x"), np.full((2, 2), 0.5)),
        }
    )
    loader = Mock(return_value=dataset)
    monkeypatch.setattr(
        "modular_dales.Surface.LSM.LCZ.from_netcdf.load_from_netcdf", loader
    )
    apply_slurb = Mock()
    monkeypatch.setattr(lsm_slurb, "apply_from_netcdf", apply_slurb)
    module = LSMModule()
    module._attach_to_sim(grid.sim)
    generator = Mock() if include_slurb else None
    module._init_slurb_generator = Mock(return_value=generator)
    writer = LSM_output_dales(grid, LandUseSet.DEPAC, include_slurb=include_slurb)
    module._apply_netcdf_source(writer, FromNetCDF(path="surface.nc"))
    loader.assert_called_once_with("surface.nc", grid, esa_cache_dir=None)
    urban = "urb"
    np.testing.assert_allclose(writer.land_use_fields[urban].cover, 0.3)
    np.testing.assert_allclose(writer.land_use_fields["aqu"].cover, 0.2)
    np.testing.assert_allclose(writer.land_use_fields["fce"].cover, 0.25)
    np.testing.assert_allclose(writer.land_use_fields["fbd"].cover, 0.25)
    np.testing.assert_allclose(writer.cover_tot, 1.0)
    np.testing.assert_array_equal(writer.soil.index_soil, 3)
    if include_slurb:
        assert apply_slurb.call_args.args[0] is dataset
        assert apply_slurb.call_args.args[1] is generator
    else:
        apply_slurb.assert_not_called()


def test_debug_initializes_arrays_without_serialization(grid, monkeypatch):
    monkeypatch.setattr(
        lsm_netcdf,
        "iter_output_fields",
        Mock(side_effect=AssertionError("serialization used")),
    )
    writer = LSM_output_dales(grid, debug=True)
    assert np.isnan(writer.soil.t_soil).all()
    assert np.isnan(writer.soil.theta_soil).all()
    assert np.isnan(writer.cover_tot).all()
    assert np.isnan(writer.c_veg_tot).all()
    np.testing.assert_array_equal(writer.soil.index_soil, 0)
    for tile in writer.land_use_fields.values():
        for parameter in LAND_USE_PARAMETERS:
            assert np.isnan(tile.get(parameter)).all()


def test_preparation_preserves_stage_order():
    from modular_dales.modular.dales_simulation import dales_simulation

    module = LSMModule()
    module._attach_to_sim(dales_simulation("test", {}))
    stages = Mock()
    writer = Mock(nlu=3)
    module.exists_soil_temp_moisture_skin_temp = Mock()
    module.module_exists = Mock(return_value=False)
    module._create_lsm_writer = Mock(return_value=writer)
    stages.attach_mock(writer.standard_fill_geometry_modification, "initialize")
    for name in (
        "_apply_land_use_source",
        "_apply_top10_overlay",
        "_apply_bofek_overlay",
        "_apply_thermal_moisture",
        "_apply_ls2d_override",
    ):
        helper = Mock()
        setattr(module, name, helper)
        stages.attach_mock(helper, name)
    stages.attach_mock(writer.trim_landuse, "trim")
    module.prepare_calculations()
    assert stages.mock_calls == [
        call.initialize(),
        call._apply_land_use_source(writer),
        call._apply_top10_overlay(writer),
        call._apply_bofek_overlay(writer),
        call._apply_thermal_moisture(writer),
        call._apply_ls2d_override(writer),
        call.trim(),
    ]
    assert module.nlu == 4


def test_netcdf_output_uses_typed_fields(grid, tmp_path):
    writer = LSM_output_dales(grid, LandUseSet.DEPAC, include_slurb=True)
    writer.standard_fill_geometry_modification()
    writer.land_use["urb"].lu_frac[:] = 0.3
    writer.init_lutypes_ifs()
    writer.recalculate_remaining_cover()
    writer.trim_landuse()
    lsm_netcdf.save_netcdf(writer, tmp_path, 1)
    with xr.open_dataset(tmp_path / "lsm.inp_001.nc") as dataset:
        assert dataset.sizes["nlu"] == 2
        assert "cover_urb" not in dataset
        np.testing.assert_allclose(dataset["cover_slb"], 0.3)
        np.testing.assert_allclose(dataset["cover_tot"], 1.0)
        assert dataset["t_soil"].dims == ("z", "y", "x")
        np.testing.assert_array_equal(
            dataset["lslurb"].values,
            [
                b"T" if state.definition.lslurb else b"F"
                for state in writer.land_use.values()
            ],
        )
