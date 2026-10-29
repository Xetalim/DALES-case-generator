from unittest.mock import Mock, call

import numpy as np
import pytest
import xarray as xr

import moda.Surface.LSM.lsm_output
from moda.Atmosphere.ls2d_atmosphere import FromLS2D
from moda.Geometry.grids import GridDales
from moda.Geometry.masks import AllMask, ExplicitArrayMask
from moda.Surface.LSM import lsm_output
from moda.Surface.LSM.lsm_data import (
    LAND_USE_PARAMETERS,
    LandUseFields,
    SoilFields,
)
from moda.Surface.LSM.lsm_module import LSMModule
from moda.Surface.LSM.lsm_output import LSM_output_dales
from moda.Surface.LSM.lsm_sources import lsm_sources
from moda.Surface.LSM.lsm_sources import slurb_modifications as lsm_slurb
from moda.Surface.LSM.lsm_sources.lsm_sources import (
    FromBofek,
    FromLCZ,
    FromNetCDF,
    FromTop10,
    SlurbModification,
    SlurbModifications,
)
from moda.Surface.LSM.lsm_sources.lu_modifications import (
    LandUseModification,
    LandUseModifications,
)
from moda.Surface.LSM.lsm_sources.modular_temps_moisture import (
    SoilTemperatureMoistureFromHarmonie,
    UniformSkinTemperature,
    UniformSoilMoisture,
    UniformSoilTemperature,
    VaryingSkinTemperature,
    VaryingSoilMoisture,
    VaryingSoilTemperature,
)
from moda.Surface.LSM.lsm_sources.top10_bofek_ags import apply_top10_to_lsm_writer
from moda.Surface.LSM.translation_tables.landuse_types import (
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


def test_lsm_requires_explicit_land_use_set():
    with pytest.raises(TypeError, match="land_use_set"):
        LSMModule()


def test_lsm_rejects_unsupported_land_use_source_before_loading(grid):
    module = LSMModule(land_use_set=LandUseSet.BASIC)
    module.add(FromLCZ(), priority=100)
    with pytest.raises(
        ValueError, match="FromLCZ does not support land-use set 'basic'"
    ):
        module._apply_sources(LSM_output_dales(grid, LandUseSet.BASIC))


@pytest.mark.parametrize(
    "source",
    [
        FromLCZ(),
        FromNetCDF(path="surface.nc"),
        FromTop10(),
        FromBofek(),
        FromLS2D(),
        UniformSkinTemperature(290.0),
        VaryingSkinTemperature(np.full((2, 2), 290.0)),
        UniformSoilTemperature([289.0] * 4),
        VaryingSoilTemperature(np.full((4, 2, 2), 289.0)),
        UniformSoilMoisture([0.2] * 4),
        VaryingSoilMoisture(np.full((4, 2, 2), 0.2)),
        SoilTemperatureMoistureFromHarmonie(
            "soil.nc", [0.1, 0.2, 0.3, 0.4], "2023-01-01T12:00:00"
        ),
        LandUseModification(mask=AllMask(), type="grs"),
        LandUseModifications([LandUseModification(mask=AllMask(), type="grs")]),
        SlurbModification(mask=AllMask(), varname="h_bld", value=12.0),
        SlurbModifications(
            [SlurbModification(mask=AllMask(), varname="h_bld", value=12.0)]
        ),
    ],
)
@pytest.mark.parametrize("as_list", [False, True])
def test_lsm_sources_require_explicit_priority(source, as_list):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    with pytest.raises(TypeError, match="required keyword-only argument: 'priority'"):
        module.add([source] if as_list else source)
    assert module.sources == []
    module.add([source] if as_list else source, priority=100)
    assert len(module.sources) == 1
    assert module.sources[0].source is source
    assert module.sources[0].priority == 100


def test_lsm_rejects_raw_source_in_constructor_before_application(grid):
    module = LSMModule(land_use_set=LandUseSet.DEPAC, sources=[FromLCZ()])
    with pytest.raises(TypeError, match=r"Use LSMModule\.add"):
        module._apply_sources(LSM_output_dales(grid))


@pytest.mark.parametrize("source", [object(), 290.0, (UniformSkinTemperature(290.0),)])
@pytest.mark.parametrize("as_list", [False, True])
def test_lsm_add_rejects_unsupported_sources(source, as_list):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    with pytest.raises(TypeError, match="Unsupported LSM source type"):
        module.add([source] if as_list else source, priority=100)
    assert module.sources == []


@pytest.mark.parametrize("priority", [None, 1.5, "100"])
@pytest.mark.parametrize("as_list", [False, True])
def test_lsm_add_rejects_non_integer_priority_without_mutation(priority, as_list):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(FromLCZ(), priority=100)
    original_entries = module.sources.copy()
    source = UniformSkinTemperature(290.0)
    with pytest.raises(TypeError, match="priority must be an integer"):
        module.add([source, FromLCZ()] if as_list else source, priority=priority)
    assert module.sources == original_entries
    assert module.sources[0] is original_entries[0]


@pytest.mark.parametrize("priority", [-100, 0, 100])
def test_lsm_add_accepts_integer_priorities(priority):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    source = UniformSkinTemperature(290.0)

    module.add(source, priority=priority)

    assert module.sources[0].source is source
    assert module.sources[0].priority == priority


def test_lsm_add_requires_keyword_only_priority():
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    with pytest.raises(TypeError):
        module.add(FromLCZ(), 100)
    assert module.sources == []


@pytest.mark.parametrize("invalid_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_source", [object(), [UniformSkinTemperature(290.0)]])
def test_lsm_add_validates_batches_atomically(invalid_index, invalid_source):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(FromLCZ(), priority=100)
    original_entries = module.sources.copy()
    sources = [UniformSkinTemperature(290.0), FromLCZ()]
    sources.insert(invalid_index, invalid_source)

    with pytest.raises(TypeError, match="Unsupported LSM source type"):
        module.add(sources, priority=200)

    assert module.sources == original_entries
    assert module.sources[0] is original_entries[0]


def test_lsm_configuration_uses_add_interface():
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    modification = LandUseModification(mask=AllMask(), type="grs")
    source = FromLCZ()

    module.add(modification, priority=200)
    module.add(source, priority=100)

    assert not hasattr(module, "__add__")
    assert not hasattr(module, "__iadd__")
    assert not hasattr(module, "land_use_modifications")
    assert not hasattr(module, "profiles")
    assert [entry.source for entry in module.sources] == [modification, source]
    assert [entry.priority for entry in module.sources] == [200, 100]


def test_lsm_sources_apply_in_priority_order():
    module = LSMModule(land_use_set=LandUseSet.IFS)
    module.add(FromBofek(), priority=40)
    module.add(FromTop10(), priority=30)
    module.add(FromNetCDF(path="surface.nc"), priority=20)
    module.add(FromLCZ(), priority=10)
    applied = []
    module._apply_lcz_source = Mock(side_effect=lambda *_: applied.append("lcz"))
    module._apply_netcdf_source = Mock(side_effect=lambda *_: applied.append("netcdf"))
    module._apply_top10_overlay = Mock(side_effect=lambda *_: applied.append("top10"))
    module._apply_bofek_overlay = Mock(side_effect=lambda *_: applied.append("bofek"))

    module._apply_sources(Mock())

    assert applied == ["lcz", "netcdf", "top10", "bofek"]


@pytest.mark.parametrize("missing", [np.nan, 0, -9999, 999])
@pytest.mark.parametrize("remap_lcz", [False, True])
def test_top10_overlay_preserves_fractions_and_lcz_outside_coverage(
    grid, tmp_path, missing, remap_lcz
):
    writer = LSM_output_dales(grid, LandUseSet.DEPAC)
    writer.land_use["grs"].lu_frac[:] = 0.4
    writer.land_use["urb"].lu_frac[:] = 0.6
    writer.init_lutypes_ifs()
    writer.recalculate_remaining_cover()
    writer.LCZ_ds = xr.Dataset({"lcz": (("y", "x"), np.full((2, 2), 11))})
    path = tmp_path / "top10.nc"
    xr.Dataset(
        {"land_use": (("y", "x"), [[10.0, missing], [missing, 20.0]])},
        coords={"x": grid.xt, "y": grid.yt},
    ).to_netcdf(path)

    apply_top10_to_lsm_writer(
        writer, path, fill_north_sea=False, urban_natural_lcz_to_10=remap_lcz
    )

    np.testing.assert_allclose(writer.land_use["grs"].lu_frac, [[1.0, 0.4], [0.4, 0.0]])
    np.testing.assert_allclose(writer.land_use["urb"].lu_frac, [[0.0, 0.6], [0.6, 1.0]])
    np.testing.assert_allclose(writer.cover_tot, 1.0)
    np.testing.assert_array_equal(
        writer.LCZ_ds["lcz"], [[11, 11], [11, 10 if remap_lcz else 11]]
    )


def test_top10_overlay_preserves_fractions_outside_raster_extent(grid, tmp_path):
    writer = LSM_output_dales(grid, LandUseSet.DEPAC)
    writer.land_use["urb"].lu_frac[:] = 1.0
    writer.init_lutypes_ifs()
    path = tmp_path / "top10.nc"
    xr.Dataset(
        {"land_use": (("y", "x"), np.full((2, 2), 10.0))},
        coords={
            "x": [grid.xt[0] - 1.0, grid.xt[0] + 1.0],
            "y": [grid.yt[0] - 1.0, grid.yt[0] + 1.0],
        },
    ).to_netcdf(path)

    apply_top10_to_lsm_writer(writer, path, fill_north_sea=False)

    np.testing.assert_allclose(writer.land_use["grs"].lu_frac, [[1, 0], [0, 0]])
    np.testing.assert_allclose(writer.land_use["urb"].lu_frac, [[0, 1], [1, 1]])
    np.testing.assert_allclose(writer.cover_tot, 1.0)


def test_lsm_adds_thermal_sources_without_reordering():
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    skin = UniformSkinTemperature(290.0)
    temperature = UniformSoilTemperature([289.0, 288.0])
    moisture = UniformSoilMoisture([0.2, 0.3])
    replacement = UniformSoilTemperature([280.0, 279.0])

    module.add(skin, priority=40)
    module.add([temperature, moisture], priority=10)
    module.add(replacement, priority=20)

    assert [entry.source for entry in module.sources] == [
        skin,
        temperature,
        moisture,
        replacement,
    ]
    assert all(
        entry.source is source
        for entry, source in zip(
            module.sources, [skin, temperature, moisture, replacement]
        )
    )
    assert [entry.priority for entry in module.sources] == [40, 10, 10, 20]


@pytest.fixture
def grid():
    from moda.modular.dales_simulation import dales_simulation

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


@pytest.fixture
def lcz_loader(grid, monkeypatch):
    monkeypatch.setattr(grid, "crs", "EPSG:28992")
    dataset = xr.Dataset(
        {
            "ifs_land_cover": (("y", "x"), np.full((2, 2), 20)),
            "index_soil": (("y", "x"), np.full((2, 2), 3, dtype=np.int32)),
        }
    )
    loader = Mock(return_value=dataset)
    monkeypatch.setattr("moda.Surface.LSM.LCZ.get_from_lcz.do_everything", loader)
    return loader


@pytest.mark.parametrize("skin_priority, before_lcz", [(200, 273.15), (0, 295.0)])
def test_skin_temperature_priority_relative_to_lcz(
    grid, lcz_loader, skin_priority, before_lcz
):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(UniformSkinTemperature(295.0), priority=skin_priority)
    module.add(FromLCZ(), priority=100)
    module._attach_to_sim(grid.sim)
    dataset = lcz_loader.return_value

    def inspect_before_lcz(*args, **kwargs):
        writer = module.lsm_writer
        assert isinstance(writer, LSM_output_dales)
        for tile in writer.land_use_fields.values():
            np.testing.assert_allclose(tile.tskin, before_lcz)
        return dataset

    lcz_loader.side_effect = inspect_before_lcz

    module.prepare_calculation()

    lcz_loader.assert_called_once()
    writer = module.lsm_writer
    assert isinstance(writer, LSM_output_dales)
    assert set(writer.land_use_fields) == {"urb"}
    np.testing.assert_allclose(writer.land_use_fields["urb"].tskin, 295.0)
    np.testing.assert_allclose(writer.cover_tot, 1.0)


@pytest.mark.parametrize("priority", [0, 200])
@pytest.mark.parametrize("collection", [False, True])
def test_land_use_modifications_run_on_both_sides_of_lcz(
    grid, lcz_loader, priority, collection
):
    modification = LandUseModification(mask=AllMask(), type="grs")
    source = LandUseModifications([modification]) if collection else modification
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(source, priority=priority)
    module.add(FromLCZ(), priority=100)
    module._attach_to_sim(grid.sim)
    writer = LSM_output_dales(grid)
    writer.standard_fill_geometry_modification()
    dataset = lcz_loader.return_value

    def inspect_before_lcz(*args, **kwargs):
        np.testing.assert_allclose(
            writer.land_use_fields["grs"].cover, 1.0 if priority < 100 else 0.0
        )
        return dataset

    lcz_loader.side_effect = inspect_before_lcz

    module._apply_sources(writer)

    lcz_loader.assert_called_once()
    np.testing.assert_allclose(
        writer.land_use_fields["grs"].cover, 1.0 if priority > 100 else 0.0
    )
    np.testing.assert_allclose(
        writer.land_use_fields["urb"].cover, 0.0 if priority > 100 else 1.0
    )
    np.testing.assert_allclose(writer.cover_tot, 1.0)


@pytest.mark.parametrize("priority", [0, 200])
@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("dataset_source", ["lcz", "netcdf"])
def test_slurb_modification_priority_relative_to_dataset(
    grid,
    lcz_loader,
    monkeypatch,
    priority,
    collection,
    dataset_source,
):
    dataset = lcz_loader.return_value
    for name in (
        "building_surface_fraction",
        "aspect_ratio",
        "height_roughness_element",
        "roughness_length",
    ):
        dataset[name] = (("y", "x"), np.full((2, 2), 0.4))
    for name, value in (
        ("frac_water", 0.0),
        ("frac_sea", 0.0),
        ("frac_nature", 0.0),
        ("frac_town", 1.0),
        ("h_bld", 0.4),
    ):
        dataset[name] = (("y", "x"), np.full((2, 2), value))
    monkeypatch.setattr(
        "moda.Surface.LSM.LCZ.from_netcdf.load_from_netcdf",
        Mock(return_value=dataset),
    )
    mask = [[True, False], [False, True]]
    modification = SlurbModification(
        mask=ExplicitArrayMask(mask), varname="h_bld", value=12.0
    )
    source = SlurbModifications([modification]) if collection else modification
    module = LSMModule(land_use_set=LandUseSet.DEPAC, include_slurb=True)
    # Insert the modification first even when it should apply after the dataset.
    module.add(source, priority=priority)
    module.add(
        FromLCZ() if dataset_source == "lcz" else FromNetCDF(path="surface.nc"),
        priority=100,
    )
    grid.sim.add(module)

    module.ensure_prepared()
    generator = module.slb_generator
    assert generator is not None
    expected = np.full((2, 2), 0.4)
    if priority > 100:
        expected[np.asarray(mask)] = 12.0
    np.testing.assert_allclose(generator.h_bld, expected)

    module.ensure_prepared()
    assert module.slb_generator is generator
    np.testing.assert_allclose(generator.h_bld, expected)


@pytest.mark.parametrize("collection", [False, True])
def test_slurb_modifications_require_explicit_enable(grid, collection):
    modification = SlurbModification(mask=AllMask(), varname="h_bld", value=12.0)
    source = SlurbModifications([modification]) if collection else modification
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(source, priority=100)
    grid.sim.add(module)

    with pytest.raises(ValueError, match="require include_slurb=True"):
        module.prepare_calculation()
    assert module.lsm_writer is None


def test_equal_priority_slurb_sources_and_collections_preserve_order(grid, tmp_path):
    module = LSMModule(land_use_set=LandUseSet.DEPAC, include_slurb=True)
    module.add(
        SlurbModification(mask=AllMask(), varname="h_bld", value=4.0), priority=100
    )
    module.add(
        SlurbModifications(
            [
                SlurbModification(mask=AllMask(), varname="h_bld", value=8.0),
                SlurbModification(mask=AllMask(), varname="h_bld", value=12.0),
                SlurbModification(
                    mask=AllMask(), varname="material", value=2, dtype="int", n_layers=3
                ),
            ]
        ),
        priority=100,
    )
    grid.sim.add(module)

    module.prepare_calculation()
    generator = module.slb_generator
    assert generator is not None
    np.testing.assert_allclose(generator.h_bld, 12.0)
    path = tmp_path / "inslurb.nc"
    generator.output_nc(path)
    with xr.open_dataset(path) as dataset:
        np.testing.assert_allclose(dataset["h_bld"], 12.0)
        assert dataset["material"].shape == (2, 2, 3)
        assert dataset["material"].dtype.kind == "i"
        np.testing.assert_array_equal(dataset["material"], 2)


def test_lsm_owns_slurb_configuration():
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    assert module.include_slurb is False
    assert module.slb_generator is None
    for name in ("slurb_module", "slb_modifications", "__add__", "__iadd__"):
        assert not hasattr(module, name)


def test_disabled_slurb_preserves_existing_namelist_section(grid):
    module = LSMModule(
        land_use_set=LandUseSet.DEPAC,
        ps=100000.0,
        z0mav=0.1,
        z0hav=0.01,
        iinterp_t=1,
        iinterp_theta=1,
    )
    grid.sim.add(module)
    module.nml["NAMSLURB"] = {"deep_soil_temperature": 300.0}

    module.apply_namelist_from_fields()

    assert module.nml["NAMSLURB"] == {"deep_soil_temperature": 300.0}


@pytest.mark.parametrize(
    "setting",
    ["urban_fraction", "deep_soil_temperature", "building_indoor_temperature"],
)
def test_disabled_slurb_rejects_explicit_namelist_settings(grid, setting):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    setattr(module, setting, 0.5)
    grid.sim.add(module)

    with pytest.raises(ValueError, match=f"{setting} requires include_slurb=True"):
        module.prepare_calculation()
    assert module.lsm_writer is None


@pytest.mark.parametrize("enabled", [False, True])
def test_lsm_writes_slurb_file_only_when_enabled(grid, tmp_path, monkeypatch, enabled):
    module = LSMModule(land_use_set=LandUseSet.DEPAC, include_slurb=enabled)
    if enabled:
        module.add(
            SlurbModification(mask=AllMask(), varname="h_bld", value=12.0), priority=100
        )
    grid.sim.add(module)
    grid.sim.output_path_ = tmp_path
    grid.sim.exp_id = 1
    (tmp_path / "input").mkdir()
    module.prepare_calculation()
    monkeypatch.setattr(module, "apply_namelist_from_fields", Mock())
    save_lsm = Mock()
    monkeypatch.setattr(lsm_output, "save_netcdf", save_lsm)
    monkeypatch.setattr("moda.Surface.LSM.plot_lsm.plot_lsm_cover", Mock())

    module.write_files()

    save_lsm.assert_called_once()
    path = tmp_path / "input" / "inslurb.001.nc"
    assert path.exists() is enabled
    if enabled:
        with xr.open_dataset(path) as dataset:
            np.testing.assert_allclose(dataset["h_bld"], 12.0)
    else:
        assert module.slb_generator is None


@pytest.mark.parametrize("varying", [False, True])
def test_soil_sources_apply_independently_by_priority(grid, varying):
    temperature = np.arange(16, dtype=float).reshape(4, 2, 2) + 280.0
    moisture = np.arange(16, dtype=float).reshape(4, 2, 2) / 100.0 + 0.1
    if varying:
        temperature_source = VaryingSoilTemperature(temperature)
        moisture_source = VaryingSoilMoisture(moisture)
    else:
        temperature_source = UniformSoilTemperature([281.0, 282.0, 283.0, 284.0])
        moisture_source = UniformSoilMoisture([0.1, 0.2, 0.3, 0.4])
        temperature = np.broadcast_to(
            np.array(temperature_source.soil_temperature)[:, None, None], (4, 2, 2)
        )
        moisture = np.broadcast_to(
            np.array(moisture_source.soil_moisture)[:, None, None], (4, 2, 2)
        )
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(temperature_source, priority=40)
    module.add(moisture_source, priority=30)
    module.add(UniformSoilTemperature([270.0] * 4), priority=10)
    module.add(UniformSoilMoisture([0.01] * 4), priority=20)
    writer = LSM_output_dales(grid)
    writer.standard_fill_geometry_modification()

    module._apply_sources(writer)

    np.testing.assert_allclose(writer.soil.t_soil, temperature)
    np.testing.assert_allclose(writer.soil.theta_soil, moisture)
    for tile in writer.land_use_fields.values():
        np.testing.assert_allclose(tile.tskin, 273.15)


def test_varying_skin_temperature_with_aquatic_override(grid, lcz_loader):
    temperature = np.array([[290.0, 291.0], [292.0, 293.0]])
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(
        VaryingSkinTemperature(temperature, aquatic_skin_temperature=285.0),
        priority=200,
    )
    module.add(FromLCZ(), priority=100)
    module._attach_to_sim(grid.sim)
    writer = LSM_output_dales(grid)
    writer.standard_fill_geometry_modification()

    module._apply_sources(writer)

    lcz_loader.assert_called_once()
    for name, tile in writer.land_use_fields.items():
        expected = 285.0 if writer.land_use[name].definition.laqu else temperature
        np.testing.assert_allclose(tile.tskin, expected)
    temperature.fill(0.0)
    np.testing.assert_allclose(
        writer.land_use_fields["urb"].tskin, [[290.0, 291.0], [292.0, 293.0]]
    )


@pytest.mark.parametrize("temperatures", [(290.0, 295.0), (295.0, 290.0)])
@pytest.mark.parametrize("as_list", [False, True])
def test_equal_priority_sources_preserve_insertion_order(grid, temperatures, as_list):
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    sources = [UniformSkinTemperature(temperature) for temperature in temperatures]
    if as_list:
        module.add(sources, priority=100)
    else:
        for source in sources:
            module.add(source, priority=100)
    writer = LSM_output_dales(grid)

    module._apply_sources(writer)

    for tile in writer.land_use_fields.values():
        np.testing.assert_allclose(tile.tskin, temperatures[-1])


@pytest.mark.parametrize("use_as_tskin", [False, True])
def test_harmonie_source_dispatches_soil_and_optional_skin(
    grid, monkeypatch, use_as_tskin
):
    source = SoilTemperatureMoistureFromHarmonie(
        "soil.nc",
        [0.1, 0.2, 0.3, 0.4],
        "2023-01-01T12:00:00",
        use_as_tskin=use_as_tskin,
        aquatic_skin_temperature=285.0,
    )
    loader = Mock()
    temperature = np.arange(16, dtype=float).reshape(4, 2, 2) + 280.0
    moisture = np.arange(16, dtype=float).reshape(4, 2, 2) / 100.0
    skin = np.array([[290.0, 291.0], [292.0, 293.0]])
    temperature_getter = Mock(return_value=temperature)
    moisture_getter = Mock(return_value=moisture)
    skin_getter = Mock(return_value=skin)
    monkeypatch.setattr(source, "get_soil_temp_moisture_arrays", loader)
    monkeypatch.setattr(source, "get_soil_temperature_array", temperature_getter)
    monkeypatch.setattr(source, "get_soil_moisture_array", moisture_getter)
    monkeypatch.setattr(source, "get_tskin_array", skin_getter)
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module.add(source, priority=100)
    module._attach_to_sim(grid.sim)
    writer = LSM_output_dales(grid)
    writer.standard_fill_geometry_modification()

    module._apply_sources(writer)

    loader.assert_called_once_with(grid, module.dz_soil)
    temperature_getter.assert_called_once_with(grid, module.dz_soil)
    moisture_getter.assert_called_once_with(grid, module.dz_soil)
    np.testing.assert_allclose(writer.soil.t_soil, temperature)
    np.testing.assert_allclose(writer.soil.theta_soil, moisture)
    if use_as_tskin:
        skin_getter.assert_called_once_with()
    else:
        skin_getter.assert_not_called()
    for name, tile in writer.land_use_fields.items():
        expected = (
            (285.0 if writer.land_use[name].definition.laqu else skin)
            if use_as_tskin
            else 273.15
        )
        np.testing.assert_allclose(tile.tskin, expected)


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
        "moda.Surface.LSM.LCZ.get_from_lcz.do_everything",
        loader,
    )
    writer = LSM_output_dales(grid, LandUseSet.DEPAC, include_slurb=include_slurb)
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
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
    monkeypatch.setattr("moda.Surface.LSM.LCZ.get_from_lcz.do_everything", loader)
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
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
    monkeypatch.setattr("moda.Surface.LSM.LCZ.from_netcdf.load_from_netcdf", loader)
    apply_slurb = Mock()
    monkeypatch.setattr(lsm_slurb, "apply_from_netcdf", apply_slurb)
    module = LSMModule(land_use_set=LandUseSet.DEPAC)
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
        lsm_output,
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
    from moda.modular.dales_simulation import dales_simulation

    module = LSMModule(land_use_set=LandUseSet.DEPAC)
    module._attach_to_sim(dales_simulation("test", {}))
    stages = Mock()
    writer = Mock(nlu=3)
    module.module_exists = Mock(return_value=False)
    module._create_lsm_writer = Mock(return_value=writer)
    stages.attach_mock(writer.standard_fill_geometry_modification, "initialize")
    for name in ("_apply_sources",):
        helper = Mock()
        setattr(module, name, helper)
        stages.attach_mock(helper, name)
    stages.attach_mock(writer.trim_landuse, "trim")
    module.prepare_calculation()
    assert stages.mock_calls == [
        call.initialize(),
        call._apply_sources(writer),
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
    moda.Surface.LSM.lsm_output.save_netcdf(writer, tmp_path, 1)
    with xr.open_dataset(tmp_path / "lsm.inp_001.nc") as dataset:
        assert dataset.sizes["nlu"] == 2
        assert "cover_slb" not in dataset
        np.testing.assert_allclose(dataset["cover_urb"], 0.3)
        np.testing.assert_allclose(dataset["cover_tot"], 1.0)
        assert dataset["t_soil"].dims == ("z", "y", "x")
        np.testing.assert_array_equal(
            dataset["lslurb"].values,
            [
                b"T" if state.definition.lslurb else b"F"
                for state in writer.land_use.values()
            ],
        )
