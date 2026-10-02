from __future__ import annotations

from modular_dales.Configuration import (
    ColumnStatisticsOutputModule,
    SprayingModule,
    VirtualMeasurementOutputModule,
)

from tests.sim_builders.test_basic import _build_basic_sim


def _configure_sim(sim) -> None:
    sim.init_output_folder()
    sim.do_config()
    sim.check_settings()
    sim.prepare_all_calculations()


def test_virtual_measurement_resolves_real_coordinates(machine_conf):
    sim = _build_basic_sim(machine_conf("virtual_measurement"))
    sim += VirtualMeasurementOutputModule(
        x=[14.9, 149.0],
        y=[6.0, 144.9],
    )

    _configure_sim(sim)

    assert sim.nml["NAMVIRTUALMEASUREMENT"]["lvirtualmeasurement"] is True
    assert sim.nml["NAMVIRTUALMEASUREMENT"]["npoints"] == 2
    assert sim.nml["NAMVIRTUALMEASUREMENT"]["x_idx"] == [2, 15]
    assert sim.nml["NAMVIRTUALMEASUREMENT"]["y_idx"] == [1, 15]


def test_colstat_preserves_explicit_indices(machine_conf):
    sim = _build_basic_sim(machine_conf("colstat"))
    sim += ColumnStatisticsOutputModule(
        x_idx=[2, 5],
        y_idx=[3, 4],
    )

    _configure_sim(sim)

    assert sim.nml["NAMCOLSTAT"]["lcolstat"] is True
    assert sim.nml["NAMCOLSTAT"]["npoints"] == 2
    assert sim.nml["NAMCOLSTAT"]["x_idx"] == [2, 5]
    assert sim.nml["NAMCOLSTAT"]["y_idx"] == [3, 4]


def test_spraying_sources_are_routed_from_shared_point_fields(machine_conf):
    sim = _build_basic_sim(machine_conf("spraying_shared_points"))
    sim += SprayingModule(
        x_idx=[3],
        y_idx=[4],
        z_idx=[2],
        lwater_spraying=True,
    )

    _configure_sim(sim)

    assert sim.nml["namspraying"]["i_glob_spray"] == 3
    assert sim.nml["namspraying"]["j_glob_spray"] == 4
    assert sim.nml["namspraying"]["k_glob_spray"] == 2


def test_spraying_sources_resolve_real_coordinates(machine_conf):
    sim = _build_basic_sim(machine_conf("spraying_real_coords"))
    sim += SprayingModule(
        x=14.0,
        y=25.0,
        z=5.0,
        lwater_spraying=True,
    )

    _configure_sim(sim)

    assert sim.nml["namspraying"]["i_glob_spray"] == 2
    assert sim.nml["namspraying"]["j_glob_spray"] == 3
    assert sim.nml["namspraying"]["k_glob_spray"] == 1
