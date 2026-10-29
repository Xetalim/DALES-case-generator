from dataclasses import dataclass, field

import numpy as np
import pytest
import xarray as xr

import moda
from moda.modular.forcing import ForcingSet
from moda.modular.simulation_module import simulation_module


@dataclass
class SeriesProvider(simulation_module):
    series: dict[str, xr.DataArray] = field(default_factory=dict)
    module_name: str = field(default="SeriesProvider", init=False)

    def provide_forcings(self) -> ForcingSet:
        return ForcingSet(series=self.series)


def simulation_with_series():
    sim = moda.dales_simulation("timedep", {})
    sim.add(moda.TimeModule(runtime=60))
    sim.add(
        SeriesProvider(
            series={
                "dpdx": xr.DataArray(
                    [[1.0, 1.0], [3.0, 3.0]],
                    dims=("time", "zt"),
                    coords={"time": [0.0, 60.0], "zt": [10.0, 30.0]},
                )
            }
        )
    )
    sim.add(
        SeriesProvider(
            series={
                "thls": moda.TimeDependentScalar(
                    times=[0.0, 20.0, 40.0], values=[290.0, 292.0, 294.0]
                ).as_series()
            }
        )
    )
    return sim


def test_simulation_collects_module_time_axes():
    sim = simulation_with_series()
    sim.prepare_time_dependent_forcings()
    np.testing.assert_array_equal(sim.timedep.time, [0.0, 20.0, 40.0, 60.0])
    np.testing.assert_allclose(
        sim.timedep.dpdx_ls_timedep.isel(zt=0), [1.0, 5 / 3, 7 / 3, 3.0]
    )
    np.testing.assert_allclose(sim.timedep.thlsurf_timedep, [290, 292, 294, 294])
    assert sim.nml["PHYSICS"]["ltimedep"] is True
    assert sim.nml["PHYSICS"]["ntimedep"] == 3


def test_simulation_explicit_axis_and_yaml_roundtrip():
    sim = simulation_with_series()
    sim.timesteps = [0.0, 30.0, 60.0]
    sim.prepare_time_dependent_forcings()
    np.testing.assert_array_equal(sim.timedep.time, sim.timesteps)
    configured = moda.dales_simulation(
        "configured", {}, timesteps=np.array([0.0, 30.0, 60.0])
    )
    restored = moda.dales_simulation.load_sim_from_yaml(
        configured.save_sim_to_yaml(), machine_conf={}
    )
    assert restored.timesteps == configured.timesteps
    assert restored.modules == []


def test_simulation_without_series_disables_time_dependent_forcing():
    sim = moda.dales_simulation("static", {})
    sim.prepare_time_dependent_forcings()
    assert not sim.timedep.data_vars
    assert sim.nml["PHYSICS"]["ltimedep"] is False
    assert sim.nml["PHYSICS"]["ntimedep"] == 0


def test_simulation_rejects_insufficient_time_points():
    sim = simulation_with_series()
    sim.timesteps = [0.0, 60.0]
    with pytest.raises(ValueError, match="at least 3 time points"):
        sim.prepare_time_dependent_forcings()


def test_removed_module_is_not_exported():
    assert not hasattr(moda, "TimedependentModule")
    assert "TimedependentModule" not in moda.module_registry
