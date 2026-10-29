from dataclasses import dataclass, field, fields
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from moda.Atmosphere import AtmosphereModule, lin
from moda.Atmosphere.Profile import AtmosphericProfile
from moda.modular.dales_simulation import DefaultNamelistModule, dales_simulation
from moda.modular.simulation_module import simulation_module
from moda.Surface.surfacemodules import ConstantFluxesModule
from moda.vars import ua


@dataclass
class RequiredModule(simulation_module):
    value: int
    module_name: str = field(
        default="RequiredModule", init=False, metadata={"serialize": False}
    )
    attachments: int = field(default=0, init=False, metadata={"serialize": False})

    def config_value(self):
        return self.value

    def simulation_name(self):
        return self.sim.case_name

    def _initialize_from_sim(self, sim):
        assert self.sim is sim
        self.attachments += 1

    def do_config(self):
        pass

    def check_settings(self):
        pass

    def prepare_calculation(self):
        pass

    def write_files(self):
        pass


@pytest.mark.parametrize("factory", [DefaultNamelistModule, lambda: RequiredModule(3)])
def test_module_allows_deferred_construction(factory):
    module = factory()
    module.check_settings()
    with pytest.raises(RuntimeError, match="detached"):
        _ = module.sim
    sim = dales_simulation("test", {})
    module.module_name = "test"
    sim.add(module)
    module.do_config()
    assert module.sim is sim
    assert sim.modules == [module]


def test_attachment_is_idempotent_and_rejects_reassignment():
    sim = dales_simulation("test", {})
    module = RequiredModule(3)
    module._attach_to_sim(sim)
    assert module.sim is sim
    assert sim.modules == []
    sim.add(module)
    sim.add(module)
    assert sim.modules == [module]
    assert module.attachments == 1
    assert module.simulation_name() == "test"
    other = dales_simulation("other", {})
    with pytest.raises(ValueError, match="different simulation"):
        other.add(module)
    assert other.modules == []
    with pytest.raises(AttributeError):
        module.sim = other


@pytest.mark.parametrize("invalid", [None, object(), "test"])
def test_attachment_requires_simulation(invalid):
    module = RequiredModule(3)
    with pytest.raises(TypeError, match="dales_simulation"):
        module._attach_to_sim(invalid)
    with pytest.raises(RuntimeError, match="detached"):
        _ = module.sim


def test_detached_methods_are_not_wrapped():
    module = RequiredModule(3)
    assert module.config_value() == 3
    module.do_config()
    module.ensure_prepared()
    assert module.prepare_calculation_done
    assert not hasattr(RequiredModule.config_value, "__wrapped__")
    assert not hasattr(module, "parent")
    with pytest.raises(RuntimeError, match="detached"):
        module.simulation_name()


def test_binding_is_private_non_constructor_state():
    for module_type in (DefaultNamelistModule, RequiredModule):
        definitions = {item.name: item for item in fields(module_type)}
        assert "sim" not in definitions
        sim_field = definitions["_sim"]
        assert sim_field.default is None
        assert not sim_field.init
        assert not sim_field.repr
        assert not sim_field.compare
        assert sim_field.metadata["serialize"] is False
        assert "sim" not in signature(module_type).parameters
    assert simulation_module.sim.fget.__annotations__["return"] == "dales_simulation"
    assert RequiredModule(3).do_config_done is False
    assert RequiredModule(3)._provided_forcings is None
    assert "_sim=" not in repr(RequiredModule(3))


def test_atmosphere_configuration_is_detached_until_grid_access():
    module = AtmosphereModule()
    module.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=2.0, ddz=0.0)))
    assert len(module.profiles) == 1
    with pytest.raises(RuntimeError, match="detached"):
        module.provide_forcings()
    sim = dales_simulation("atmosphere", {})
    sim.grid = SimpleNamespace(zt=np.array([5.0, 15.0]))
    assert not sim.has_atmosphere_module
    sim.add(module)
    assert sim.has_atmosphere_module
    np.testing.assert_allclose(module.provide_forcings().initial[ua.name], 2.0)


def test_surface_attachment_sets_simulation_flag():
    module = ConstantFluxesModule()
    sim = dales_simulation("surface", {})
    assert not sim.has_surface_module
    sim.add(module)
    assert sim.has_surface_module
    assert module.sim is sim
