from dataclasses import dataclass, fields

import pytest

from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.modular.dales_simulation import dales_simulation
from modular_dales.modular.simulation_module import simulation_module


@dataclass
class RequiredModule(simulation_module):
    value: int

    def do_config(self):
        pass

    def check_settings(self):
        pass

    def prepare_calculation(self):
        pass

    def write_files(self):
        pass


@pytest.mark.parametrize("kwargs", [{}, {"sim": None}])
@pytest.mark.parametrize(
    "factory", [DefaultNamelistModule, lambda **kwargs: RequiredModule(3, **kwargs)]
)
def test_module_can_be_created_detached_and_attached_later(factory, kwargs):
    module = factory(**kwargs)
    assert module.sim is None
    assert module.do_config_done is False
    assert module._provided_forcings is None
    assert "sim=" not in repr(module)
    module.module_name = "detached"
    sim = dales_simulation("test", {})

    sim += module

    assert module.sim is sim
    assert sim.modules == [module]


@pytest.mark.parametrize(
    "factory", [DefaultNamelistModule, lambda **kwargs: RequiredModule(3, **kwargs)]
)
def test_module_can_receive_sim_at_construction(factory):
    sim = dales_simulation("test", {})
    module = factory(sim=sim)
    assert module.sim is sim
    assert module.do_config_done is False


def test_sim_field_is_inherited_and_optional():
    for module_type in (DefaultNamelistModule, RequiredModule):
        assert "sim" not in module_type.__annotations__
        sim_field = next(item for item in fields(module_type) if item.name == "sim")
        assert sim_field.default is None
        assert sim_field.kw_only
        assert not sim_field.repr
