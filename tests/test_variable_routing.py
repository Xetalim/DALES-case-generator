from types import SimpleNamespace

import numpy as np
import pytest

from modular_dales import vars
from modular_dales.Atmosphere import AtmosphereModule, AtmosphericProfile, lin


@pytest.mark.parametrize(
    "group, names",
    [
        (vars.UVariable, ("ua", "ug")),
        (vars.VVariable, ("va", "vg")),
        (vars.WVariable, ("w", "wa")),
        (vars.SoilVariable, ("t_soil", "theta_soil", "type_soil")),
        (vars.SurfaceVariable, ("thls", "z0m", "z0h")),
    ],
)
def test_variable_groups(group, names):
    assert all(type(getattr(vars, name)) is group for name in names)


@pytest.mark.parametrize(
    "nudging, target, file_name, boundary_name",
    [
        (vars.ua_nudge, vars.ua, "ua_nud", "u"),
        (vars.va_nudge, vars.va, "va_nud", "v"),
        (vars.wa_nudge, vars.w, "wa_nud", "w"),
        (vars.thl_nudge, vars.thetal, "thetal_nud", "thl"),
        (vars.qt_nudge, vars.qt, "qt_nud", "qt"),
    ],
)
def test_nudging_derives_routing_from_target(nudging, target, file_name, boundary_name):
    assert isinstance(nudging, vars.NudgingVariable)
    assert nudging.physical_variable is target
    assert not hasattr(nudging, "fallback")
    assert nudging.unit == target.unit
    assert nudging.init_name == file_name
    assert nudging.boundary_name == target.boundary_name == boundary_name
    assert nudging.init_time_height
    assert nudging.can_be_time_dependent
    assert not nudging.is_initial_profile
    assert target.is_initial_profile
    assert vars.get_var_by_name()[nudging.name] is nudging
    assert target.nudger is nudging
    assert nudging.nudger is nudging


def test_custom_nudging_variable():
    target = vars.UVariable("custom_u", "Custom eastward wind", "m/s")
    nudging = vars.NudgingVariable(target)
    assert nudging.name == "custom_u_nudge"
    assert nudging.init_name == "custom_u_nud"
    assert nudging.boundary_name == "u"
    assert not hasattr(nudging, "fallback")
    assert hash(nudging)


def test_custom_variable_nudger_is_registered_once():
    target = vars.UVariable("routing_test_u", "Test wind", "m/s")
    previous = list(vars.ALL_VARIABLES)
    try:
        nudger = target.nudger
        assert target.nudger is nudger
        assert nudger.target is target
        assert vars.get_var_by_name()[nudger.name] is nudger
    finally:
        vars.ALL_VARIABLES[:] = previous
        vars.ATMO_VARS_BY_NAME.clear()
        vars.ATMO_VARS_BY_NAME.update(
            {variable.name: variable for variable in previous}
        )


@pytest.mark.parametrize("variable", [vars.dudt_ls, vars.nudging_constant_ua])
def test_non_state_profiles_have_no_nudger(variable):
    with pytest.raises(TypeError, match="state profile"):
        _ = variable.nudger


@pytest.mark.parametrize("variable", [vars.ua, vars.va, vars.w, vars.thetal, vars.qt])
def test_add_routes_physical_profiles_by_destination(variable):
    atmo = AtmosphereModule()
    atmo.sim = SimpleNamespace(grid=SimpleNamespace(zt=np.array([5.0, 15.0])))
    state = AtmosphericProfile(variable=variable, shape=lin(surf_val=2.0, ddz=0.0))
    target = AtmosphericProfile(variable=variable, shape=lin(surf_val=4.0, ddz=0.0))
    assert atmo.add([(0.0, target), (60.0, target)], nudging=True) is atmo
    forcings = atmo.provide_forcings()
    assert variable.name not in forcings.initial
    np.testing.assert_allclose(forcings.series[variable.nudger.name], 4.0)
    assert target.variable is variable
    assert atmo.add(state) is atmo
    forcings = atmo.provide_forcings()
    np.testing.assert_allclose(forcings.initial[variable.name], 2.0)
    atmo.forcings = forcings
    dataset = atmo.init_dataset()
    np.testing.assert_allclose(dataset[variable.nudger.init_name], 4.0)
    np.testing.assert_allclose(dataset[variable.name], 2.0)


def test_profile_addition_operators_are_removed():
    atmo = AtmosphereModule()
    configured = AtmosphericProfile(variable=vars.ua, shape=lin(surf_val=2.0, ddz=0.0))
    with pytest.raises(TypeError):
        atmo + configured
    with pytest.raises(TypeError):
        atmo += configured


def test_nudging_requires_a_state_profile():
    with pytest.raises(TypeError, match="non-nudging profile"):
        vars.NudgingVariable(vars.thls)
    with pytest.raises(TypeError, match="non-nudging profile"):
        vars.NudgingVariable(vars.ua_nudge)
