import pytest

import MoDa
import MoDa.Surface


def test_core_modules_are_available_at_package_root():
    assert MoDa.AtmosphereModule is MoDa.Atmosphere.AtmosphereModule
    assert MoDa.OpenBoundaryModule is MoDa.LBC.OpenBoundaryModule
    assert MoDa.DefaultNamelistModule is MoDa.Configuration.DefaultNamelistModule


def test_surface_modules_are_available_from_surface_namespace():
    assert MoDa.Surface.LSMModule is MoDa.LSMModule
    assert MoDa.Surface.LSMHomogeneousModule is MoDa.LSMHomogeneousModule
    assert MoDa.Surface.SlurbModule is MoDa.SLURBModule
    assert MoDa.Surface.SLURBModule is MoDa.SLURBModule


def test_lsm_modifications_are_available_from_lsm_namespace():
    assert MoDa.Surface.LSM.LandUseModification is MoDa.LandUseModification
    assert MoDa.Surface.LSM.LandUseModifications is MoDa.LandUseModifications


def test_documented_surface_construction():
    lsm = MoDa.Surface.LSMModule(land_use_set=MoDa.Surface.LSM.LandUseSet.IFS)
    lsm.add(
        MoDa.Surface.LSM.LandUseModification(
            geometry=MoDa.Geometry.AllGeometry(), type="grs"
        ),
        priority=0,
    )
    slurb = MoDa.Surface.SlurbModule()
    assert type(slurb) is MoDa.SLURBModule
    assert MoDa.Surface.LSM.FromLS2D is MoDa.Atmosphere.FromLS2D


@pytest.mark.parametrize(
    "namespace",
    [
        MoDa,
        MoDa.Atmosphere,
        MoDa.Configuration,
        MoDa.Emission,
        MoDa.Geometry,
        MoDa.IBM,
        MoDa.LBC,
        MoDa.Radiation,
        MoDa.Surface,
        MoDa.Surface.LSM,
    ],
)
def test_all_advertised_exports_exist(namespace):
    for name in namespace.__all__:
        assert hasattr(namespace, name), f"{namespace.__name__}.{name} is missing"


def test_specialist_modules_remain_available_at_package_root():
    for namespace in (
        MoDa.Atmosphere,
        MoDa.Configuration,
        MoDa.Emission,
        MoDa.Geometry,
        MoDa.IBM,
        MoDa.LBC,
        MoDa.Radiation,
        MoDa.Surface,
        MoDa.Surface.LSM,
    ):
        for name in namespace.__all__:
            if hasattr(MoDa, name) and name not in MoDa.__all__:
                assert getattr(MoDa, name) is getattr(namespace, name)


def test_configuration_exposes_simulation_checks():
    from MoDa.Configuration.output_modules import CheckSimulationModule

    assert MoDa.Configuration.CheckSimulationModule is CheckSimulationModule
