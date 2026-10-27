import pytest

from MoDa.Geometry.geometry_modification import AllGeometry
from MoDa.IBM.ibm import IBMModification, IBMModule
from MoDa.Surface.LSM.lu_modifications import (
    LandUseModification,
    LandUseModifications,
)


def test_ibm_module_add_is_supported_and_iadd_is_deprecated() -> None:
    modification = IBMModification(geometry=AllGeometry(), height=2.0)
    module = IBMModule(thlibm=293.15, thlroof=293.15)

    assert module.add(modification) is module
    assert module.ibm_modifications == [modification]

    with pytest.warns(DeprecationWarning, match=r"IBM configuration.*add\(\)"):
        module.add(modification)

    assert module.ibm_modifications == [modification, modification]


def test_land_use_add_is_supported_and_iadd_is_deprecated() -> None:
    modification = LandUseModification(geometry=AllGeometry(), type="grs")
    modifications = LandUseModifications()

    assert modifications.add(modification) is modifications
    assert modifications.modifications == [modification]

    with pytest.warns(DeprecationWarning, match=r"land-use modifications.*add\(\)"):
        modifications.add(modification)

    assert modifications.modifications == [modification, modification]
