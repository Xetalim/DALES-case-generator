from unittest.mock import patch

import numpy as np
import pytest

from MoDa.Atmosphere.Profile import (
    TimedAtmosphereProfile,
)
from MoDa.Atmosphere.Profile import AtmosphericProfile, InterpolatedProfile
from MoDa.Atmosphere.shapes import exp, expsinw, lin, linmlsurf
from MoDa.modular.serialize_deserialize import (
    _deserialize_dataclass,
    asdict_user_set,
)
from MoDa.modular.time_dependent_scalars import TimeDependentScalar
from MoDa.vars import ua


def test_interpolated_profile_roundtrip_skips_incompatible_union_member():
    original = TimedAtmosphereProfile(
        time=0.0,
        profile=InterpolatedProfile(variable=ua, z=[0.0, 100.0], points=[2.0, 4.0]),
    )
    serialized = asdict_user_set(original)
    with patch.object(
        AtmosphericProfile,
        "__init__",
        side_effect=AssertionError(
            "An interpolated profile must not use this constructor"
        ),
    ):
        restored = _deserialize_dataclass(TimedAtmosphereProfile, serialized)
    assert isinstance(restored.profile, InterpolatedProfile)
    assert restored.profile.variable is ua
    np.testing.assert_allclose(restored.profile.evaluate([50.0]), [3.0])


@pytest.mark.parametrize(
    "shape",
    [
        lin(surf_val=2.0, ddz=0.01),
        exp(surf_val=2.0, lambda_val=100.0),
        expsinw(surf_val=2.0, H=100.0, amp=0.1, Hp=200.0),
        linmlsurf(lapse_rate=0.01, surf_val=2.0),
    ],
)
def test_shaped_profile_roundtrip_preserves_concrete_shape(shape):
    original = TimedAtmosphereProfile(
        time=0.0, profile=AtmosphericProfile(variable=ua, shape=shape)
    )
    serialized = asdict_user_set(original)
    profile_data = serialized["profile"]["AtmosphericProfile"]
    assert list(profile_data["shape"]) == [type(shape).__name__]
    assert "name" not in profile_data["shape"][type(shape).__name__]
    restored = _deserialize_dataclass(TimedAtmosphereProfile, serialized)
    assert type(restored.profile.shape) is type(shape)
    np.testing.assert_allclose(
        restored.profile.evaluate([5.0, 150.0]), original.profile.evaluate([5.0, 150.0])
    )


def test_registered_shape_preserves_nested_time_dependent_scalar():
    original = AtmosphericProfile(
        variable=ua,
        shape=lin(
            surf_val=TimeDependentScalar(times=[0.0, 60.0], values=[2.0, 4.0]),
            ddz=0.0,
        ),
    )
    serialized = asdict_user_set(original)
    assert "TimeDependentScalar" in serialized["shape"]["lin"]["surf_val"]
    restored = _deserialize_dataclass(AtmosphericProfile, serialized)
    assert isinstance(restored.shape.surf_val, TimeDependentScalar)
    np.testing.assert_allclose(
        restored.profile_at(30.0).profile.evaluate([5.0, 15.0]), [3.0, 3.0]
    )
