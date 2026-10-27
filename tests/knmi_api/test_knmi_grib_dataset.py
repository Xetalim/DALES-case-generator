from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import xarray as xr

from MoDa.LBC.nest_dales_in_HARMONIE import nest_dales_in_KNMI as knmi


def test_grib_variables_keep_independent_height_coordinates(monkeypatch):
    import earthkit.data as ekd

    heights = {"grad": [0], "t": [0, 2, 10, 50, 100, 300], "td": [2]}
    fields = []
    for parameter in heights:
        field = MagicMock()
        field.parameter.variable.return_value = parameter
        field.vertical.level_type.return_value = "height_above_ground_level"
        field.get.return_value = 0
        fields.append(field)

    def to_fieldlist(parameter_fields):
        assert len(parameter_fields) == 1
        parameter = parameter_fields[0].parameter.variable()
        levels = heights[parameter]
        dataset = xr.Dataset(
            {
                parameter: (
                    ("step", "height_above_ground_level", "y", "x"),
                    np.arange(len(levels), dtype=float).reshape(1, -1, 1, 1),
                )
            },
            coords={
                "forecast_reference_time": [np.datetime64("2026-10-19")],
                "step": [np.timedelta64(0, "h")],
                "height_above_ground_level": levels,
                "y": [0.0],
                "x": [0.0],
            },
        )
        return SimpleNamespace(to_xarray=lambda **kwargs: dataset)

    monkeypatch.setattr(
        ekd, "from_source", lambda *args: SimpleNamespace(to_fieldlist=lambda: fields)
    )
    monkeypatch.setattr(ekd, "FieldList", SimpleNamespace(from_fields=to_fieldlist))
    monkeypatch.setattr(knmi, "fix_lambert_offsets", lambda dataset: dataset)

    dataset = knmi._open_knmi_grib_dataset("forecast.grib")

    assert dataset["grad"].sizes["height_above_ground_level_grad"] == 1
    assert dataset["t"].sizes["height_above_ground_level_t"] == 6
    assert dataset["td"].sizes["height_above_ground_level_td"] == 1
    assert float(knmi._select_height_level(dataset["t"]).item()) == 1.0
    assert float(knmi._select_height_level(dataset["td"]).item()) == 0.0
    assert not dataset["grad"].isnull().any()
