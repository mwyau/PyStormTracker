from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.models.units import normalize_variable_units


def _data(values: list[float], units: str | None) -> xr.DataArray:
    attrs = {} if units is None else {"units": units}
    return xr.DataArray(values, dims="x", attrs=attrs, name="field")


def test_pressure_values_and_threshold_are_converted_to_pa() -> None:
    normalized, threshold, unit = normalize_variable_units(
        _data([1000.0], "hPa"), variable="msl", intensity_threshold=2.0
    )
    np.testing.assert_allclose(normalized.values, [100000.0])
    assert threshold == 200.0
    assert unit == "Pa"
    assert normalized.attrs["units"] == "Pa"


def test_scaled_vorticity_is_converted_to_s_inverse() -> None:
    normalized, threshold, unit = normalize_variable_units(
        _data([5.0], "10**-5 s^-1"), variable="vo", intensity_threshold=2.0
    )
    np.testing.assert_allclose(normalized.values, [5.0e-5])
    assert threshold == 2.0e-5
    assert unit == "s^-1"


def test_custom_units_are_preserved_and_missing_defaults_to_one() -> None:
    declared, threshold, unit = normalize_variable_units(
        _data([1.0], "kg m-2"), variable="rain", intensity_threshold=3.0
    )
    assert declared.attrs["units"] == "kg m-2"
    assert threshold == 3.0
    assert unit == "kg m-2"

    missing, _, missing_unit = normalize_variable_units(
        _data([1.0], None), variable="rain", intensity_threshold=None
    )
    assert missing_unit == "1"
    assert "units" not in missing.attrs


def test_recognized_incompatible_units_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported units"):
        normalize_variable_units(
            _data([1.0], "K"), variable="msl", intensity_threshold=None
        )
