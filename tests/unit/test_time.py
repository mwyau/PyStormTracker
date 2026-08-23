from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.io.trackjson import _MAX_SAFE_JSON_INTEGER
from pystormtracker.models._cftime import is_cftime_available
from pystormtracker.models.time import (
    _GREGORIAN_REFORM_DATE,
    PROLEPTIC_GREGORIAN,
    decode_time_values,
    encode_cf_datetimes,
    encode_numeric_time_values,
    encode_time_values,
    format_time,
    infer_calendar,
    normalize_calendar,
)


def test_import_pystormtracker_does_not_import_cftime() -> None:
    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location("
        "'pystormtracker.models._cftime', 'src/pystormtracker/models/_cftime.py')\n"
        "assert spec and spec.loader\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'cftime' not in sys.modules, 'cftime was imported unexpectedly'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_canonical_units_and_modern_numeric_values() -> None:
    values = encode_cf_datetimes(
        [datetime(2025, 1, 1), datetime(2025, 1, 1, 6)],
        calendar="standard",
    )
    np.testing.assert_array_equal(values, [1735689600000, 1735711200000])
    decoded = decode_time_values(values)
    assert decoded == (
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 6, 0),
    )


def test_proleptic_gregorian_supports_historical_dates() -> None:
    value = datetime(1500, 1, 1, 12, 0, 0, 123000)
    packed = encode_cf_datetimes([value], calendar=PROLEPTIC_GREGORIAN)
    decoded = decode_time_values(packed)[0]
    assert (decoded.year, decoded.month, decoded.day, decoded.microsecond) == (
        1500,
        1,
        1,
        123000,
    )


def test_standard_calendar_rejects_pre_transition_dates() -> None:
    with pytest.raises(ValueError, match="mixed Julian/Gregorian") as error:
        encode_numeric_time_values(
            [0.0],
            units="days since 1500-01-01",
            calendar="standard",
        )
    assert _GREGORIAN_REFORM_DATE.isoformat() in str(error.value)


def test_calendar_aliases_and_unsupported_calendars() -> None:
    assert normalize_calendar(None) == PROLEPTIC_GREGORIAN
    assert normalize_calendar("gregorian") == PROLEPTIC_GREGORIAN
    assert normalize_calendar("standard") == PROLEPTIC_GREGORIAN
    assert normalize_calendar(PROLEPTIC_GREGORIAN) == PROLEPTIC_GREGORIAN
    for value in (
        "360_day",
        "noleap",
        "365_day",
        "all_leap",
        "366_day",
        "julian",
        "utc",
        "tai",
        "none",
    ):
        with pytest.raises(ValueError, match="Broader CF-calendar support is deferred"):
            normalize_calendar(value)


def test_sub_millisecond_values_are_rejected_but_internal_int64_is_not_js_limited() -> (
    None
):
    with pytest.raises(ValueError, match="sub-millisecond"):
        encode_cf_datetimes([datetime(2025, 1, 1, 0, 0, 0, 1)], calendar="standard")
    value = encode_time_values([_MAX_SAFE_JSON_INTEGER + 1])[0]
    assert int(value) == _MAX_SAFE_JSON_INTEGER + 1
    assert np.iinfo(np.int64).max > _MAX_SAFE_JSON_INTEGER


def test_numeric_cf_units_are_converted_to_canonical_milliseconds() -> None:
    values = encode_numeric_time_values(
        [0.0, 6.0],
        units="hours since 2025-01-01 00:00:00",
        calendar="gregorian",
    )
    np.testing.assert_array_equal(values, [1735689600000, 1735711200000])


def test_missing_source_calendar_defaults_to_proleptic_gregorian() -> None:
    data = xr.DataArray(
        np.zeros((2, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={
            "time": xr.DataArray(
                [0.0, 6.0],
                dims="time",
                attrs={"units": "hours since 2025-01-01 00:00:00"},
            ),
            "lat": [0.0],
            "lon": [0.0],
        },
        name="msl",
    )
    loaded = DataLoader(data).ensure_open()
    np.testing.assert_array_equal(
        loaded.time.values.astype("datetime64[ms]"),
        np.array(
            ["2025-01-01T00:00:00.000", "2025-01-01T06:00:00.000"],
            dtype="datetime64[ms]",
        ),
    )


def test_source_calendar_aliases_are_canonicalized() -> None:
    values = np.array([np.datetime64("2025-01-01")])
    for calendar in ("standard", "gregorian", PROLEPTIC_GREGORIAN):
        assert (
            infer_calendar(values, attrs={"calendar": calendar}) == PROLEPTIC_GREGORIAN
        )


def test_explicit_standard_source_date_before_transition_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="mixed Julian/Gregorian calendar conversion is not implemented",
    ):
        infer_calendar(
            np.array([np.datetime64("1500-01-01")]),
            attrs={"calendar": "standard"},
        )


def test_data_loader_rejects_pre_transition_numeric_standard_source() -> None:
    historical_time = xr.Variable(
        ("time",),
        np.array([0.0], dtype=np.float64),
        attrs={
            "units": "days since 1500-01-01 00:00:00",
            "calendar": "standard",
        },
    )
    data = xr.DataArray(
        np.zeros((1, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={
            "time": historical_time,
            "lat": [0.0],
            "lon": [0.0],
        },
        name="msl",
    )
    with pytest.raises(ValueError, match="mixed Julian/Gregorian"):
        DataLoader(data).ensure_open()


def test_display_format_has_fractional_seconds_only_when_needed() -> None:
    value = encode_cf_datetimes(
        [datetime(2025, 1, 1, 0, 0, 0, 123000)], calendar=PROLEPTIC_GREGORIAN
    )[0]
    assert format_time(int(value)) == "2025-01-01 00:00:00.123"
    assert format_time(0) == "1970-01-01 00:00:00"
    assert format_time(-1000) == "1969-12-31 23:59:59"


def test_cftime_lazy_adapter_when_cftime_objects_used() -> None:
    if is_cftime_available():
        import cftime

        cf_dt = cftime.DatetimeProlepticGregorian(2024, 6, 15, 12, 0, 0)
        encoded = encode_time_values([cf_dt])
        assert encoded[0] == 1718452800000


def test_data_loader_default_engine_h5netcdf(tmp_path: Path) -> None:
    data = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(2), "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        name="msl",
    )
    nc_path = tmp_path / "test.nc"
    data.to_netcdf(nc_path, engine="h5netcdf")

    loader = DataLoader(nc_path)
    ds = loader.ensure_open()
    assert ds is not None
    assert loader.engine is None  # resolved to h5netcdf in ensure_open
