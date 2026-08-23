from __future__ import annotations

import pytest

from pystormtracker.backends import (
    resolve_frame_workers,
    resolve_mge_workers,
    resolve_sht_threads,
    validate_execution_parameters,
)


def test_hodges_resolution_defaults_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pystormtracker.backends.available_cpu_count", lambda: 8)

    assert resolve_frame_workers(None, "dask") == 8
    assert resolve_mge_workers(None, "dask") == 8
    assert resolve_sht_threads(None, "dask") == 1
    assert resolve_sht_threads(None, "serial") == 0
    assert resolve_frame_workers(1, "dask") == 1
    assert resolve_mge_workers(16, "dask") == 16
    assert resolve_sht_threads(16, "dask") == 16


@pytest.mark.parametrize("name", ["frame_workers", "sht_threads", "mge_workers"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_hodges_controls_require_positive_integers(name: str, value: int) -> None:
    control = {name: value}
    with pytest.raises((TypeError, ValueError), match=name):
        validate_execution_parameters("dask", **control)


def test_hodges_scheduler_controls_are_rejected_outside_dask() -> None:
    with pytest.raises(ValueError, match="frame_workers"):
        validate_execution_parameters("serial", frame_workers=1)
    with pytest.raises(ValueError, match="mge_workers"):
        validate_execution_parameters("serial", mge_workers=1)
    with pytest.raises(ValueError, match="frame_workers"):
        validate_execution_parameters("mpi", frame_workers=1)
    with pytest.raises(ValueError, match="mge_workers"):
        validate_execution_parameters("mpi", mge_workers=1)

    validate_execution_parameters("serial", sht_threads=16)
    validate_execution_parameters("mpi", sht_threads=16)
