"""Legacy parity: current Simple tracker vs v0.0.2 reference output."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from pystormtracker.io.imilast import read_imilast
from pystormtracker.metrics.compare import TrackComparisonConfig, compare_tracks
from tests.utils import fetch_era5_msl, fetch_era5_vo850, get_legacy_track_path

pytestmark = [pytest.mark.parity, pytest.mark.data]


@pytest.fixture(scope="module")
def test_data_msl() -> str:
    """Download MSL test data once per module."""
    return fetch_era5_msl(resolution="2.5x2.5")


@pytest.fixture(scope="module")
def test_data_vo() -> str:
    """Download VO test data once per module."""
    return fetch_era5_vo850(resolution="2.5x2.5")


LEGACY_CONFIGS = [
    pytest.param(("msl", "min", "msl"), id="legacy_msl", marks=pytest.mark.slow),
    pytest.param(("vo", "max", "vo"), id="legacy_vo", marks=pytest.mark.slow),
]


@pytest.fixture(scope="module", params=LEGACY_CONFIGS)
def config_params(request: pytest.FixtureRequest) -> tuple[str, str, str]:
    """Select one of the retained v0.0.2 comparison inputs."""
    return cast(tuple[str, str, str], request.param)


@pytest.fixture(scope="module")
def config(
    request: pytest.FixtureRequest,
    config_params: tuple[str, str, str],
    test_data_msl: str,
    test_data_vo: str,
) -> tuple[str, str, str]:
    variable_name, mode, _ = config_params
    data_path = test_data_msl if variable_name == "msl" else test_data_vo
    return data_path, variable_name, mode


@pytest.mark.parity
@pytest.mark.slow
def test_simple_v0_0_2_parity(
    tmp_path: Path,
    config: tuple[str, str, str],
) -> None:
    """Legacy parity: current Simple tracker vs v0.0.2 reference output."""
    data_path, variable_name, mode = config
    ref_file = get_legacy_track_path(variable_name)

    assert ref_file.is_file(), f"Legacy parity reference data missing: {ref_file}"

    output_file = tmp_path / f"legacy_{variable_name}.txt"

    # Run tracker with output to temp file
    import argparse

    from pystormtracker import track

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    track.setup_parser(subparsers)
    args = parser.parse_args(
        [
            "track",
            "-i",
            data_path,
            "--variable",
            variable_name,
            "-m",
            mode,
            "--backend",
            "serial",
            "-o",
            str(output_file),
        ]
        + (["--feature-threshold", "1e-4"] if variable_name == "vo" else [])
    )
    args.func(args)

    tracks_comp = read_imilast(output_file)
    tracks_ref = read_imilast(ref_file)

    if variable_name == "msl":
        max_dist, min_overlap, min_match_rate = 220.0, 0.8, 0.95
    else:
        max_dist, min_overlap, min_match_rate = 220.0, 0.8, 0.90

    comparison = compare_tracks(
        tracks_ref,
        tracks_comp,
        config=TrackComparisonConfig(
            max_mean_separation_deg=max_dist / 111.195,
            min_overlap_fraction=min_overlap,
        ),
    )

    match_rate = comparison.reference_coverage
    assert match_rate >= min_match_rate
