from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.io.imilast import read_imilast
from pystormtracker.models.tracks import Tracks
from tests.utils import get_integration_msl_path


def run_command_direct(cmd_args: list[str]) -> Tracks | None:
    """Utility to run the tracker directly and return results."""
    import argparse

    from pystormtracker import compare, convert, sample, track

    # Prepend 'track' if missing
    if cmd_args and cmd_args[0] not in ["track", "sample", "convert", "compare"]:
        cmd_args = ["track", *cmd_args]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    track.setup_parser(subparsers)
    sample.setup_parser(subparsers)
    convert.setup_parser(subparsers)
    compare.setup_parser(subparsers)

    args = parser.parse_args(cmd_args)
    if hasattr(args, "func"):
        return cast(Tracks, args.func(args))
    return None


@pytest.mark.integration
def test_hodges_msl_minimum_integration(tmp_path: Path) -> None:
    """Track coherent December MSL minima through the real Hodges path."""
    out_file = tmp_path / "hodges_tracks.txt"

    args = [
        "track",
        "-i",
        str(get_integration_msl_path()),
        "--variable",
        "msl",
        "-m",
        "min",
        "--object-threshold",
        "98000",
        "--feature-refinement",
        "grid",
        "--backend",
        "serial",
        "--no-progress",
        "-o",
        str(out_file),
        "-a",
        "hodges",
        "--format",
        "imilast",
    ]

    run_command_direct(args)

    assert out_file.exists()
    tracks = read_imilast(out_file)
    assert len(tracks) > 0
    assert any(len(tr) >= 2 for tr in tracks)


@pytest.mark.integration
def test_segmented_hodges_tracking_matches_monolithic_tracking() -> None:
    """Segmentation and splicing preserve the complete synthetic trajectory."""
    times = np.arange(6).astype("timedelta64[h]") + np.datetime64("2025-12-01")
    values = np.full((6, 9, 12), 10.0)
    values[:, 4, 3] = -10.0
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": np.linspace(-80.0, 80.0, 9),
            "lon": np.arange(0.0, 360.0, 30.0),
        },
        name="msl",
    )

    def run(*, segment_frames: int | None) -> Tracks:
        return HodgesTracker(
            min_track_points=2,
            feature_refinement="grid",
            segment_frames=segment_frames,
        ).track(data, "msl", detection_mode="min", object_threshold=0.0)

    monolithic = run(segment_frames=None)
    segmented = run(segment_frames=2)

    assert len(monolithic) == len(segmented) == 1
    assert len(monolithic[0]) == len(segmented[0]) == 6
    assert monolithic.metadata == segmented.metadata
    np.testing.assert_array_equal(monolithic.ids, segmented.ids)
    np.testing.assert_array_equal(monolithic.offsets, segmented.offsets)
    np.testing.assert_array_equal(monolithic.times, segmented.times)
    np.testing.assert_array_equal(monolithic.lats, segmented.lats)
    np.testing.assert_array_equal(monolithic.lons, segmented.lons)
    np.testing.assert_array_equal(
        monolithic.variables["msl"], segmented.variables["msl"]
    )
