# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration check for a real-data TrackJSON round trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from pystormtracker.io.trackjson import read_trackjson
from pystormtracker.simple.tracker import SimpleTracker
from tests.utils import get_integration_msl_path


@pytest.mark.integration
def test_msl_tracking_trackjson_round_trip(tmp_path: Path) -> None:
    """Preserve complete canonical Tracks through a temporary TrackJSON file."""
    source = get_integration_msl_path()
    expected = SimpleTracker(backend="serial").track(
        source,
        variable="msl",
        detection_mode="min",
    )

    output = tmp_path / "december_msl.trackjson"
    expected.write(output)
    loaded = read_trackjson(output)

    assert loaded == expected
