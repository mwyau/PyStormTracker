from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pystormtracker.convert import generate_html
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks


@pytest.fixture
def dummy_tracks() -> Tracks:
    tracks = Tracks()
    centers = [
        Center(
            time=np.datetime64("2020-01-01T00:00"),
            lat=50.0,
            lon=0.0,
            vars={"msl": 1000.0},
        )
    ]
    tracks.add_track(centers)
    return tracks


def test_generate_html_standalone(dummy_tracks: Tracks, tmp_path: Path) -> None:
    outfile = tmp_path / "explorer.html"
    generate_html(dummy_tracks, outfile, split=False)

    assert outfile.exists()
    content = outfile.read_text()
    assert "window.TRACKS_DATA =" in content
    # The JSON format uses column-oriented SoA: "lat":[50.0]
    assert '"lat":[50.0]' in content


def test_generate_html_split(dummy_tracks: Tracks, tmp_path: Path) -> None:
    outfile = tmp_path / "explorer.html"
    generate_html(dummy_tracks, outfile, split=True)

    js_file = tmp_path / "explorer.tracks.js"
    assert outfile.exists()
    assert js_file.exists()

    html_content = outfile.read_text()
    js_content = js_file.read_text()

    assert 'src="explorer.tracks.js"' in html_content
    assert "window.TRACKS_DATA =" in js_content
    assert '"lat":[50.0]' in js_content
