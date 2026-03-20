from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pystormtracker.io.imilast import read_imilast, write_imilast
from pystormtracker.models.tracks import Tracks


def test_write_imilast_filename_extension(tmp_path: Path) -> None:
    tracks = Tracks()
    outfile = tmp_path / "test"
    write_imilast(tracks, outfile)
    assert Path(str(outfile) + ".txt").exists()


def test_write_imilast_content(tmp_path: Path) -> None:
    tracks = Tracks()
    t1_time = np.datetime64("2025-12-01T00:00:00")
    # IMILAST longitude is typically -180 to 180 or 0 to 360.
    # write_imilast converts > 180 to negative.
    tracks.bulk_append(
        tids=np.array([1, 1], dtype=np.int64),
        times=np.array(
            [t1_time, t1_time + np.timedelta64(6, "h")], dtype="datetime64[s]"
        ),
        lats=np.array([10.0, 11.0]),
        lons=np.array([190.0, 20.0]),
        vars_dict={"Intensity1": np.array([1000.0, 990.0])},
    )

    outfile = tmp_path / "test_content.txt"
    write_imilast(tracks, outfile, decimal_places=2)

    content = outfile.read_text()
    lines = content.splitlines()

    assert lines[0].startswith("99 00")
    assert lines[1] == "90 1 2"
    # 190.0 should become -170.00
    assert " -170.00 10.00 1000.00" in lines[2]
    assert " 20.00 11.00 990.00" in lines[3]


def test_read_imilast_different_time_formats(tmp_path: Path) -> None:
    # Avoid 10-digit timestamps that conflict with YYYYMMDDHH
    t_06 = int(datetime(2025, 12, 1, 6, tzinfo=UTC).timestamp())
    content = (
        "99 00,CycloneNo,StepNo,DateI10,Year,Month,Day,Time,LongE,LatN,Intensity1\n"
        "90 1 3\n"
        "00 1 1 2025120100 2025 12 01 00 10.0 20.0 1000.0\n"
        f"00 1 2 {t_06}.0 2025 12 01 06 11.0 21.0 990.0\n"  # Add .0 to avoid length 10
        "00 1 3 2025-12-01T12:00:00 2025 12 01 12 12.0 22.0 980.0\n"  # iso-like
    )
    infile = tmp_path / "test_read.txt"
    infile.write_text(content)

    tracks = read_imilast(infile)

    assert len(tracks) == 1
    track = tracks[0]
    assert len(track) == 3
    assert track[0].time == np.datetime64("2025-12-01T00:00:00")
    assert track[1].time == np.datetime64("2025-12-01T06:00:00")
    assert track[2].time == np.datetime64("2025-12-01T12:00:00")


def test_read_imilast_empty_lines(tmp_path: Path) -> None:
    content = "99 00,...\n\n90 1 1\n00 1 1 2025120100 2025 12 01 00 10.0 20.0 1000.0\n"
    infile = tmp_path / "test_empty.txt"
    infile.write_text(content)
    tracks = read_imilast(infile)
    assert len(tracks) == 1


def test_write_imilast_exception_handling(tmp_path: Path) -> None:
    """Test handling of invalid times during writing."""
    tracks = Tracks()
    # np.datetime64('NaT') might trigger exception in datetime.fromtimestamp
    tracks.bulk_append(
        tids=np.array([1], dtype=np.int64),
        times=np.array(["NaT"], dtype="datetime64[s]"),
        lats=np.array([0.0]),
        lons=np.array([0.0]),
        vars_dict={"Intensity1": np.array([1000.0])},
    )

    outfile = tmp_path / "test_nat.txt"
    write_imilast(tracks, outfile)
    content = outfile.read_text()
    assert "0000000000 0 00 00 00" in content
