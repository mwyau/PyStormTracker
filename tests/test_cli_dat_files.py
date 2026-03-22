from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pystormtracker.cli import main


def test_cli_load_dat_files(tmp_path: Path) -> None:
    """Test parsing of legacy zone.dat and adapt.dat from CLI."""
    # 1. Prepare fake zone.dat (with header)
    zone_file = tmp_path / "zone.dat"
    zone_content = "3\n  0.0  360.0  -90.0  -20.0  6.5\n  0.0  360.0  -20.0   20.0  3.0\n  0.0  360.0   20.0   90.0  6.5"
    zone_file.write_text(zone_content)

    # 2. Prepare fake adapt.dat (4x2 format)
    adapt_file = tmp_path / "adapt.dat"
    adapt_content = "1.0 1.0\n2.0 0.3\n5.0 0.1\n8.0 0.0"
    adapt_file.write_text(adapt_content)

    # 3. Patch run_tracker and sys.argv to run main()
    test_args = [
        "stormtracker",
        "-i", "dummy.nc",
        "-v", "msl",
        "-o", "output.txt",
        "--algorithm", "hodges",
        "--zone-file", str(zone_file),
        "--adapt-file", str(adapt_file)
    ]

    with patch("sys.argv", test_args), \
         patch("pystormtracker.cli.run_tracker") as mock_run:
        main()

    # 4. Verify that run_tracker was called with parsed numpy arrays
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    
    zones_parsed = kwargs["zones"]
    assert zones_parsed.shape == (3, 5)
    assert zones_parsed[1, 4] == 3.0

    adapt_parsed = kwargs["adapt_params"]
    # Should be 2x4 after transpose
    assert adapt_parsed.shape == (2, 4)
    assert np.array_equal(adapt_parsed[0], [1.0, 2.0, 5.0, 8.0])
    assert np.array_equal(adapt_parsed[1], [1.0, 0.3, 0.1, 0.0])


def test_cli_load_dat_json(tmp_path: Path) -> None:
    """Test parsing of zones and adapt-params from JSON strings."""
    zones_json = "[[0.0, 360.0, -90.0, 90.0, 10.0]]"
    adapt_json = "[[1.0, 2.0, 3.0, 4.0], [1.0, 0.5, 0.2, 0.1]]"
    
    test_args = [
        "stormtracker",
        "-i", "dummy.nc",
        "-v", "msl",
        "-o", "output.txt",
        "--algorithm", "hodges",
        "--zones", zones_json,
        "--adapt-params", adapt_json
    ]

    with patch("sys.argv", test_args), \
         patch("pystormtracker.cli.run_tracker") as mock_run:
        main()

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    
    assert np.array_equal(kwargs["zones"], np.array(json.loads(zones_json)))
    assert np.array_equal(kwargs["adapt_params"], np.array(json.loads(adapt_json)))
