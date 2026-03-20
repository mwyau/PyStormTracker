from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.tracks import Tracks


def write_hodges(tracks: Tracks, outfile: str) -> None:
    """
    Writes tracks in the Hodges (TRACK) ASCII format.

    Args:
        tracks: The Tracks object to write.
        outfile: Path to the output file.
    """
    n_tracks = len(tracks)

    with open(outfile, "w") as f:
        # Headers (matching TRACK's default format)
        f.write("0\n")
        f.write("0 0\n")
        f.write(f"TRACK_NUM {n_tracks:10d} ADD_FLD    0   0 &\n")

        # Determine available variables
        var_keys = list(tracks.vars.keys())
        varname = var_keys[0] if var_keys else "intensity"

        for tr_id, track in enumerate(tracks, start=1):
            n_points = len(track)
            f.write(f"TRACK_ID {tr_id:2d}\n")
            f.write(f"POINT_NUM {n_points:3d}\n")

            # TRACK format: time_step_idx longitude latitude intensity
            # [additional_fields]
            for i, center in enumerate(track, start=1):
                intensity = center.vars.get(varname, 0.0)
                # Ensure longitude is in [0, 360] as per standard TRACK
                lon = center.lon % 360.0

                f.write(f"{i:d} {lon:10.6f} {center.lat:10.6f} {intensity:12.6e}\n")
