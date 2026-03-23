import argparse
import os
from pathlib import Path

from . import __version__
from .io.imilast import read_imilast
from .io.json import infer_track_type, read_json, write_json
from .models.tracks import Tracks


def generate_html(tracks: Tracks, outfile: str | Path, split: bool = False) -> None:
    """
    Generates an HTML explorer by injecting the json string and version info.
    If split=True, generates a separate .tracks.js file.
    """
    # 1. Generate the JSON string in memory
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        write_json(tracks, tmp_name)
        with open(tmp_name) as f:
            json_str = f.read()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 2. Read the HTML template
    base_dir = Path(__file__).parent
    template_path = base_dir / "templates" / "explorer.html"

    if not template_path.exists():
        raise FileNotFoundError(f"HTML template not found at {template_path}")

    with open(template_path) as f:
        html_content = f.read()

    # 3. Handle Split vs Standalone
    if split:
        out_path = Path(outfile)
        js_path = out_path.with_suffix(".tracks.js")

        with open(js_path, "w") as f:
            f.write(f"window.TRACKS_DATA = {json_str};")

        print(f"Data written to {js_path}")

        script_tag = f'<script src="{js_path.name}"></script>'

        if "// TRACKS_DATA_PLACEHOLDER" in html_content:
            html_content = html_content.replace(
                "// TRACKS_DATA_PLACEHOLDER", script_tag
            )
        else:
            html_content = html_content.replace(
                '<script src="tracks_data.js"></script>', script_tag
            )
    else:
        injected_script = f"<script>\nwindow.TRACKS_DATA = {json_str};\n</script>"

        if "// TRACKS_DATA_PLACEHOLDER" in html_content:
            html_content = html_content.replace(
                "// TRACKS_DATA_PLACEHOLDER", injected_script
            )
        else:
            html_content = html_content.replace(
                '<script src="tracks_data.js"></script>', injected_script
            )

    version_str = f"v{__version__}"
    html_content = html_content.replace("{{version}}", version_str)

    with open(outfile, "w") as f:
        f.write(html_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PyStormTracker data between formats and generate "
            "interactive HTML visualizations. "
            "For JSON output, the '.tracks.json' extension is recommended."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-f",
        "--in-format",
        choices=["imilast", "json"],
        required=True,
        help="Input file format",
    )
    parser.add_argument(
        "-F",
        "--out-format",
        choices=["imilast", "hodges", "json", "html"],
        required=True,
        help="Output file format",
    )
    parser.add_argument(
        "--type", choices=["msl", "vo"], help="Override track type (msl or vo)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="For 'html' output, generate separate .html and .tracks.js files.",
    )

    args = parser.parse_args()

    print(f"Reading {args.input} (format: {args.in_format})...")

    tracks = Tracks()
    if args.in_format == "imilast":
        tracks = read_imilast(args.input)
    elif args.in_format == "json":
        tracks = read_json(args.input)

    # Track Type Detection / Override
    if args.type:
        tracks.track_type = args.type
    else:
        # Detect if unknown
        tracks.track_type = infer_track_type(tracks)

    # Sync internal variable naming if Intensity1 is present
    if "Intensity1" in tracks.vars and tracks.track_type != "unknown":
        tracks.vars[tracks.track_type] = tracks.vars.pop("Intensity1")

    print(f"Loaded {len(tracks)} tracks. Detected type: {tracks.track_type}")
    print(f"Writing to {args.output} (format: {args.out_format})...")

    if args.out_format == "html":
        generate_html(tracks, args.output, split=args.split)
    else:
        tracks.write(args.output, format=args.out_format)

    print("Done!")


if __name__ == "__main__":
    main()
