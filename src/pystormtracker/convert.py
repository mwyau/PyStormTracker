"""Track-format conversion and temporary HTML placeholder output."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

from .io.format import SUPPORTED_FORMATS, load_tracks
from .models.tracks import Tracks
from .models.units import canonical_unit_for, resolve_mode


def generate_html(outfile: str | Path) -> None:
    """Write the temporary static explorer placeholder."""
    warnings.warn(
        "HTML explorer output is temporarily a static placeholder; "
        "track data was not embedded.",
        stacklevel=2,
    )
    template_path = Path(__file__).parent / "templates" / "explorer.html"
    if not template_path.exists():
        raise FileNotFoundError(f"HTML template not found at {template_path}")
    Path(outfile).write_text(
        template_path.read_text(encoding="utf-8"), encoding="utf-8"
    )


def _rename_primary_variable(
    tracks: Tracks,
    target: str,
    requested_unit: str | None,
) -> Tracks:
    variables = dict(tracks.variables)
    units = dict(tracks.units)
    if target in variables:
        source_name = target
    elif "Intensity1" in variables:
        source_name = "Intensity1"
    elif len(variables) == 1:
        source_name = next(iter(variables))
    else:
        raise ValueError(
            f"cannot select variable {target!r}; source has multiple variables and "
            "the requested target is absent"
        )
    if source_name != target:
        variables[target] = variables.pop(source_name)
        source_unit = units.pop(source_name, None)
    else:
        source_unit = units.get(source_name)
    unit = requested_unit or canonical_unit_for(target) or source_unit
    if unit is None or (
        unit == "1" and canonical_unit_for(target) is None and requested_unit is None
    ):
        raise ValueError(
            f"unit for renamed variable {target!r} cannot be established; "
            "provide --unit"
        )
    units[target] = unit
    metadata = replace(tracks.metadata, primary_var=target, units=units)
    return tracks.with_variables(variables, metadata=metadata)


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Set up the trajectory conversion command."""
    parser = subparsers.add_parser(
        "convert",
        description=(
            "Convert supported trajectory formats or write an HTML placeholder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-f",
        "--in-format",
        choices=["auto", *SUPPORTED_FORMATS],
        default="auto",
        help="Input format; inferred from the extension by default.",
    )
    parser.add_argument(
        "-F",
        "--out-format",
        choices=["auto", *SUPPORTED_FORMATS, "html"],
        default="auto",
        help="Output format; inferred from the extension by default.",
    )
    parser.add_argument("-v", "--variable", help="Override the primary variable name")
    parser.add_argument(
        "--unit", help="Unit for a renamed or otherwise ambiguous variable"
    )
    parser.add_argument(
        "-m",
        "--detection-mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Extremum mode for the final primary variable.",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """Convert a trajectory file using extension or explicitly selected formats."""
    in_format = None if args.in_format == "auto" else args.in_format
    out_format = None if args.out_format == "auto" else args.out_format
    tracks = load_tracks(
        args.input,
        format=in_format,
        primary_var=args.variable,
        mode=args.detection_mode,
    )
    if args.variable:
        tracks = _rename_primary_variable(tracks, args.variable, args.unit)
    final_mode = resolve_mode(tracks.primary_var, args.detection_mode)
    if final_mode != tracks.mode:
        tracks = tracks.with_metadata(replace(tracks.metadata, mode=final_mode))
    if out_format == "html" or (
        out_format is None and Path(args.output).suffix.lower() == ".html"
    ):
        generate_html(args.output)
    else:
        tracks.write(args.output, format=out_format)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    subparsers = argument_parser.add_subparsers()
    setup_parser(subparsers)
    namespace = argument_parser.parse_args()
    if hasattr(namespace, "func"):
        namespace.func(namespace)
