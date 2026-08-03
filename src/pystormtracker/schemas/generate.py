"""Generate the committed TrackJSON schema from the typed wire model."""

from __future__ import annotations

import json
from typing import cast

import msgspec

from ..io.trackjson import TrackJSONDocument

_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_TITLE = "TrackJSON v1.0"


def generate_trackjson_schema() -> dict[str, object]:
    """Return the deterministic JSON Schema generated from msgspec types."""
    generated = cast(dict[str, object], msgspec.json.schema(TrackJSONDocument))
    return {
        "$schema": _SCHEMA_URI,
        **generated,
        "title": _SCHEMA_TITLE,
    }


def encode_trackjson_schema() -> bytes:
    """Return deterministic UTF-8 schema bytes with a trailing newline."""
    return (
        json.dumps(generate_trackjson_schema(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
