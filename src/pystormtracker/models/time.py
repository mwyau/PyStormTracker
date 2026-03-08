from dataclasses import dataclass


@dataclass
class TimeRange:
    """Metadata for the time range covered by a set of tracks."""

    start: float
    end: float
    step: float | None = None
