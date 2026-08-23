"""Scheduler-side semantic progress for local Dask Hodges tracking."""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TextIO

from dask.callbacks import Callback

from ..utils.logging import register_progress, terminal_lock, unregister_progress
from .detections import HodgesCenterFrame

_PROGRESS_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "pystormtracker_hodges_dask_progress", default=None
)
_PREPARED_KEY_PREFIX = "hodges-prepared-frame-"
_FRAME_KEY_PREFIX = "hodges-frame-"
_SEGMENT_KEY_PREFIX = "hodges-mge-segment-"


@contextmanager
def hodges_dask_progress(enabled: bool | None) -> Iterator[None]:
    """Temporarily override interactive Dask progress for Hodges tracking."""
    token = _PROGRESS_OVERRIDE.set(enabled)
    try:
        yield
    finally:
        _PROGRESS_OVERRIDE.reset(token)


def hodges_dask_progress_enabled() -> bool:
    """Return whether this process should render interactive Hodges progress."""
    override = _PROGRESS_OVERRIDE.get()
    if override is not None:
        return override
    return sys.stderr.isatty()


@dataclass(frozen=True, slots=True)
class HodgesDaskProgressSnapshot:
    """One immutable view of scheduler-observed Hodges Dask work."""

    prepared_frames: int
    total_prepared_frames: int
    completed_frames: int
    total_frames: int
    highest_contiguous_frame: int
    completed_segments: int
    total_segments: int
    detected_features: int
    elapsed_seconds: float
    recent_frames_per_second: float | None
    eta_seconds: float | None
    active_phase: str


class HodgesDaskProgress(Callback):
    """Render semantic preprocessing, detection, and MGE progress.

    Only tracker-owned task-key prefixes affect semantic counters.  Dask's
    unrelated source and internal tasks can still cause a throttled elapsed
    time refresh, which keeps the display alive during lazy preprocessing.
    """

    def __init__(
        self,
        *,
        total_frames: int,
        total_segments: int,
        frame_workers: int,
        mge_workers: int,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        refresh_seconds: float = 1.0,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if total_segments <= 0:
            raise ValueError("total_segments must be positive")
        if frame_workers <= 0:
            raise ValueError("frame_workers must be positive")
        if mge_workers <= 0:
            raise ValueError("mge_workers must be positive")
        if refresh_seconds <= 0.0:
            raise ValueError("refresh_seconds must be positive")

        self.total_frames = total_frames
        self.total_segments = total_segments
        self.frame_workers = frame_workers
        self.mge_workers = mge_workers
        self._stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self._refresh_seconds = refresh_seconds
        self._lock = terminal_lock()
        self._prepared_frame_indices: set[int] = set()
        self._completed_frame_indices: set[int] = set()
        self._completed_segment_indices: set[int] = set()
        self._highest_contiguous_frame = -1
        self._detected_features = 0
        self._samples: deque[tuple[float, int]] = deque()
        self._started_at: float | None = None
        self._last_refresh_at: float | None = None
        self._last_line_width = 0
        self._line_active = False
        self._detection_complete_announced = False
        self._active_phase = "preprocessing"

    def _start(self, dsk: object) -> None:
        """Initialize progress at the scheduler's graph-start hook."""
        del dsk
        with self._lock:
            register_progress(self)
            self._started_at = self._clock()
            self._last_refresh_at = None
            self._write(
                "Hodges tracking: "
                f"{self.total_frames} frames | {self.total_segments} MGE segments "
                f"| Dask frame_workers={self.frame_workers} "
                f"mge_workers={self.mge_workers}\n"
            )
            self._refresh_locked(force=True)

    def _pretask(self, key: object, dsk: object, state: object) -> None:
        """Record active semantic phase without counting arbitrary tasks."""
        del dsk, state
        with self._lock:
            if (
                self._task_index(key, _PREPARED_KEY_PREFIX, self.total_frames)
                is not None
            ):
                self._active_phase = "preprocessing"
            elif (
                self._task_index(key, _FRAME_KEY_PREFIX, self.total_frames) is not None
            ):
                self._active_phase = "detection/refinement"
            elif (
                self._task_index(key, _SEGMENT_KEY_PREFIX, self.total_segments)
                is not None
            ):
                self._active_phase = "MGE"
            self._refresh_locked(force=False)

    def _posttask(
        self,
        key: object,
        result: object,
        dsk: object,
        state: object,
        worker_id: object,
    ) -> None:
        """Count only completed named tracker tasks."""
        del dsk, state, worker_id
        prepared_index = self._task_index(key, _PREPARED_KEY_PREFIX, self.total_frames)
        frame_index = self._task_index(key, _FRAME_KEY_PREFIX, self.total_frames)
        segment_index = self._task_index(key, _SEGMENT_KEY_PREFIX, self.total_segments)
        if prepared_index is None and frame_index is None and segment_index is None:
            with self._lock:
                self._refresh_locked(force=False)
            return

        with self._lock:
            if prepared_index is not None:
                self._prepared_frame_indices.add(prepared_index)
                self._active_phase = "detection/refinement"
            elif (
                frame_index is not None
                and frame_index not in self._completed_frame_indices
            ):
                self._completed_frame_indices.add(frame_index)
                self._active_phase = "detection/refinement"
                if isinstance(result, HodgesCenterFrame):
                    self._detected_features += int(result.values.size)
                self._advance_contiguous_frame_locked()
                self._record_frame_sample_locked()
                if (
                    len(self._completed_frame_indices) == self.total_frames
                    and not self._detection_complete_announced
                ):
                    self._end_status_line_locked()
                    self._write("Detection/refinement complete\n")
                    self._detection_complete_announced = True
            elif (
                segment_index is not None
                and segment_index not in self._completed_segment_indices
            ):
                self._completed_segment_indices.add(segment_index)
                self._active_phase = "MGE"
            self._refresh_locked(force=False)

    def _finish(self, dsk: object, state: object, failed: bool) -> None:
        """Release renderer resources without classifying the Dask outcome."""
        del dsk, state, failed
        with self._lock:
            self._end_status_line_locked()
            unregister_progress(self)

    @staticmethod
    def _task_index(key: object, prefix: str, limit: int) -> int | None:
        """Parse one stable Dask task key without accepting incidental keys."""
        if not isinstance(key, str) or not key.startswith(prefix):
            return None
        suffix = key.removeprefix(prefix)
        if not suffix.isdecimal():
            return None
        index = int(suffix)
        return index if 0 <= index < limit else None

    def _advance_contiguous_frame_locked(self) -> None:
        """Advance the ordered frontier only across a gap-free prefix."""
        candidate = self._highest_contiguous_frame + 1
        while candidate in self._completed_frame_indices:
            self._highest_contiguous_frame = candidate
            candidate += 1

    def _record_frame_sample_locked(self) -> None:
        """Keep a short wall-clock window for throughput estimation."""
        now = self._clock()
        self._samples.append((now, len(self._completed_frame_indices)))
        cutoff = now - 15.0
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def snapshot(self) -> HodgesDaskProgressSnapshot:
        """Return semantic progress suitable for tests or renderers."""
        with self._lock:
            elapsed = self._elapsed_seconds_locked()
            rate = self._recent_frame_rate_locked()
            remaining = self.total_frames - len(self._completed_frame_indices)
            eta = remaining / rate if rate is not None and rate > 0.0 else None
            return HodgesDaskProgressSnapshot(
                prepared_frames=len(self._prepared_frame_indices),
                total_prepared_frames=self.total_frames,
                completed_frames=len(self._completed_frame_indices),
                total_frames=self.total_frames,
                highest_contiguous_frame=self._highest_contiguous_frame,
                completed_segments=len(self._completed_segment_indices),
                total_segments=self.total_segments,
                detected_features=self._detected_features,
                elapsed_seconds=elapsed,
                recent_frames_per_second=rate,
                eta_seconds=eta,
                active_phase=self._active_phase,
            )

    def interrupt(self) -> None:
        """End the live line for an outer CLI interruption report."""
        with self._lock:
            self._end_status_line_locked()
            unregister_progress(self)

    def interrupted(self) -> None:
        """Report interruption without classifying it as a failure."""
        with self._lock:
            self._end_status_line_locked()
            self._write("Hodges tracking interrupted\n")
            unregister_progress(self)

    def failed(self) -> None:
        """Report a failed Dask phase when the outer tracker has the exception."""
        with self._lock:
            self._end_status_line_locked()
            self._write("Hodges tracking failed\n")
            unregister_progress(self)

    def mge_complete(self) -> None:
        """Render the completed Dask phase before deterministic splicing."""
        with self._lock:
            self._end_status_line_locked()
            self._write("MGE complete\n")

    def splicing_segments(self) -> None:
        """Announce the serial deterministic splice phase."""
        with self._lock:
            self._write("Splicing segments...\n")

    def applying_postfilters(self) -> None:
        """Announce post-link filtering after deterministic splicing."""
        with self._lock:
            self._write("Applying postfilters...\n")

    def done(self, track_count: int, point_count: int) -> None:
        """Render the final concise completion summary."""
        with self._lock:
            self._end_status_line_locked()
            elapsed = self._elapsed_seconds_locked()
            self._write(
                f"Done: {track_count} tracks / {point_count} points | "
                f"elapsed {elapsed:.1f}s\n"
            )

    def _refresh_locked(self, *, force: bool) -> None:
        """Render at most once per refresh interval while the graph runs."""
        now = self._clock()
        if (
            not force
            and self._last_refresh_at is not None
            and now - self._last_refresh_at < self._refresh_seconds
        ):
            return
        self._last_refresh_at = now
        snapshot = self.snapshot()
        highest = (
            str(snapshot.highest_contiguous_frame + 1)
            if snapshot.highest_contiguous_frame >= 0
            else "none"
        )
        frame_percentage = 100.0 * snapshot.completed_frames / snapshot.total_frames
        segment_percentage = (
            100.0 * snapshot.completed_segments / snapshot.total_segments
        )
        rate = (
            f"{snapshot.recent_frames_per_second:.1f} frame/s"
            if snapshot.recent_frames_per_second is not None
            else "rate estimating"
        )
        eta = self._format_eta(snapshot.eta_seconds)
        line = (
            f"Prepared {snapshot.prepared_frames}/{snapshot.total_prepared_frames} | "
            f"Frames {snapshot.completed_frames}/{snapshot.total_frames} "
            f"[{frame_percentage:5.1f}%] | through {highest} | {rate} | ETA {eta} "
            f"| Features {snapshot.detected_features} | MGE "
            f"{snapshot.completed_segments}/{snapshot.total_segments} "
            f"[{segment_percentage:5.1f}%] | Elapsed "
            f"{self._format_elapsed(snapshot.elapsed_seconds)} | Active "
            f"{snapshot.active_phase}"
        )
        self._stream.write("\r" + line.ljust(self._last_line_width))
        self._stream.flush()
        self._last_line_width = len(line)
        self._line_active = True

    @staticmethod
    def _format_eta(seconds: float | None) -> str:
        """Format an optional approximate ETA."""
        if seconds is None or not (seconds >= 0.0):
            return "estimating"
        rounded = round(seconds)
        minutes, remaining_seconds = divmod(rounded, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        return f"{minutes}m{remaining_seconds:02d}s"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed wall time with compact units."""
        return HodgesDaskProgress._format_eta(seconds)

    def _elapsed_seconds_locked(self) -> float:
        """Return elapsed display time."""
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    def _recent_frame_rate_locked(self) -> float | None:
        """Compute a rate from the retained sample window."""
        if len(self._samples) < 2:
            return None
        start_time, start_count = self._samples[0]
        end_time, end_count = self._samples[-1]
        duration = end_time - start_time
        if duration <= 0.0:
            return None
        return (end_count - start_count) / duration

    def _end_status_line_locked(self) -> None:
        """Terminate a carriage-return display before a phase message."""
        if self._line_active:
            self._stream.write("\n")
            self._stream.flush()
            self._line_active = False

    def suspend_for_log(self) -> None:
        """Clear the current carriage-return line before a log record."""
        if self._line_active:
            self._stream.write("\r" + (" " * self._last_line_width) + "\r")
            self._stream.flush()

    def resume_after_log(self) -> None:
        """Redraw the latest state after a log record."""
        if self._line_active:
            self._refresh_locked(force=True)

    def _write(self, text: str) -> None:
        """Write a phase transition to the progress stream."""
        self._stream.write(text)
        self._stream.flush()
