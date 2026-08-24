from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from io import StringIO
from unittest.mock import Mock, call, patch

import pytest

from pystormtracker.backends import (
    defer_dask_interrupt_cleanup,
    drain_pending_dask_executors,
    local_dask_executor,
)
from pystormtracker.cli import _emergency_interrupt, main
from pystormtracker.hodges.progress import (
    HodgesDaskProgress,
    hodges_dask_progress,
    hodges_dask_progress_enabled,
)
from pystormtracker.utils.logging import _TerminalLoggingHandler, configure_cli_logging


def test_cli_logging_levels() -> None:
    root = logging.getLogger()
    old_level = root.level
    old_handlers = root.handlers[:]
    try:
        configure_cli_logging(0)
        assert root.level == logging.WARNING
        configure_cli_logging(1)
        assert root.level == logging.INFO
        configure_cli_logging(2)
        assert root.level == logging.DEBUG
    finally:
        for handler in root.handlers[:]:
            if handler not in old_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in old_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(old_level)


def test_cli_parses_global_verbose_and_long_only_track_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = Mock()
    monkeypatch.setattr("pystormtracker.cli.track.main", called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stormtracker",
            "-vv",
            "track",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "output.trackjson",
        ],
    )

    main()

    args = called.call_args.args[0]
    assert args.verbose == 2
    assert args.variable == "msl"


def test_progress_is_disabled_for_non_tty_and_override_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    assert hodges_dask_progress_enabled() is False
    with hodges_dask_progress(True):
        assert hodges_dask_progress_enabled() is True
    with hodges_dask_progress(False):
        assert hodges_dask_progress_enabled() is False


def test_progress_counts_prepared_detection_and_mge_tasks() -> None:
    progress = HodgesDaskProgress(
        total_frames=2,
        total_segments=1,
        frame_workers=1,
        mge_workers=1,
        stream=StringIO(),
    )
    progress._start(None)
    progress._posttask("hodges-prepared-frame-000000", object(), None, None, None)
    progress._posttask("unrelated-source-000000", object(), None, None, None)
    snapshot = progress.snapshot()
    assert snapshot.prepared_frames == 1
    assert snapshot.completed_frames == 0
    assert snapshot.completed_segments == 0

    progress._posttask("hodges-frame-000000", object(), None, None, None)
    progress._posttask("hodges-mge-segment-000000", object(), None, None, None)
    snapshot = progress.snapshot()
    assert snapshot.prepared_frames == 1
    assert snapshot.completed_frames == 1
    assert snapshot.completed_segments == 1
    progress.interrupt()


def test_logging_record_is_separated_from_live_progress() -> None:
    output = StringIO()
    progress = HodgesDaskProgress(
        total_frames=1,
        total_segments=1,
        frame_workers=1,
        mge_workers=1,
        stream=output,
    )
    progress._start(None)

    handler = _TerminalLoggingHandler(output)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger = logging.getLogger("pystormtracker.test-terminal")
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate
    try:
        logger.handlers[:] = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info("phase boundary")
    finally:
        logger.handlers[:] = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        handler.close()
        progress.interrupt()

    text = output.getvalue()
    assert "INFO: phase boundary\n" in text
    assert "| Active" in text
    assert "boundary\r" not in text


def test_first_interrupt_exits_130_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stormtracker",
            "track",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "output.trackjson",
        ],
    )
    monkeypatch.setattr(
        "pystormtracker.cli.track.main", Mock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 130
    assert "Interrupted; cancelling pending work." in captured.err
    assert "Press Ctrl-C again" in captured.err
    assert "Traceback" not in captured.err


def test_second_interrupt_uses_emergency_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    emergency = Mock()
    monkeypatch.setattr("pystormtracker.cli.os._exit", emergency)
    _emergency_interrupt(2, None)
    emergency.assert_called_once_with(130)


def test_interrupted_dask_executor_cancels_pending_futures() -> None:
    executor = Mock()
    with defer_dask_interrupt_cleanup():
        with (
            patch("pystormtracker.backends.ThreadPoolExecutor", return_value=executor),
            pytest.raises(KeyboardInterrupt),
            local_dask_executor(1),
        ):
            raise KeyboardInterrupt
        drain_pending_dask_executors()
    assert executor.shutdown.call_args_list == [
        call(wait=False, cancel_futures=True),
        call(wait=True, cancel_futures=False),
    ]


def test_library_dask_exception_waits_for_running_work() -> None:
    import dask

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    failure_seen = threading.Event()

    def running_work() -> None:
        started.set()
        release.wait(timeout=2.0)
        finished.set()

    def failing_work() -> None:
        assert started.wait(timeout=2.0)
        failure_seen.set()
        raise RuntimeError("library failure")

    def release_work() -> None:
        assert failure_seen.wait(timeout=2.0)
        time.sleep(0.05)
        release.set()

    releaser = threading.Thread(target=release_work)
    releaser.start()
    with pytest.raises(RuntimeError, match="library failure"), local_dask_executor(2):
        dask.compute(
            dask.delayed(failing_work)(),
            dask.delayed(running_work)(),
            scheduler="threads",
        )  # type: ignore[no-untyped-call]
    releaser.join(timeout=2.0)
    assert finished.is_set()


def test_import_does_not_configure_root_logging() -> None:
    code = (
        "import logging;\n"
        "before=(logging.getLogger().level, "
        "len(logging.getLogger().handlers));\n"
        "import pystormtracker;\n"
        "after=(logging.getLogger().level, "
        "len(logging.getLogger().handlers));\n"
        "print(before, after)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "(30, 0) (30, 0)"
