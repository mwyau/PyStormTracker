#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s CASE RUN_LABEL dask|serial\n' "$0" >&2
    exit 2
fi

CASE_NAME=$1
RUN_LABEL=$2
BACKEND=$3
: "${RESULT_BASE:=/home/albert/PyStormTracker-Validation/results/pst_track_comparison-20260819-corrected}"
: "${BENCHMARK_REPO:=/home/albert/PyStormTracker}"

case "$BACKEND" in
    dask) BACKEND_BASE=$RESULT_BASE ;;
    serial) BACKEND_BASE="$RESULT_BASE/serial" ;;
    *) printf 'unknown backend: %s\n' "$BACKEND" >&2; exit 2 ;;
esac

RESULT="$BACKEND_BASE/$CASE_NAME/$RUN_LABEL"
if [[ -e "$RESULT" ]]; then
    printf 'refusing to overwrite existing result directory: %s\n' "$RESULT" >&2
    exit 1
fi
mkdir -p "$RESULT"

TIME_FORMAT='wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_ARGS=(
    "$CASE_NAME"
    "$RUN_LABEL"
    --backend "$BACKEND"
    --result-base "$BACKEND_BASE"
)
if [[ -n "${FRAME_WORKERS:-}" ]]; then
    RUN_ARGS+=(--frame-workers "$FRAME_WORKERS")
fi
if [[ -n "${SHT_THREADS:-}" ]]; then
    RUN_ARGS+=(--sht-threads "$SHT_THREADS")
fi
if [[ -n "${MGE_WORKERS:-}" ]]; then
    RUN_ARGS+=(--mge-workers "$MGE_WORKERS")
fi

/usr/bin/time -f "$TIME_FORMAT" -o "$RESULT/workflow.time" \
    uv run python "$BENCHMARK_REPO/benchmarks/track_comparison/scripts/run_pst_repeat.py" \
    "${RUN_ARGS[@]}" \
    > "$RESULT/run.log" 2>&1
