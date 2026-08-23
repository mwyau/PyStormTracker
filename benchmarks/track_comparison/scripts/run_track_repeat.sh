#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s CASE RUN_LABEL correctness|measure\n' "$0" >&2
    exit 2
fi

CASE_NAME=$1
RUN_LABEL=$2
MODE=$3

: "${TRACK_ROOT:=/home/albert/TRACK-run}"
: "${BENCHMARK_REPO:=/home/albert/PyStormTracker}"
: "${RESULT_BASE:=/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/repeats}"
: "${RUN_AT_JOB:=/tmp/codex-run-at.in}"

case "$CASE_NAME" in
    f320_to_t42_january)
        INPUT=era5_f320_jan_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/f320/specfilt_f320_to_t42"
        INITIAL=initial.f320_to_t42; NUMS=1,62,2 ;;
    f320_to_t42_full_year)
        INPUT=era5_f320_full_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/f320/specfilt_f320_to_t42"
        INITIAL=initial.f320_to_t42; NUMS=1,62,24 ;;
    f320_to_f320_january)
        INPUT=era5_f320_jan_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/f320/specfilt_f320_to_f320"
        INITIAL=initial.f320_to_f320; NUMS=1,62,2 ;;
    f320_to_f320_full_year)
        INPUT=era5_f320_full_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/f320/specfilt_f320_to_f320"
        INITIAL=initial.f320_to_f320; NUMS=1,62,24 ;;
    regular-2p5-dec)
        INPUT=era5_regular_2p5_dec_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/regular_latlon/specfilt_regular_2p5_T42"
        INITIAL=initial.T42; NUMS=1,62,2 ;;
    regular-2p5-season)
        INPUT=era5_regular_2p5_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/regular_latlon/specfilt_regular_2p5_T42"
        INITIAL=initial.T42; NUMS=1,62,6 ;;
    regular-0p25-dec)
        INPUT=era5_regular_0p25_dec_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/regular_latlon/specfilt_regular_0p25_T42"
        INITIAL=initial.T42; NUMS=1,62,2 ;;
    regular-0p25-season)
        INPUT=era5_regular_0p25_track.nc
        STREAM="$BENCHMARK_REPO/benchmarks/track_comparison/configs/regular_latlon/specfilt_regular_0p25_T42"
        INITIAL=initial.T42; NUMS=1,62,6 ;;
    *)
        printf 'unknown case: %s\n' "$CASE_NAME" >&2
        exit 2 ;;
esac

case "$MODE" in
    correctness|measure) ;;
    *)
        printf 'unknown mode: %s\n' "$MODE" >&2
        exit 2 ;;
esac

test -x "$TRACK_ROOT/bin/track.linux"
test -r "$STREAM"
test -r "$TRACK_ROOT/indat/$INPUT"
test -r "$RUN_AT_JOB"

EXT="rep_${CASE_NAME//-/_}_${RUN_LABEL}"
RESULT="$RESULT_BASE/$CASE_NAME/$RUN_LABEL"
if [[ -e "$RESULT" ]]; then
    printf 'refusing to overwrite existing result directory: %s\n' "$RESULT" >&2
    exit 1
fi
mkdir -p "$RESULT"

export PATH="$TRACK_ROOT:$TRACK_ROOT/bin:/tmp:$PATH"
export TRACK_LEGACY_DEPS=0
cd "$TRACK_ROOT"

rm -f "outdat/specfil.${EXT}_band000" "outdat/specfil.${EXT}_band001"
rm -f "indat/${EXT}.dat" ".run_at.lock.${EXT}" "RUN_${EXT}"

TIME_FORMAT='wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x'

run_spectral() {
    if [[ "$MODE" == measure ]]; then
        /usr/bin/time -f "$TIME_FORMAT" -o "$RESULT/spectral.time" \
            bin/track.linux -i "$INPUT" -f "$EXT" \
            < "$STREAM" > "$RESULT/spectral.log" 2>&1
    else
        bin/track.linux -i "$INPUT" -f "$EXT" \
            < "$STREAM" > "$RESULT/spectral.log" 2>&1
    fi
}

run_tracking() {
    if [[ "$MODE" == measure ]]; then
        /usr/bin/time -f "$TIME_FORMAT" -o "$RESULT/tracking.time" \
            ./master -excn=track.linux -fext="$EXT" -inpf="${EXT}.dat" \
            -jd="$RUN_AT_JOB" -kinit="$INITIAL" -nums="$NUMS" \
            -outdir="$RESULT/tracking" -cdir="$EXT" -rfil=RUN_ \
            -s=RUNDATIN.era5_MSLP_latlng \
            > "$RESULT/tracking/master.log" 2>&1
    else
        ./master -excn=track.linux -fext="$EXT" -inpf="${EXT}.dat" \
            -jd="$RUN_AT_JOB" -kinit="$INITIAL" -nums="$NUMS" \
            -outdir="$RESULT/tracking" -cdir="$EXT" -rfil=RUN_ \
            -s=RUNDATIN.era5_MSLP_latlng \
            > "$RESULT/tracking/master.log" 2>&1
    fi
}

if [[ "$MODE" == measure ]]; then
    start_ns=$(date +%s%N)
fi
run_spectral
test -s "outdat/specfil.${EXT}_band000"
test -s "outdat/specfil.${EXT}_band001"
ln -s "$TRACK_ROOT/outdat/specfil.${EXT}_band001" "indat/${EXT}.dat"
mkdir -p "$RESULT/tracking"
run_tracking

if [[ "$MODE" == measure ]]; then
    end_ns=$(date +%s%N)
    awk -v start="$start_ns" -v end="$end_ns" \
        'BEGIN { printf "elapsed_wall_seconds=%.6f\n", (end-start)/1000000000.0 }' \
        > "$RESULT/total.time"
fi

printf 'case=%s\nrun=%s\nmode=%s\ninput=%s\nstream=%s\ninitial=%s\nnums=%s\next=%s\n' \
    "$CASE_NAME" "$RUN_LABEL" "$MODE" "$TRACK_ROOT/indat/$INPUT" "$STREAM" "$INITIAL" "$NUMS" "$EXT" \
    > "$RESULT/metadata.txt"

for sign in neg pos; do
    for product in tr_trs ff_trs; do
        product_path=$(find "$RESULT/tracking" -type f -name "${product}_${sign}*" -print -quit)
        test -n "$product_path"
        product_size=$(stat -c '%s' "$product_path")
        if [[ "$product_path" == *.gz ]]; then
            product_stream=(zcat "$product_path")
        else
            product_stream=(cat "$product_path")
        fi
        track_count=$("${product_stream[@]}" | awk '$1 == "TRACK_NUM" && count == "" { count = $2 } END { print count + 0 }')
        point_count=$("${product_stream[@]}" | awk '$1 == "POINT_NUM" { total += $2 } END { print total + 0 }')
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$product" "$sign" "$product_path" "$product_size" "$track_count/$point_count" \
            >> "$RESULT/products.tsv"
    done
done

rm -f "outdat/specfil.${EXT}_band000" "outdat/specfil.${EXT}_band001" "indat/${EXT}.dat"
printf 'validated=1\n' > "$RESULT/status.txt"
