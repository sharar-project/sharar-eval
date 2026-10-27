#!/bin/bash
# One engine baseline on this Spark, end to end: reserve the Spark, start the engine in one of its settings, run
# sharar-eval's efficiency suite against it, stop the engine and release the Spark.
#
#   engines/common/run.sh ENGINE SETTING [SUITE]
#   engines/common/run.sh vllm nemotron-3.5-bolt          # example
#
# ENGINE is a folder of engines/ (vllm, sharar); SETTING a file in its settings/ without .env, named after the model;
# SUITE a suite of evals/workloads/efficiency_v1.json (default cross-engine; smoke for a few-minute check).
# FLUSH_RESTART=1 restarts the engine before every warm-decode burst (sharar-eval's --flush-cmd), for an engine with
# no cache flush endpoint. OUT=<an earlier run's directory> resumes it: finished cells are skipped.
# Results: $OUT_ROOT/ENGINE/SETTING/<UTC time>-<host>/: run.log, the engine's logs, disk.log (SSD reads, GPU use, SM
# clock, free memory every 2 s), sampling-probe.json, suite/ (sharar-eval's run directory) and report.txt.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)                      # the sharar-eval checkout
ENGINE=${1:?usage: run.sh ENGINE SETTING [SUITE]}
SETTING=${2:?usage: run.sh ENGINE SETTING [SUITE]}
E=$HERE/../$ENGINE
. "$HERE/defaults.env"; . "$HERE/lib.sh"
SUITE=${3:-$SUITE}
[ -f "$E/serve.sh" ] || die "no engine folder engines/$ENGINE"
[ -f "$E/settings/$SETTING.env" ] || die "no setting '$SETTING' for $ENGINE: $(ls "$E/settings" | sed 's/\.env$//' | tr '\n' ' ')"
. "$E/versions.env"; . "$E/settings/$SETTING.env"
export SHARAR_RESULTS PYTHONDONTWRITEBYTECODE=1
command -v uv > /dev/null || die "uv is not on PATH (sharar-eval runs through uv run): add its directory to PATH"
case " $AVOID_HOSTS " in *" $(hostname) "*)
  [ "${RUN_ON_ANY_HOST:-0}" = 1 ] || die "$(hostname) is in AVOID_HOSTS ($AVOID_HOSTS): benchmark on another Spark, or set RUN_ON_ANY_HOST=1" ;; esac

OUT=${OUT:-$OUT_ROOT/$ENGINE/$SETTING/$(date -u +%Y%m%d-%H%M)-$(hostname)}
mkdir -p "$OUT"; LOG=$OUT/run.log
RESERVE_OWNER=${RESERVE_OWNER:-$ENGINE-baseline}
reserve "$RESERVE_OWNER" "$ENGINE baseline, $SETTING, suite $SUITE"
UP=0
finish() { [ $UP = 1 ] && bash "$E/serve.sh" down "$OUT" >> "$LOG" 2>&1; touch "$OUT/.done"; release; }
trap finish EXIT
rm -f "$OUT/.done"
nohup bash "$HERE/dlog.sh" "$OUT/disk.log" 172800 "$OUT/.done" > /dev/null 2>&1 < /dev/null &
VERSION="$ENGINE_LABEL; setting $SETTING: $SETTING_DESC ($(hostname))"
log "$ENGINE/$SETTING on $(hostname), $(avail_gib) GiB free: suite $SUITE -> $OUT"
bash "$E/serve.sh" up "$SETTING" "$OUT" >> "$LOG" 2>&1 || die "$ENGINE did not start (see $OUT)"
UP=1
log "$ENGINE serving at $URL, $(avail_gib) GiB free"
log "sampling probe: $(sampling_probe "$URL" "$OUT/sampling-probe.json")"
FLUSH=()
[ "${FLUSH_RESTART:-0}" = 1 ] && FLUSH=(--flush-cmd "bash $E/serve.sh down $OUT >> $LOG 2>&1; bash $E/serve.sh up $SETTING $OUT >> $LOG 2>&1")
(cd "$ROOT" && uv run -q sharar-eval efficiency suite --suite "$SUITE" --engine "$EVAL_ENGINE" --engine-version "$VERSION" \
   --url "$URL" --out "$OUT/suite" "${FLUSH[@]}") >> "$OUT/suite.txt" 2>&1 || log "the suite exited non-zero (see suite.txt)"
(cd "$ROOT" && uv run -q sharar-eval efficiency report "$OUT/suite") > "$OUT/report.txt" 2>&1
cat "$OUT/report.txt" >> "$LOG"
log "done: $OUT"
