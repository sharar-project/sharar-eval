#!/bin/bash
# Warm decode on llama.cpp, one fresh server per N: llama-server's slot count (-np) is fixed at start and N requests at
# once need N slots, so common/run.sh (one server for the whole suite) cannot run its N 1-32 row. For each N this starts
# the setting with exactly N slots of SLOT_TOKENS each (-c N x SLOT_TOKENS, shared with -kvu) and the setting's
# WARM_ARGS (-cram 0: the host prompt cache would take finished slot states out of their slots), runs sharar-eval's
# warm-decode at that N with the suite's own options, and stops it. The row ends at the first N whose server does not
# start or whose cell stops.
#
#   engines/llamacpp/warm.sh SETTING [N ...]            # default N: the suite's 1-32, up to the setting's WARM_N_MAX
#   CONTEXT=16k engines/llamacpp/warm.sh qwen3.8-27b-dflash2      # the published rows are 16k only
#
# SUITE (default cross-engine) and CONTEXT (default 16k) pick the cells; OUT resumes an earlier run's directory.
# Results: $OUT_ROOT/llamacpp/SETTING/<UTC time>-<host>-<context>/n<N>/: server.log, warm.txt (the cell's table) and
# run/ (sharar-eval's run directory); run.log for the whole row.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SETTING=${1:?usage: warm.sh SETTING [N ...]}; shift
. "$HERE/../common/defaults.env"; . "$HERE/../common/lib.sh"; . "$HERE/versions.env"
[ -f "$HERE/settings/$SETTING.env" ] || die "no setting '$SETTING': $(ls "$HERE/settings" | sed 's/\.env$//' | tr '\n' ' ')"
. "$HERE/settings/$SETTING.env"
CONTEXT=${CONTEXT:-16k}
export SHARAR_RESULTS PYTHONDONTWRITEBYTECODE=1
case " $AVOID_HOSTS " in *" $(hostname) "*)
  [ "${RUN_ON_ANY_HOST:-0}" = 1 ] || die "$(hostname) is in AVOID_HOSTS ($AVOID_HOSTS): benchmark on another Spark, or set RUN_ON_ANY_HOST=1" ;; esac
# the suite's warm-decode options as sharar-eval flags (its concurrency list becomes the default N)
eval "$(python3 -I - "$ROOT/evals/workloads/efficiency_v1.json" "$SUITE" <<'PY'
import json, shlex, sys
w = json.load(open(sys.argv[1]))["suites"][sys.argv[2]]["warm-decode"]
flags = []
for k, v in w.items():
    if k in ("context", "concurrency"):
        continue
    if v is True:
        flags.append("--" + k.replace("_", "-"))
    elif v is not False:
        flags += ["--" + k.replace("_", "-"), ",".join(map(str, v)) if isinstance(v, list) else str(v)]
print("SUITE_FLAGS=" + shlex.quote(" ".join(flags)))
print("SUITE_N=" + shlex.quote(" ".join(map(str, w["concurrency"]))))
PY
)"
NS=${*:-$(for n in $SUITE_N; do [ "$n" -le "${WARM_N_MAX:-32}" ] && echo "$n"; done | tr '\n' ' ')}

OUT=${OUT:-$OUT_ROOT/llamacpp/$SETTING/$(date -u +%Y%m%d-%H%M)-$(hostname)-$CONTEXT}
mkdir -p "$OUT"; LOG=$OUT/run.log
RESERVE_OWNER=${RESERVE_OWNER:-llamacpp-baseline}
reserve "$RESERVE_OWNER" "llama.cpp warm decode, $SETTING, $CONTEXT, N $NS"
CUR=
finish() { [ -n "$CUR" ] && bash "$HERE/serve.sh" down "$CUR" >> "$LOG" 2>&1; touch "$OUT/.done"; release; }
trap finish EXIT
rm -f "$OUT/.done"
nohup bash "$HERE/../common/dlog.sh" "$OUT/disk.log" 172800 "$OUT/.done" > /dev/null 2>&1 < /dev/null &
log "llama.cpp/$SETTING on $(hostname): suite $SUITE warm decode at $CONTEXT, N $NS -> $OUT"
CHECKS=
for n in $NS; do
  D=$OUT/n$n; mkdir -p "$D"
  if grep -qE "^ *$CONTEXT " "$D/warm.txt" 2>/dev/null; then log "N $n: done before, skipped"; CHECKS=--no-checks; continue; fi
  export LLAMA_SLOTS=$n LLAMA_CTX=$((n * SLOT_TOKENS)) LLAMA_ARGS=$WARM_ARGS
  . "$HERE/settings/$SETTING.env"                                # SETTING_DESC with this N's sizes
  log "N $n: -np $SLOTS -c $CTX_SIZE $LLAMA_ARGS, $(avail_gib) GiB free"
  bash "$HERE/serve.sh" up "$SETTING" "$D" >> "$LOG" 2>&1 || { log "N $n: llama.cpp did not start; the row ends"; break; }
  CUR=$D
  (cd "$ROOT" && uv run -q sharar-eval efficiency warm-decode --engine "$EVAL_ENGINE" \
     --engine-version "$ENGINE_LABEL; setting $SETTING: $SETTING_DESC ($(hostname))" --url "$URL" --out "$D/run" \
     --context "$CONTEXT" --concurrency "$n" $SUITE_FLAGS $CHECKS) >> "$D/warm.txt" 2>&1 || log "N $n: warm-decode exited non-zero"
  grep -E "^ *$CONTEXT " "$D/warm.txt" | tail -1 | tee -a "$LOG"
  grep -a "draft acceptance" "$D/server.log" | tail -1 >> "$LOG"
  bash "$HERE/serve.sh" down "$D" >> "$LOG" 2>&1; CUR=; CHECKS=--no-checks
  grep -qE "^ *$CONTEXT .*stopped" "$D/warm.txt" && { log "N $n stopped; the row ends"; break; }
done
log "done: $OUT"
