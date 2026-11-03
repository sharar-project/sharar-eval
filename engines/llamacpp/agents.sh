#!/bin/bash
# Agent replay on llama.cpp, one fresh server per agent count. Every agent's conversation stays in its slot (the host
# prompt cache is off, -cram 0), so the shared pool (-kvu) must hold all of them: AGENT_POOLS in the setting gives the
# pool per level, and AGENT_SLOTS_MAX caps the slots where the box cannot hold one per agent (the other agents then
# wait for a slot, like TensorFold's 16 streams). The setting's AGENT_ARGS keep 3 context checkpoints a slot (-ctxcp 3):
# with fewer, llama.cpp cannot step back before the SWE-bench turn message and re-reads every SWE-bench prompt.
#
#   engines/llamacpp/agents.sh SETTING [AGENTS ...]      # default 1 8 16 32, the cross-engine-agents levels
#   REPLAY=~/.local/share/sharar/replays/agentperf-default-v1-nonote engines/llamacpp/agents.sh SETTING 1 8 16
#
# The suite's agent-replay options (cross-engine-agents: the thinking-off setting, tool time none); REPLAY replays
# another copy of the traces (default: the pinned AA traces as published). Results: $OUT_ROOT/llamacpp-agents/
# SETTING/<UTC time>-<host>/a<AGENTS>/: server.log, agents.txt and agents/ (sharar-eval's run directory).
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SETTING=${1:?usage: agents.sh SETTING [AGENTS ...]}; shift
. "$HERE/../common/defaults.env"; . "$HERE/../common/lib.sh"; . "$HERE/versions.env"
[ -f "$HERE/settings/$SETTING.env" ] || die "no setting '$SETTING': $(ls "$HERE/settings" | sed 's/\.env$//' | tr '\n' ' ')"
. "$HERE/settings/$SETTING.env"
export SHARAR_RESULTS PYTHONDONTWRITEBYTECODE=1
case " $AVOID_HOSTS " in *" $(hostname) "*)
  [ "${RUN_ON_ANY_HOST:-0}" = 1 ] || die "$(hostname) is in AVOID_HOSTS ($AVOID_HOSTS): benchmark on another Spark, or set RUN_ON_ANY_HOST=1" ;; esac
eval "$(python3 -I - "$ROOT/evals/workloads/efficiency_v1.json" <<'PY'
import json, shlex, sys
w = json.load(open(sys.argv[1]))["suites"]["cross-engine-agents"]["agent-replay"]
flags = sum((["--" + k.replace("_", "-"), str(v)] for k, v in w.items() if k != "agents"), [])
print("SUITE_FLAGS=" + shlex.quote(" ".join(flags)))
print("SUITE_AGENTS=" + shlex.quote(" ".join(map(str, w["agents"]))))
PY
)"
LEVELS=${*:-$SUITE_AGENTS}
pool() { for p in $AGENT_POOLS; do [ "${p%%:*}" = "$1" ] && { echo "${p#*:}"; return; }; done; echo $((32768 * $1)); }

OUT=${OUT:-$OUT_ROOT/llamacpp-agents/$SETTING/$(date -u +%Y%m%d-%H%M)-$(hostname)${REPLAY:+-$(basename "$REPLAY")}}
mkdir -p "$OUT"; LOG=$OUT/run.log
RESERVE_OWNER=${RESERVE_OWNER:-llamacpp-baseline}
reserve "$RESERVE_OWNER" "llama.cpp agent replay, $SETTING, agents $LEVELS"
CUR=
finish() { [ -n "$CUR" ] && bash "$HERE/serve.sh" down "$CUR" >> "$LOG" 2>&1; touch "$OUT/.done"; release; }
trap finish EXIT
rm -f "$OUT/.done"
nohup bash "$HERE/../common/dlog.sh" "$OUT/disk.log" 172800 "$OUT/.done" > /dev/null 2>&1 < /dev/null &
log "llama.cpp/$SETTING on $(hostname): agent replay at $LEVELS agents, ${REPLAY:-AA traces as published} -> $OUT"
for n in $LEVELS; do
  D=$OUT/a$n; mkdir -p "$D"
  if grep -qE "^ +$n +[0-9]+ " "$D/agents.txt" 2>/dev/null; then log "$n agents: done before, skipped"; continue; fi
  slots=$(( n < AGENT_SLOTS_MAX ? n : AGENT_SLOTS_MAX ))
  export LLAMA_SLOTS=$slots LLAMA_CTX=$(pool "$n") LLAMA_ARGS=$AGENT_ARGS
  . "$HERE/settings/$SETTING.env"
  log "$n agents: -np $SLOTS -c $CTX_SIZE $LLAMA_ARGS, $(avail_gib) GiB free$([ "$slots" -lt "$n" ] && echo "; $((n - slots)) agents wait for a slot")"
  bash "$HERE/serve.sh" up "$SETTING" "$D" >> "$LOG" 2>&1 || { log "$n agents: llama.cpp did not start"; continue; }
  CUR=$D
  (cd "$ROOT" && uv run -q sharar-eval efficiency agent-replay --engine "$EVAL_ENGINE" \
     --engine-version "$ENGINE_LABEL; setting $SETTING: $SETTING_DESC ($(hostname))" --url "$URL" --out "$D/agents" \
     --agents "$n" $SUITE_FLAGS ${REPLAY:+--replay "$REPLAY"}) 2>&1 | tee "$D/agents.txt" >> "$LOG"
  echo "context full: $(grep -ac 'Context size has been exceeded' "$D/server.log") requests" | tee -a "$LOG"
  bash "$HERE/serve.sh" down "$D" >> "$LOG" 2>&1; CUR=
done
log "done: $OUT"
