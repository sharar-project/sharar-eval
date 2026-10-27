# Helpers for run.sh and the engines' serve.sh scripts. Source it; it sets nothing else.

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${LOG:-/dev/null}" >&2; }
die() { log "ERROR: $*"; exit 1; }
avail_gib() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
gpu_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

# The Spark's reservation: ~/.sp-claim, the file the `sp` driver uses (owner, UTC start, note, tab-separated; created
# only if absent). reserve takes it when it is free and the GPU idle, or keeps it when OWNER already holds it;
# release drops it only if this script took it.
CLAIM_FILE=$HOME/.sp-claim
reserve() {  # OWNER NOTE
  local o=""
  [ -f "$CLAIM_FILE" ] && IFS=$'\t' read -r o _ < "$CLAIM_FILE"
  if [ -n "$o" ]; then
    [ "$o" = "$1" ] || die "this Spark is reserved by $o ($(cut -f2- "$CLAIM_FILE" | tr '\t' ' ')); pick another"
    TOOK_CLAIM=0; log "reservation: already held by $1"; return
  fi
  gpu_idle || die "a GPU process is running and nobody reserved the Spark: find out whose it is first"
  ( set -C; printf '%s\t%s\t%s\t\n' "$1" "$(date -u -Is)" "$2" > "$CLAIM_FILE" ) 2>/dev/null \
    || die "someone reserved this Spark just now"
  TOOK_CLAIM=1; log "reserved $(hostname) as $1"
}
release() {
  [ "${TOOK_CLAIM:-0}" = 1 ] || return 0
  local o=""; [ -f "$CLAIM_FILE" ] && IFS=$'\t' read -r o _ < "$CLAIM_FILE"
  [ "$o" = "$RESERVE_OWNER" ] && rm -f "$CLAIM_FILE" && log "released $(hostname)"
}

# Memory guard: on GB10 an engine's GPU memory is not in its RSS, so the kernel's OOM killer cannot free it and the
# machine can hang. Stop the engine when MemAvailable stays under GUARD_LOW_GIB for GUARD_SUSTAIN_S seconds. Start it
# before the engine loads. guard_pid kills a process group; guard_container kills a container through the Docker socket
# (no docker CLI under memory pressure). Both end when their target is gone or FLAG is removed.
guard_pid() {  # PGID LABEL GUARD_LOG
  ( low=0; while kill -0 -- "-$1" 2>/dev/null; do
      if [ "$(avail_gib)" -lt "$GUARD_LOW_GIB" ]; then low=$((low + 1)); else low=0; fi
      if [ $low -ge "$GUARD_SUSTAIN_S" ]; then
        echo "[$(date -u +%H:%M:%S)] GUARD: MemAvailable under $GUARD_LOW_GIB GiB for $GUARD_SUSTAIN_S s, killed $2" >> "$3"
        kill -KILL -- "-$1" 2>/dev/null; break
      fi; sleep 1
    done ) < /dev/null > /dev/null 2>&1 &
}
guard_container() {  # NAME FLAG GUARD_LOG
  ( low=0; while [ -e "$2" ]; do
      if [ "$(avail_gib)" -lt "$GUARD_LOW_GIB" ]; then low=$((low + 1)); else low=0; fi
      if [ $low -ge "$GUARD_SUSTAIN_S" ]; then
        curl -s --max-time 20 --unix-socket /var/run/docker.sock -X POST "http://localhost/containers/$1/kill" > /dev/null
        echo "[$(date -u +%H:%M:%S)] GUARD: MemAvailable under $GUARD_LOW_GIB GiB for $GUARD_SUSTAIN_S s, killed $1" >> "$3"
        rm -f "$2"; break
      fi; sleep 1
    done ) < /dev/null > /dev/null 2>&1 &
}

wait_health() {  # URL TIMEOUT_S [ALIVE_CMD]: until URL/health (or /v1/models) answers; ALIVE_CMD failing ends it early
  local i
  for i in $(seq 1 $(($2 / 5))); do
    curl -sf "$1/health" > /dev/null 2>&1 || curl -sf "$1/v1/models" > /dev/null 2>&1 && return 0
    [ -n "${3:-}" ] && ! eval "$3" && return 1
    sleep 5
  done
  return 1
}

# One request with the benchmark's sampling (TEMPERATURE, TOP_P, REASONING), 64 tokens, not streamed: whether the engine
# thought (a reasoning field) and what it counted, kept in OUTFILE. Prints one line for the log.
sampling_probe() {  # URL OUTFILE
  python3 -I - "$1" "$2" "${TEMPERATURE:-0}" "${REASONING:-off}" "${TOP_P:-0.95}" <<'PY' 2>&1 | tail -1
import json, sys, urllib.request
url, out, temperature, reasoning, top_p = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5])
body = {"model": json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"], "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is 17*23?"}], "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": reasoning != "off",
                                 **({"reasoning_effort": reasoning} if reasoning != "off" else {})}}
if temperature:
    body.update(top_p=top_p, top_k=20, min_p=0.0)
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
json.dump({"request": body, "response": r}, open(out, "w"), indent=1)
m = r["choices"][0]["message"]
field = next((k for k in ("reasoning_content", "reasoning") if m.get(k)), None)
print(f"thinking {'on (' + field + ', ' + str(len(m[field])) + ' chars)' if field else 'off (no reasoning field)'}, "
      f"content {len(m.get('content') or '')} chars, {(r.get('usage') or {}).get('completion_tokens')} tokens, "
      f"finish {r['choices'][0].get('finish_reason')}")
PY
}
