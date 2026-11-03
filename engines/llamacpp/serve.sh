#!/bin/bash
# llama.cpp baseline: start or stop llama-server for one model (engines/common/ENGINE_FOLDER.md).
#   serve.sh up SETTING DIR     start llama-server with the setting's GGUF, draft head and sizes; wait until it answers
#   serve.sh down DIR           SIGINT to its process group, SIGKILL after 2 min; wait until the GPU is free
# The server runs in its own process group (setsid, recorded in DIR/llama-server.pid) under choom -n 1000, so a host
# OOM picks it first, with common/lib.sh's memory guard on that group. Log: DIR/server.log.
# SPEC_TYPE (default draft-mtp; e.g. draft-dflash) and LLAMA_DRAFTS (default the setting's DRAFTS) override the drafter.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HERE/../common/defaults.env"; . "$HERE/../common/lib.sh"; . "$HERE/versions.env"
CMD=${1:-}
case $CMD in
up) SETTING=${2:?usage: serve.sh up SETTING DIR}; DIR=${3:?usage: serve.sh up SETTING DIR}
    [ -f "$HERE/settings/$SETTING.env" ] || die "no setting '$SETTING': $(ls "$HERE/settings" | sed 's/\.env$//' | tr '\n' ' ')"
    . "$HERE/settings/$SETTING.env" ;;
down) DIR=${2:?usage: serve.sh down DIR} ;;
*) echo "usage: serve.sh up SETTING DIR | serve.sh down DIR" >&2; exit 2 ;;
esac
mkdir -p "$DIR"; PIDF=$DIR/llama-server.pid

if [ "$CMD" = down ]; then
  P=$(cat "$PIDF" 2>/dev/null) || { log "no $PIDF: nothing of ours to stop"; exit 0; }
  kill -INT -- "-$P" 2>/dev/null                     # SIGINT: llama-server finishes and exits
  for i in $(seq 1 24); do kill -0 "$P" 2>/dev/null || break; sleep 5; done
  kill -KILL -- "-$P" 2>/dev/null
  for i in $(seq 1 30); do gpu_idle && break; sleep 2; done
  rm -f "$PIDF"
  log "llama-server stopped, MemAvailable $(avail_gib) GiB (GPU processes: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')])"
  exit 0
fi

SERVER=$LLAMA_DIR/build/bin/llama-server
[ -x "$SERVER" ] || die "no $SERVER: build llama.cpp $LLAMA_COMMIT (README section 2) or set LLAMA_DIR"
[ -f "$GGUF" ] || die "GGUF shard 1 not found at $GGUF (README section 2, or set GGUF)"
[ -f "$DRAFT_GGUF" ] || die "draft head not found at $DRAFT_GGUF (README section 2, or set DRAFT_GGUF)"
gpu_idle || die "a GPU process is running: one engine at a time"
curl -sf "$URL/health" > /dev/null 2>&1 && die "something already answers at $URL"

ARGS=(-m "$GGUF" --alias "$SERVED_MODEL" --host 127.0.0.1 --port "$PORT" -ngl 99 -fa on
      -c "$CTX_SIZE" -kvu -np "$SLOTS" -ctk "$KV_TYPE" -ctv "$KV_TYPE" -b 2048 -ub 2048
      --spec-type "${SPEC_TYPE:-draft-mtp}" -md "$DRAFT_GGUF" -ngld 99 --spec-draft-n-max "${LLAMA_DRAFTS:-$DRAFTS}" ${LLAMA_ARGS:-})
log "llama.cpp ($SETTING): ${ARGS[*]}; MemAvailable $(avail_gib) GiB"
echo "=== $(date -u -Is) ${ARGS[*]}" >> "$DIR/server.log"
setsid choom -n 1000 -- "$SERVER" "${ARGS[@]}" >> "$DIR/server.log" 2>&1 < /dev/null &
P=$!; echo "$P" > "$PIDF"
guard_pid "$P" llama-server "$DIR/guard.log"         # MemAvailable under GUARD_LOW_GIB for GUARD_SUSTAIN_S: kill the group

if wait_health "$URL" "$START_TIMEOUT_S" "kill -0 $P 2>/dev/null"; then
  log "llama.cpp live: $SLOTS slots sharing $CTX_SIZE tokens of KV, pid $P, MemAvailable $(avail_gib) GiB"
  exit 0
fi
log "llama.cpp did not start: $(tail -3 "$DIR/server.log" | tr '\n' ' ' | cut -c1-250)"
kill -KILL -- "-$P" 2>/dev/null; rm -f "$PIDF"
exit 1
