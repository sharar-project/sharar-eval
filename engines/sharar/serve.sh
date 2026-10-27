#!/bin/bash
# Sharar baseline: start or stop Sharar for one model (engines/common/ENGINE_FOLDER.md).
#   serve.sh up SETTING DIR    start `sharar serve $TARGET --$PROFILE` from the pinned checkout, wait until it answers
#   serve.sh down DIR          stop it (one Ctrl-C: it drains and stores its cached prefixes), then make sure it is gone
# The server's log goes to DIR/server.log, its process group to DIR/sharar.pid.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HERE/../common/defaults.env"; . "$HERE/../common/lib.sh"; . "$HERE/versions.env"
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH SHARAR_RESULTS PYTHONDONTWRITEBYTECODE=1

case ${1:-} in
up)
  SETTING=${2:?usage: serve.sh up SETTING DIR}; DIR=${3:?usage: serve.sh up SETTING DIR}
  . "$HERE/settings/$SETTING.env"; mkdir -p "$DIR"
  if [ -f "$DIR/sharar.pid" ] && kill -0 "$(cat "$DIR/sharar.pid")" 2>/dev/null; then log "Sharar already running"; exit 0; fi
  gpu_idle || die "a GPU process is running"
  [ "$(git -C "$SHARAR_DIR" rev-parse HEAD 2>/dev/null)" = "$SHARAR_COMMIT" ] \
    || die "$SHARAR_DIR is not a Sharar checkout at $SHARAR_COMMIT (see README.md, section 2)"
  SSD=$SHARAR_SSD_ROOT/$SETTING; rm -rf "$SSD"; mkdir -p "$SSD"   # an empty SSD-tier store for every start
  # cd on its own line: "cd X && setsid ... &" backgrounds a subshell, so $! would not be the server's process group
  cd "$SHARAR_DIR"
  SHARAR_SSD_DIR=$SSD setsid nohup choom -n 1000 -- uv run sharar serve "$TARGET" --"$PROFILE" > "$DIR/server.log" 2>&1 < /dev/null &
  SPID=$!
  cd "$HERE"
  echo "$SPID" > "$DIR/sharar.pid"
  [ "$(ps -o pgid= -p "$SPID" | tr -d ' ')" = "$SPID" ] || log "WARNING: process $SPID does not lead its process group; down may miss it"
  guard_pid "$SPID" "Sharar --$PROFILE" "$DIR/guard.log"   # beside Sharar's own guard (4 GiB floor, SIGTERM, then KILL)
  # warm: about 1.5 min; a first start on a new machine downloads the checkpoint and compiles kernels (20 min or more)
  if ! wait_health "$URL" "${START_TIMEOUT_S:-7200}" "kill -0 $SPID 2>/dev/null"; then
    log "Sharar --$PROFILE did not answer: $(tail -3 "$DIR/server.log" | tr '\n' ' ' | cut -c1-300)"
    bash "$HERE/serve.sh" down "$DIR"; exit 1
  fi
  log "Sharar --$PROFILE live: $(grep -m1 'pools:' "$DIR/server.log" | sed 's/^ *//')"
  ;;
down)
  DIR=${2:?usage: serve.sh down DIR}
  [ -f "$DIR/sharar.pid" ] || exit 0
  SPID=$(cat "$DIR/sharar.pid")
  kill -INT -- "-$SPID" 2>/dev/null          # once: a second Ctrl-C interrupts the prefix store (KeyboardInterrupt)
  for i in $(seq 1 24); do kill -0 "$SPID" 2>/dev/null || break; sleep 5; done
  kill -KILL -- "-$SPID" 2>/dev/null; sleep 5
  rm -f "$DIR/sharar.pid"
  for i in $(seq 1 12); do gpu_idle && break; sleep 5; done
  gpu_idle || log "WARNING: GPU processes left after stopping Sharar: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')"
  ;;
*) echo "usage: serve.sh up SETTING DIR | serve.sh down DIR" >&2; exit 2 ;;
esac
