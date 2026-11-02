#!/bin/bash
# TensorFold baseline: start or stop MiaAI-Lab's one-Spark TensorFold recipe for one model (engines/common/ENGINE_FOLDER.md).
#   serve.sh up SETTING DIR     start the guard, then the recipe's start.sh with the setting's variables; wait until it answers
#   serve.sh down DIR           the recipe's stop.sh; wait until the GPU is free
# start.sh builds the image (scripts/prepare.sh) when it is missing, launches the container detached, waits for the API
# and sends one smoke request; its output goes to DIR/server-start.log, the container's log to DIR/server.log at each
# stop. The pool TensorFold sizes at start is MemAvailable less TENSORFOLD_MEMORY_RESERVE_GIB: up sets that reserve to
# MemAvailable - ENGINE_BUDGET_GIB (at least the recipe's 9), so the engine's total is the budget every engine gets;
# TF_RESERVE_GIB (setting or environment) sets it outright.
# TF_LAUNCH=native (a setting's choice) runs TensorFold's release binary on the host instead of a recipe: TF_BIN serves
# the checkpoint's snapshot in the Hugging Face cache with the setting's PARALLEL, TF_CONTEXT and TF_ARGS; its output
# goes to DIR/server.log, its process group id to DIR/tf.pid.
set -u
if ! docker info > /dev/null 2>&1 && [ -z "${SERVE_SG:-}" ]; then
  exec sg docker -c "SERVE_SG=1 bash $(printf '%q ' "$(readlink -f "$0")" "$@")"
fi
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HERE/../common/defaults.env"; . "$HERE/../common/lib.sh"; . "$HERE/versions.env"
export DOCKER_CONTEXT=${DOCKER_CONTEXT:-default}
CMD=${1:-}
case $CMD in
up) SETTING=${2:?usage: serve.sh up SETTING DIR}; DIR=${3:?usage: serve.sh up SETTING DIR}
    [ -f "$HERE/settings/$SETTING.env" ] || die "no setting $SETTING"; . "$HERE/settings/$SETTING.env" ;;
down) DIR=${2:?usage: serve.sh down DIR} ;;
*) echo "usage: serve.sh up SETTING DIR | down DIR" >&2; exit 2 ;;
esac
mkdir -p "$DIR"; LOG=$DIR/serve.log; FLAG=$DIR/tf.running
# down runs without the setting: up leaves the container and recipe checkout it started in DIR/tf.vars
[ "$CMD" = down ] && [ -f "$DIR/tf.vars" ] && . "$DIR/tf.vars"
[ "$CMD" = up ] && printf 'CONTAINER=%q\nTF_DIR=%q\n' "$CONTAINER" "$TF_DIR" > "$DIR/tf.vars"
docker info > /dev/null 2>&1 || die "no Docker access: add your user to the docker group, or run under sg docker -c \"...\""

if [ "$CMD" = down ] && [ -f "$DIR/tf.pid" ]; then             # a native server
  rm -f "$FLAG"; pg=$(cat "$DIR/tf.pid")
  kill -TERM -- "-$pg" 2>/dev/null
  for i in $(seq 1 60); do kill -0 -- "-$pg" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pg" 2>/dev/null; rm -f "$DIR/tf.pid"
  for i in $(seq 1 60); do gpu_idle && break; sleep 2; done
  sleep 10
  log "TensorFold stopped, MemAvailable $(avail_gib) GiB (GPU processes: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')])"
  exit 0
fi
if [ "$CMD" = down ]; then
  rm -f "$FLAG"                                                   # ends the guard
  if docker ps -q -f "name=^$CONTAINER\$" | grep -q .; then
    { echo "=== $(date -u -Is)"; docker logs "$CONTAINER"; } >> "$DIR/server.log" 2>&1
  fi
  [ -x "$TF_DIR/stop.sh" ] && (cd "$TF_DIR" && ./stop.sh) >> "$DIR/server-start.log" 2>&1
  docker rm -f "$CONTAINER" > /dev/null 2>&1
  for i in $(seq 1 60); do gpu_idle && break; sleep 2; done
  sleep 10    # let unified memory settle before the next start sizes its pool
  log "TensorFold stopped, MemAvailable $(avail_gib) GiB (GPU processes: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')])"
  exit 0
fi

if [ "${DROP_PAGE_CACHE:-0}" = 1 ]; then  # Bolt's libbolt grows a stream's KV only into free memory, not the page cache
  sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2> /dev/null || log "could not drop the page cache (no passwordless sudo)"
fi
if [ "${TF_LAUNCH:-recipe}" = native ]; then
  SNAP=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION
  [ -f "$SNAP/config.json" ] || die "no snapshot $MODEL_REPO@$MODEL_REVISION in the Hugging Face cache ($SNAP)"
  [ -x "$TF_BIN" ] || die "no TensorFold binary at $TF_BIN (unpack the $TF_VERSION release archive there)"
  gpu_idle || die "a GPU process is running: one engine at a time"
  avail=$(avail_gib)
  if [ -n "${TF_RESERVE_GIB:-}" ]; then res=$TF_RESERVE_GIB; else res=$((avail - ENGINE_BUDGET_GIB)); [ $res -lt 9 ] && res=9; fi
  ARGS="serve $SNAP --name $SERVED_NAME --host 127.0.0.1 --port ${URL##*:} --context $TF_CONTEXT --parallel $PARALLEL --thinking ${TF_ARGS:-}"
  log "TensorFold $TF_VERSION ($SETTING): TENSORFOLD_MEMORY_RESERVE_GIB=$res ${TF_ENV:-} $TF_BIN $ARGS; MemAvailable $avail GiB"
  touch "$FLAG"
  echo "=== $(date -u -Is) TENSORFOLD_MEMORY_RESERVE_GIB=$res ${TF_ENV:-} $TF_BIN $ARGS" >> "$DIR/server.log"
  setsid env TENSORFOLD_MEMORY_RESERVE_GIB=$res TENSORFOLD_NO_UPDATE_CHECK=1 ${TF_ENV:-} "$TF_BIN" $ARGS >> "$DIR/server.log" 2>&1 < /dev/null &
  pg=$!; echo $pg > "$DIR/tf.pid"
  guard_pid "$pg" "TensorFold" "$DIR/guard.log"
  wait_health "$URL" "$START_TIMEOUT_S" "kill -0 -- -$pg 2>/dev/null" \
    || { log "TensorFold did not start: $(tail -3 "$DIR/server.log" | tr '\n' ' ' | cut -c1-300)"; bash "$0" down "$DIR"; exit 1; }
  log "$(grep -a -iE 'streams at once|prompt cache|GiB' "$DIR/server.log" | tail -3 | tr -s ' ' | tr '\n' ' ' | cut -c1-400)"
  log "TensorFold serving at $URL, MemAvailable $(avail_gib) GiB"
  exit 0
fi
if [ ! -d "$TF_DIR/.git" ]; then git clone -q "$RECIPE_REPO" "$TF_DIR" || die "could not clone $RECIPE_REPO"; fi
git -C "$TF_DIR" checkout -q --detach "$RECIPE_COMMIT" 2>/dev/null || { git -C "$TF_DIR" fetch -q origin && git -C "$TF_DIR" checkout -q --detach "$RECIPE_COMMIT"; } \
  || die "$TF_DIR has no commit $RECIPE_COMMIT"
[ "$(git -C "$TF_DIR" rev-parse HEAD)" = "$RECIPE_COMMIT" ] || die "$TF_DIR is not at $RECIPE_COMMIT"
[ -f "$TF_DIR/.env" ] && die "$TF_DIR/.env would override the setting: move it away"
gpu_idle || die "a GPU process is running: one engine at a time"
docker ps -q -f "name=^$CONTAINER\$" | grep -q . && die "a container named $CONTAINER is running (someone's benchmark?): leave it"
docker rm -f "$CONTAINER" > /dev/null 2>&1                        # a stopped one left by an earlier start

avail=$(avail_gib)
if [ -n "${TF_RESERVE_GIB:-}" ]; then res=$TF_RESERVE_GIB      # a setting's or the caller's reserve
else res=$((avail - ENGINE_BUDGET_GIB)); [ $res -lt 9 ] && res=9; fi
touch "$FLAG"
guard_container "$CONTAINER" "$FLAG" "$DIR/guard.log"
ARGS="TENSORFOLD_MEMORY_RESERVE_GIB=$res PARALLEL=$PARALLEL CONTEXT=$TF_CONTEXT VISION=$VISION MODEL_REVISION=$MODEL_REVISION WAIT_TIMEOUT=$START_TIMEOUT_S ${TF_ENV:-} ./start.sh ${TF_ARGS:-}"
log "TensorFold ($SETTING): $ARGS; MemAvailable $avail GiB"
echo "=== $(date -u -Is) $ARGS" >> "$DIR/server-start.log"
(cd "$TF_DIR" && env $ARGS) >> "$DIR/server-start.log" 2>&1 \
  || { log "start.sh failed: $(tail -3 "$DIR/server-start.log" | tr '\n' ' ' | cut -c1-250)"; bash "$0" down "$DIR"; exit 1; }
wait_health "$URL" 600 || { log "TensorFold does not answer at $URL"; bash "$0" down "$DIR"; exit 1; }
log "$(grep -a -iE 'streams at once|sequence memory|left after' "$DIR/server-start.log" | tail -1 | tr -s ' ' | cut -c1-200)"
log "TensorFold serving at $URL, MemAvailable $(avail_gib) GiB"
