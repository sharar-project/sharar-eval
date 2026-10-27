#!/bin/bash
# vLLM baseline: start or stop vLLM for one model (engines/common/ENGINE_FOLDER.md).
#   serve.sh up SETTING DIR     start the guard, then the recipe's container; wait until it answers (DIR/server.log)
#   serve.sh down DIR           stop the container and the guard; wait until the GPU is free
# eugr/spark-vllm-docker's recipe for the setting's model, run by that repository's own run-recipe.sh (--solo, detached)
# at RECIPE_COMMIT, in the image the recipe names pinned by digest. These are passed after the recipe's own arguments,
# so they win (vLLM keeps the last value of a flag):
#   --max-num-seqs $MAX_NUM_SEQS      the recipes run 8; the cross-engine suite decodes up to 32 at once
#   --enable-prompt-tokens-details    cached-token counts in replies, for the suite's cache-hit check
#   VLLM_SERVER_DEV_MODE=1            /reset_prefix_cache, the suite's flush before every burst
#   HF_HUB_OFFLINE=1                  serve the cached, pinned revisions
# The last three do not change decoding. A snapshot downloaded by revision has no refs/main, which an offline server
# needs to find it: up writes one naming the pin, and refuses to start if refs/main names another revision.
set -u
if ! docker info > /dev/null 2>&1 && [ -z "${SERVE_SG:-}" ]; then
  exec sg docker -c "SERVE_SG=1 bash $(printf '%q ' "$(readlink -f "$0")" "$@")"
fi
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HERE/../common/defaults.env"; . "$HERE/../common/lib.sh"; . "$HERE/versions.env"
export DOCKER_CONTEXT=${DOCKER_CONTEXT:-default}
# run-recipe.sh looks for the model in, and mounts into the container, $HF_HOME: make it the cache the pins are
# checked in (a login shell may point HF_HOME at a shared cache)
export HF_HOME=${HF_HUB_DIR%/hub}; unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
CMD=${1:-}
case $CMD in
up) SETTING=${2:?usage: serve.sh up SETTING DIR}; DIR=${3:?usage: serve.sh up SETTING DIR}
    [ -f "$HERE/settings/$SETTING.env" ] || die "no setting $SETTING"; . "$HERE/settings/$SETTING.env" ;;
down) DIR=${2:?usage: serve.sh down DIR} ;;
*) echo "usage: serve.sh up SETTING DIR | down DIR" >&2; exit 2 ;;
esac
mkdir -p "$DIR"; LOG=$DIR/serve.log; FLAG=$DIR/vllm.running
docker info > /dev/null 2>&1 || die "no Docker access: add your user to the docker group, or run under sg docker -c \"...\""

if [ "$CMD" = down ]; then
  rm -f "$FLAG"                                                   # ends the guard
  docker stop -t 60 "$CONTAINER" > /dev/null 2>&1; docker rm -f "$CONTAINER" > /dev/null 2>&1
  for i in $(seq 1 30); do gpu_idle && break; sleep 2; done
  log "vLLM stopped (GPU processes: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')])"
  exit 0
fi

pin() {  # REPO REVISION: the pinned snapshot is cached and refs/main names it (written when missing)
  local d=$HF_HUB_DIR/models--${1//\//--}
  [ -d "$d/snapshots/$2" ] || die "$1@$2 is not in $HF_HUB_DIR: hf download $1 --revision $2"
  if [ -f "$d/refs/main" ]; then
    [ "$(cat "$d/refs/main")" = "$2" ] || die "$d/refs/main names $(cat "$d/refs/main"), not the pinned $2"
  else
    mkdir -p "$d/refs" && printf '%s' "$2" > "$d/refs/main" && log "wrote $d/refs/main -> $2"
  fi
}
pin "$MODEL_REPO" "$MODEL_REVISION"
[ -n "${DRAFT_REPO:-}" ] && pin "$DRAFT_REPO" "$DRAFT_REVISION"
docker image inspect "$IMAGE" > /dev/null 2>&1 || docker pull -q "$IMAGE" > /dev/null || die "could not pull $IMAGE"
if [ ! -d "$EUGR_DIR/.git" ]; then git clone -q "$RECIPE_REPO" "$EUGR_DIR" || die "could not clone $RECIPE_REPO"; fi
git -C "$EUGR_DIR" checkout -q --detach "$RECIPE_COMMIT" 2>/dev/null || { git -C "$EUGR_DIR" fetch -q origin && git -C "$EUGR_DIR" checkout -q --detach "$RECIPE_COMMIT"; } \
  || die "$EUGR_DIR has no commit $RECIPE_COMMIT"
[ "$(git -C "$EUGR_DIR" rev-parse HEAD)" = "$RECIPE_COMMIT" ] || die "$EUGR_DIR is not at $RECIPE_COMMIT"
gpu_idle || die "a GPU process is running: one engine at a time"
docker ps -q -f "name=^$CONTAINER\$" | grep -q . && die "a container named $CONTAINER is running (someone's benchmark?): leave it"
docker rm -f "$CONTAINER" > /dev/null 2>&1                        # a stopped one left by an earlier start

touch "$FLAG"; guard_container "$CONTAINER" "$FLAG" "$DIR/guard.log"
log "vLLM $SETTING: recipe $RECIPE at ${RECIPE_COMMIT:0:7}, image $IMAGE, max-num-seqs $MAX_NUM_SEQS ($(avail_gib) GiB free)"
(cd "$EUGR_DIR" && ./run-recipe.sh "$RECIPE" --solo -d -t "$IMAGE" --name "$CONTAINER" --port "${URL##*:}" \
   -e HF_HUB_OFFLINE=1 -e VLLM_SERVER_DEV_MODE=1 \
   -- --max-num-seqs "$MAX_NUM_SEQS" --enable-prompt-tokens-details) > "$DIR/recipe.log" 2>&1 \
  || { rm -f "$FLAG"; docker rm -f "$CONTAINER" > /dev/null 2>&1; die "run-recipe.sh failed (see $DIR/recipe.log)"; }
( docker logs -f "$CONTAINER" > "$DIR/server.log" 2>&1 & )
if ! wait_health "$URL" "$START_TIMEOUT_S" "docker ps -q -f name=^$CONTAINER\$ | grep -q ."; then
  log "vLLM $SETTING did not start: $(tail -5 "$DIR/server.log" | tr '\n' ' ' | cut -c1-300)"
  rm -f "$FLAG"; docker rm -f "$CONTAINER" > /dev/null 2>&1; exit 1
fi
log "vLLM $SETTING live: $(grep -aE 'GPU KV cache size|Maximum concurrency' "$DIR/server.log" | tail -2 | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ' | cut -c1-240)"
