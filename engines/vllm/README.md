# vLLM: eugr's recipes for one DGX Spark

## 1. What runs

eugr/spark-vllm-docker's single-Spark recipe for each model, run by that repository's own `run-recipe.sh` (`--solo`,
detached) at `b62168e` (2026-10-26), in the image the recipe names, pinned by digest (`versions.env`):

| Setting | Recipe | Image | Checkpoint / draft (revision) | Changed from the recipe |
|---|---|---|---|---|
| `qwen3.8-27b` | `qwen3.8-27b-nvfp4-dflash2` | `vllm-node` | nvidia/Qwen3.8-27B-NVFP4 `482ca0f` / z-lab/Qwen3.8-27B-DFlash2 `50307d4`, 8 drafts | memory 0.84 |
| `qwen3.8-flash-next` | `qwen3.8-flash-next-nvfp4-solo` | `vllm-node-b12x` | nvidia/Qwen3.8-Flash-Next-FP4-Local `b9ce18a` / its MTP head, 4 drafts | memory 0.86, PyTorch allocator cap 0.86 |
| `nemotron-3.5-bolt` | `nemotron-3.5-bolt-24gb` | `vllm-node` | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB `cf2effc` / its LiLiCorr `449d241`, 7 drafts | nothing (memory 0.8) |
| `nemotron-3.5-super` | `nemotron-3.5-super-vl-nvfp4` | `vllm-node` | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 `6aae0d6` / its LiLiCorr `292bf1e`, 7 drafts | memory 0.86 |

Images, pinned by digest in `versions.env`: `vllm-node` = `eugr/spark-vllm@sha256:431593a0a706...`
(`nightly-20261016`), `vllm-node-b12x` = `eugr/spark-vllm-b12x@sha256:57c24fc397c6...` (`nightly-20261026`, after the
Flash-Next recipe's 2026-10-25 regression fix). The recipes' memory share is 0.8; 0.84-0.86 gives the KV cache room
for 32 warm requests (the recipes' cache filled before N 32) and does not change low-N speed. Everything else in the
recipe is kept: FP8 KV, batched tokens, draft counts (except in the variants below), parsers. `serve.sh` passes these
after the recipe's arguments, so they win:

- `--max-num-seqs 32` (`-s64` settings: 64): the recipes run 8 sequences; the cross-engine suite decodes up to 32.
- `--enable-prompt-tokens-details`: cached-token counts in replies, for the suite's cache-hit check.
- `VLLM_SERVER_DEV_MODE=1`: `/reset_prefix_cache`, the suite's flush before every burst.
- `HF_HUB_OFFLINE=1`: the cached, pinned revisions are served.
- the setting's `VLLM_ARGS` (memory share, draft count) and `VLLM_ENV`, last.

The middle three do not change decoding. For Bolt and Super each burst is primed twice (`PRIME_PASSES=2`; `-d3`:
three times): the recipes' Mamba shared-prefix checkpoints keep a prompt only once vLLM has seen it twice. Their
prefix cache reuses a prompt only up to a Mamba-block boundary, so Bolt and Super `-d3` run with a 50% cache-hit floor
(`MIN_HIT=0.5`). Super's revision is the one Sharar pins; Hugging Face's later `da5b265` only deletes a backup chat
template.

Variants of these four. The settings with a suffix after `-d2` source `qwen3.8-flash-next-d2` and add one flag; the
others are full copies of their base with the change below:

| Setting | Change from the recipe | Used for |
|---|---|---|
| `qwen3.8-27b-d4` | 4 DFlash2 drafts (recipe 8), memory 0.84 | **baseline**: warm decode (fewer drafts were faster from 16 requests up; figures take the faster of this and `qwen3.8-27b` per N) and agent replay |
| `qwen3.8-27b-d4-s64` | as `-d4`, `--max-num-seqs 64` | suite `cross-engine-n64` |
| `qwen3.8-flash-next-d2` | 2 MTP drafts (recipe 4; a recurrent state per draft token, so more requests fit), memory 0.86 | **baseline**: warm decode and agent replay |
| `qwen3.8-flash-next-d2-m084` | as `-d2`, memory 0.84 | a Spark that cannot free 0.86 (0.86 tripped the memory guard on spark-9931) |
| `nemotron-3.5-bolt-s64` | `--max-num-seqs 64` (memory 0.8) | suite `cross-engine-n64` (at N 32 vLLM held at most 39.5% of its KV cache) |
| `nemotron-3.5-bolt-m086` | memory 0.86 | **baseline**: agent replay (turns up to 57,587 tokens at 16 agents) |
| `nemotron-3.5-super-d3` | 3 LiLiCorr drafts (recipe 7), memory 0.86, primed 3 times, cache-hit floor 50% | **baseline**: warm decode (figures take the faster of this and `nemotron-3.5-super` per N) and agent replay |
| `nemotron-3.5-super-d3-m084` | as `-d3`, memory 0.84 | a Spark that cannot free 0.86 (spark4: 103.9 GiB free, 104.65 needed), or whose driver runs the 0.86 start out of memory in the fused-MoE autotune (spark5, driver 580.142); the note-removed agent replay of Super ran this |
| `nemotron-3.5-super-m081` | 7 drafts, memory 0.81 | a Spark with less free memory (spark-911c: 100.4 GiB at start); about 8 warm 8k requests at once instead of 11 |
| `qwen3.8-flash-next-d2-mnbt1024`, `-mnbt2048` | `--max-num-batched-tokens` 1,024 / 2,048 | prefix-cache experiment: vLLM keeps one recurrent checkpoint before a long prompt's last prefill chunk, so smaller chunks leave it closer to the end (2,048: SWE-bench agent turns 44% -> 69% cached) |
| `qwen3.8-flash-next-d2-rb2992`, `-al2992` | recurrent checkpoints every 2,992 tokens (default policy; aligned policy) | prefix-cache experiment: ignored under the default request-boundary policy; aligned loses the exact end-of-prompt hit |
| `qwen3.8-flash-next-d2-ssmfp16`, `-ssmbf16`, `-tri`, `-tri-ssmbf16` | recurrent state in FP16 / BF16, Triton DeltaNet kernels | experiments: the recipe's b12x kernels refuse FP16 / BF16 state; Triton with BF16 matches the recipe at 1 agent |

The baselines are the four bold rows plus `nemotron-3.5-bolt` (warm decode); `qwen3.8-27b` and `nemotron-3.5-super`
were run beside them for the per-N comparison. The rest are experiments or memory fallbacks, not baselines.

## 2. One-time setup on a Spark

- Docker with the NVIDIA runtime, and your user in the `docker` group (`serve.sh` falls back to `sg docker`).
- Nothing to download by hand: `serve.sh up` downloads a pinned checkpoint or draft that is not in
  `~/.cache/huggingface/hub` (or `HF_HUB_DIR`), by revision (`hf`, or `uvx --from huggingface_hub hf` when `hf` is not
  installed; all four are public). A snapshot downloaded by revision has no `refs/main`,
  which an offline server needs: `serve.sh up` writes one naming the pin, and refuses to start if `refs/main` names
  another revision.
- Nothing else: the first `serve.sh up` clones the recipes to `$EUGR_DIR` (default `~/spark-vllm-docker-b62168e`) and
  pulls the image by digest (about 20 GB each; Flash-Next uses the b12x image, the other three the regular one).
- `serve.sh up` refuses a cached checkpoint whose `refs/main` names another revision (a cache filled by a plain
  `hf download`, or shared with another user): download the pin into a fresh `HF_HUB_DIR`, or point `refs/main` at the
  pin if that revision is also in `snapshots/`.
- Passwordless `sudo` for the page-cache drop before a start (`DROP_PAGE_CACHE=1`): at 0.84-0.86 vLLM refuses to start
  when less memory is free than its share asks for.

## 3. Run it

```sh
engines/common/run.sh vllm nemotron-3.5-bolt                 # reserve, serve, suite cross-engine, stop, release
engines/common/run.sh vllm qwen3.8-27b-d4-s64 cross-engine-n64
engines/common/run.sh vllm nemotron-3.5-bolt-m086 cross-engine-agents   # the agent replay
engines/vllm/serve.sh up nemotron-3.5-bolt ~/vllm-bolt       # by hand: start and wait (log in ~/vllm-bolt/server.log)
engines/vllm/serve.sh down ~/vllm-bolt                       # stop
```

The server answers at `http://127.0.0.1:8000` in a container named `vllm-baseline`. A first start compiles kernels
(the caches under `~/.cache/vllm`, `~/.cache/flashinfer` and `~/.triton` are kept): allow up to 3 hours
(`START_TIMEOUT_S`). `DIR/recipe.log` holds what `run-recipe.sh` ran. The container outlives a killed `run.sh`: stop
it with `serve.sh down <the run's directory>`, which also hands the root-owned kernel caches in `$HOME` back to you.

## 4. What to expect

- **Memory caps the warm cache.** vLLM keeps a recurrent state per draft token for every request, so its memory share
  and draft count decide how many warm requests fit (Super at 8k: 7 at 0.8 and 11 at 0.86 with 7 drafts; Flash-Next at
  8k with 4 drafts, about 15 on 2026-10-22). Cells at large N and long prompts stop with the reason "cache" or
  "queued": that is the engine's capacity, reported as such.
- **The start can run out of memory before serving** (fused-MoE / FP8 GEMM autotune, `NV_ERR_NO_MEMORY` in `dmesg`),
  depending on the box's free memory and driver: retry once, then use the setting's `-m084` fallback.
- **vLLM keeps 80-95% of these hybrid-model prompts cached** (its prefix cache matches whole blocks), which is why the
  suite's cache-hit floor is 60%.
- **Streams that stop early.** Under `ignore_eos`, vLLM stops streaming text after the model's own end while it keeps
  decoding; sharar-eval times such a request over its whole span and counts it as held back.
