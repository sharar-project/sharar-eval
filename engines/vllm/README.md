# vLLM: eugr's recipes for one DGX Spark

## 1. What runs

eugr/spark-vllm-docker's single-Spark recipe for each model, run by that repository's own `run-recipe.sh` (`--solo`,
detached) at `b62168e` (2026-10-26), in the image the recipe names, pinned by digest (`versions.env`):

| Setting | Recipe | Image | Checkpoint / draft (revision) |
|---|---|---|---|
| `qwen3.8-27b` | `qwen3.8-27b-nvfp4-dflash2` | `vllm-node`: eugr/spark-vllm `nightly-20261016` | nvidia/Qwen3.8-27B-NVFP4 `482ca0f` / z-lab/Qwen3.8-27B-DFlash2 `50307d4`, 8 drafts |
| `qwen3.8-flash-next` | `qwen3.8-flash-next-nvfp4-solo` | `vllm-node-b12x`: eugr/spark-vllm-b12x `nightly-20261026` | nvidia/Qwen3.8-Flash-Next-FP4-Local `b9ce18a` / its MTP head, 4 drafts |
| `nemotron-3.5-bolt` | `nemotron-3.5-bolt-24gb` | `vllm-node` | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB `cf2effc` / its LiLiCorr `449d241`, 7 drafts |
| `nemotron-3.5-super` | `nemotron-3.5-super-vl-nvfp4` | `vllm-node` | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 `6aae0d6` / its LiLiCorr `292bf1e`, 7 drafts |

Everything in the recipe is kept: FP8 KV, memory 0.8, batched tokens, draft counts, parsers. `serve.sh` passes these
after the recipe's arguments, so they win:

- `--max-num-seqs 32`: the recipes run 8 sequences; the cross-engine suite decodes up to 32 at once.
- `--enable-prompt-tokens-details`: cached-token counts in replies, for the suite's cache-hit check.
- `VLLM_SERVER_DEV_MODE=1`: `/reset_prefix_cache`, the suite's flush before every burst.
- `HF_HUB_OFFLINE=1`: the cached, pinned revisions are served.

The last three do not change decoding. For Bolt and Super each burst is primed twice (`PRIME_PASSES=2`): the recipes' Mamba
shared-prefix checkpoints keep a prompt only once vLLM has seen it twice. Super's revision is the one Sharar pins; Hugging Face's later `da5b265` only
deletes a backup chat template.

Variants of these four, each a settings file that sources its base and changes one thing:

| Setting | Change | Used for |
|---|---|---|
| `qwen3.8-27b-d4` | 4 DFlash2 drafts, memory 0.84 | the cross-engine runs (faster than 8 at N >= 4); agent replay |
| `qwen3.8-27b-d4-s64`, `nemotron-3.5-bolt-s64` | `--max-num-seqs 64` | suite `cross-engine-n64` |
| `qwen3.8-flash-next-d2` | 2 MTP drafts (a recurrent state per draft token, so more requests fit) | the cross-engine runs; agent replay |
| `qwen3.8-flash-next-d2-m084` | memory 0.84 | a Spark that cannot free 0.86 |
| `nemotron-3.5-bolt-m086` | memory 0.86 | agent replay |
| `nemotron-3.5-super-d3` | 3 drafts, memory 0.86 | the cross-engine runs; agent replay |
| `nemotron-3.5-super-d3-m084` | memory 0.84 | a Spark that cannot free 0.86 of its memory (spark4: 103.9 GiB free, 104.65 needed), or whose driver runs the 0.86 start out of memory in the fused-MoE autotune (spark5, driver 580.142) |
| `nemotron-3.5-super-m081` | memory 0.81 | a Spark with less free memory |
| `qwen3.8-flash-next-d2-mnbt1024`, `-mnbt2048` | `--max-num-batched-tokens` 1,024 / 2,048 | prefix-cache experiment: vLLM keeps one recurrent checkpoint before a long prompt's last prefill chunk, so smaller chunks leave it closer to the end (2,048: SWE-bench agent turns 44% -> 69% cached) |
| `qwen3.8-flash-next-d2-rb2992`, `-al2992` | recurrent checkpoints every 2,992 tokens (default policy; aligned policy) | prefix-cache experiment: ignored under the default request-boundary policy; aligned loses the exact end-of-prompt hit |
| `qwen3.8-flash-next-d2-ssmfp16`, `-ssmbf16`, `-tri`, `-tri-ssmbf16` | recurrent state in FP16 / BF16, Triton DeltaNet kernels | experiments: the recipe's b12x kernels refuse FP16 / BF16 state; Triton with BF16 matches the recipe at 1 agent |

The experiment settings are not baselines: figures use each recipe as shipped (with the draft and memory changes above).

## 2. One-time setup on a Spark

- Docker with the NVIDIA runtime, and your user in the `docker` group (`serve.sh` falls back to `sg docker`).
- Nothing to download by hand: `serve.sh up` downloads a pinned checkpoint or draft that is not in
  `~/.cache/huggingface/hub` (or `HF_HUB_DIR`), by revision (`hf`, or `uvx --from huggingface_hub hf` when `hf` is not
  installed; all four are public). A snapshot downloaded by revision has no `refs/main`,
  which an offline server needs: `serve.sh up` writes one naming the pin, and refuses to start if `refs/main` names
  another revision.
- Nothing else: the first `serve.sh up` clones the recipes to `$EUGR_DIR` (default `~/spark-vllm-docker-b62168e`) and
  pulls the image by digest (about 20 GB).

## 3. Run it

```sh
engines/common/run.sh vllm nemotron-3.5-bolt                 # reserve, serve, suite cross-engine, stop, release
engines/vllm/serve.sh up nemotron-3.5-bolt ~/vllm-bolt       # by hand: start and wait (log in ~/vllm-bolt/server.log)
engines/vllm/serve.sh down ~/vllm-bolt                       # stop
```

The server answers at `http://127.0.0.1:8000` in a container named `vllm-baseline`. A first start compiles kernels
(the caches under `~/.cache/vllm`, `~/.cache/flashinfer` and `~/.triton` are kept): allow up to 3 hours
(`START_TIMEOUT_S`). `DIR/recipe.log` holds what `run-recipe.sh` ran.

## 4. What to expect

- **Memory caps the warm cache.** Each recipe gives vLLM 80% of the Spark's memory. On the 2026-10-22 Flash-Next
  comparison, eugr's recipe kept about 15 warm 8k prompts cached with 4 MTP drafts, so cells at large N and long
  prompts stop with the reason "cache" or "queued": that is the engine's capacity, reported as such.
- **vLLM keeps 80-95% of these hybrid-model prompts cached** (its prefix cache matches whole blocks), which is why the
  suite's cache-hit floor is 60%.
- **Streams that stop early.** Under `ignore_eos`, vLLM stops streaming text after the model's own end while it keeps
  decoding; sharar-eval times such a request over its whole span and counts it as held back.
