# TensorFold

## 1. What runs

MiaAI-Lab's one-Spark recipe, [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `fd3f43e` (2026-10-27): TensorFold v1.1.4 (`ashhart/TensorFold` `eab85828`, the Zig engine `tensorfold-native`) with
the recipe's patches 0001-0017, in the image `tensorfold-qwen38:v1.1.4-eab8582` that the recipe's `scripts/prepare.sh`
builds on `nvcr.io/nvidia/pytorch:26.07-py3`. The recipe pins everything (`scripts/config.sh`); `versions.env` pins the
recipe. Checkpoint: nvidia/Qwen3.8-Flash-Next-FP4-Local `b9ce18a` (the recipe's default, the same checkpoint vLLM's
eugr recipe serves). Every setting keeps the recipe's fp8 KV cache, MTP drafts, sampling and thinking, with vision off.

| Setting | What it is | Streams | Window (`CONTEXT`) | Reserve |
|---|---|---|---|---|
| `qwen3.8-flash-next` | The comparison: inside the 102 GiB budget every engine gets | 16 | 40,960 | `MemAvailable` - 102 (15-16 GiB) |
| `qwen3.8-flash-next-maxfit` | As many warm requests as fit one GB10, outside the budget; one prompt size per run | 8k: 32, 16k: 32, 32k: 16 | 8k: 12,288, 16k: 19,968, 32k: 40,960 | 8k: 9, 16k: 5, 32k: 7 GiB |
| `qwen3.8-flash-next-p32` | For trying a size by hand: `TF_PARALLEL`, `TF_CONTEXT`, `TF_RESERVE_GIB`, `TF_ENV` from the environment | 32 | 12,288 | 9 GiB |

`-maxfit` at 16k also sets `TF_FLASHNEXT_HEAD_KV=fp8` (the MTP head's own cache in fp8 too; the recipe notes it costs
about a fifth of accepted drafts) and `TF_FLASHNEXT_CHUNK=1024` (prompt passes of 1,024 rows instead of 3,072: a
smaller per-stream buffer, slower priming, decode unchanged). Section 4 says why each number is what it is.

There is no `-fast` setting: the recipe as shipped (5 streams of 262,144 tokens, vision on) does not start under the
budget ("--parallel 5 needs 23.92 GiB for its streams at a 262144-token window, and the CUDA memory budget leaves
22.93 GiB", spark3, 2026-10-28), and suite `cross-engine-n4`'s cells are `cross-engine`'s N = 1, 2 and 4 anyway.

## 2. One-time setup on a Spark

- Docker with the NVIDIA runtime, your user in the `docker` group (`serve.sh` re-runs itself under `sg docker`).
- About 50 GB free disk for the image, plus 96 GiB for the checkpoint if it is not in `~/.cache/huggingface` yet.
- The first `serve.sh up` clones the recipe into `$TF_DIR` (default `~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`)
  at `RECIPE_COMMIT` and its `start.sh` builds the image (30-60 minutes) and downloads the checkpoint. Building first,
  with no engine running, is quicker to debug: `cd $TF_DIR && ./scripts/prepare.sh`. No prebuilt image for this patch
  set was on ghcr.io (2026-10-27).
- No `.env` in the recipe checkout (it would override the setting; `serve.sh` refuses one).
- Resolve the prompt sets once, with no engine serving: `uv run sharar-eval efficiency prepare --context 8k,16k,32k`.

## 3. Run it

One prompt size per run: TensorFold has no cache flush, so a fresh server per size keeps one size's kept prompts out of
the next one's cache, and `-maxfit` sizes its server for one size. `CONTEXT` picks the suite's sizes (`suite --context`).

```sh
for c in 8k 16k 32k; do CONTEXT=$c engines/common/run.sh tensorfold qwen3.8-flash-next; done          # the comparison
for c in 8k 32k 16k; do CONTEXT=$c engines/common/run.sh tensorfold qwen3.8-flash-next-maxfit; done   # max-fit (16k last: lowest reserve)
CONTEXT=16k TF_PARALLEL=32 TF_CONTEXT=22528 TF_RESERVE_GIB=6 engines/common/run.sh tensorfold qwen3.8-flash-next-p32   # by hand
engines/tensorfold/serve.sh up qwen3.8-flash-next ~/tf-fn    # serve only (start.sh output in ~/tf-fn/server-start.log)
engines/tensorfold/serve.sh down ~/tf-fn                     # stop.sh; the container's log goes to ~/tf-fn/server.log
uv run python engines/tensorfold/restore_probe.py http://127.0.0.1:8888 32k 1024    # which prompts resume from their kept state
```

Results: `~/engine-baselines/tensorfold/<setting>/<UTC time>-<host>/` (run.sh). Read `report.txt`, then `server.log`
for `RestoreRejected`, `evicted` and `refused` when a row stops early.

## 4. Sizing: streams, window and reserve

**The pool.** TensorFold's pool is `MemAvailable` after the weights load (74.35 GiB on the GPU) less
`TENSORFOLD_MEMORY_RESERVE_GIB`: about 25 GiB at reserve 15, 31 at 9, 34.7 at 5 (vision off). `serve.sh` sets the
reserve to `MemAvailable` at start less `ENGINE_BUDGET_GIB` (102), or to `TF_RESERVE_GIB`. Keep at least 5: at 5 free
memory fell to 4 GiB under load, and the guard stops the engine under 2 GiB (a GB10 out of memory can freeze). What the
reserve leaves free is also the page cache the 26.8 GiB n-gram table is read through.

**What a warm request costs.** Every stream gets its window when the server starts, and a warm request's prompt is
also kept as a separate copy in the prompt cache (a restore copies it into the stream; the cache keeps its own):

| Part | Bytes | 16k request (window 19,968) |
|---|---|---|
| Stream, fixed: DeltaNet state (110 MiB), the draft head's rollback copy of it (108 MiB), a prompt-chunk buffer (chunk x 10,240 x 2 B: 60 MiB at 3,072 rows, 20 at 1,024), draft logits (8 MiB) | ~0.28-0.34 GiB | ~0.3 GiB |
| Stream, KV: window x 17,888 B (fp8, 12 attention layers and the MTP head) | 17.5 KiB a token | ~0.33 GiB |
| Kept prompt: DeltaNet state + the prompt's KV | ~0.11 GiB + prompt x ~17.9 KiB | ~0.41 GiB |

The engine reports the stream size at start ("32 streams at once, 0.68 GiB each at a 20480-token window ... a 13.1 GiB
prompt cache"); the prompt cache is what the streams leave of the pool. N warm requests need N streams plus N kept
prompts: 32 at 8k ~26 GiB, at 16k ~34 GiB, at 32k ~60 GiB (32 streams of 40,960 alone ~33 GiB). The recipe's
"1.4-1.6 million fp8 tokens" counts the KV row only; its benchmarks send fresh prompts, so nothing is kept.

**The window: at least (largest prompt + 1,024 + 16) / 0.95.** Patch 0005 keeps a stream's MTP head cache in bf16 unless
the stream (prompt + `max_tokens` + 16) comes within about a twentieth of the window, then fp8, and a kept state
restores only into the same head format. A prompt primed with 1 token sits below that line and the same prompt timed
with 1,024 above it: the restore is refused (`RestoreRejected` in the server log), the prompt is prefilled again and the
cell reads cache hit 0% (32k at a 36,864 window, 2026-10-28; 16k at 20,480). Largest prompts in the suite's balanced
sets: 8k 9,331, 16k 18,430, 32k 36,703, so at least 11,000 / 20,500 / 39,800. `TF_FLASHNEXT_HEAD_KV=fp8` keeps every
head in fp8 and removes the line, so the window only has to hold the request.

**Streams.** The recipe's `start.sh` takes `PARALLEL` 1 to 16; `--parallel N` as an extra argument (`TF_ARGS`, which
`-maxfit` and `-p32` set) wins over it. The engine decodes up to 32 streams a round (`round_streams` in
`cuda_lanes.zig`) and more in turns; batched draft heads cover 16 (`max_lanes`), above that each request drafts alone.

## 5. What to expect

- **A row ends where the pool runs out**: "cache hit 0% under 60%" (kept prompts evicted: `evicted` grows in the
  server log) or "decodes at most k of N" (fewer streams than N). Check `RestoreRejected` first: it is the window line
  above, not memory.
- **No cache flush.** TensorFold has no flush endpoint (`bench_v1.json`): prompts primed for earlier bursts stay
  cached until the pool needs the room (oldest first). A burst's own prompts are primed just before it.
- **No queue for memory.** A request that does not fit the pool is refused, not queued.
- **First pass against repeats.** Decode reads n-gram rows through the page cache: the first burst over new prompts
  can be slower than later ones while those rows come off the SSD (`disk.log` shows the reads).
- **Name clash.** run.sh's `CONTEXT` is the prompt sizes; the recipe's `CONTEXT` (the window) is `TF_CONTEXT` here.
- A start takes about 3 minutes (the weights load, then `start.sh`'s smoke request). `START_TIMEOUT_S` (default 90
  minutes) covers a first start that builds the image.

## 6. Measured (spark3, cross-engine suite, thinking on)

Aggregate decode tok/s at N = 1 / 2 / 4 / 8 / 16, every cell at 100% cache hit:

| Size | Setting | Window | N 1-16 | N 32 | Run |
|---|---|---|---|---|---|
| 8k | `qwen3.8-flash-next` | 36,864 | 51.3 / 76.7 / 109.0 / 143.8 / 192.6 | 16 streams | 20261028-0036-spark3 |
| 16k | `qwen3.8-flash-next` | 36,864 | 48.7 / 71.3 / 102.6 / 142.9 / 183.9 | 16 streams | 20261027-2257-spark3 |
| 32k | `qwen3.8-flash-next` | 40,960 | 51.2 / 75.8 / 108.7 / 147.1 / stopped | - | 20261028-0151-spark3 |

8k and 16k ran before the window moved to 40,960 (no request there comes near the window line). 32k N 16 stopped on
cache hit: a 7.7 GiB prompt cache cannot keep 16 primed 32k prompts at that reserve.

`-maxfit` values (run 2026-10-28 through `-p32` with the same overrides; folders `tensorfold/qwen3.8-flash-next-p32/`):

| Size | Streams, window, reserve | N 1 / 2 / 4 / 8 / 16 / 32 | Run |
|---|---|---|---|
| 8k | 32, 12,288, 9 GiB | 52.2 / 76.9 / 107.1 / 142.7 / 188.1 / 163.6 | 20261028-0401-spark3 |
| 16k | 32, 19,968, 5 GiB, head KV fp8, chunk 1,024 | 46.8 / 72.1 / 99.6 / 131.4 / 180.2 / 158.6 | 20261028-0641-spark3 |
| 32k | 16, 40,960, 7 GiB | 51.1 / 78.3 / 108.0 / 146.9 / 187.7 / - | 20261028-0456-spark3 |

N 32 fits at 8k and 16k (100% hit, nothing evicted) but decodes less than N 16: batched draft heads cover 16 streams
(`max_lanes`), so past 16 each stream drafts alone. At 16k the engine reported 72.54 GiB of weights and scratch with
1,024-row chunks (74.35 at 3,072) and a 37.8 GiB pool at reserve 5: 32 streams of 0.63 GiB and a 16 GiB prompt cache.

### Max-fit rows on the other three models (2026-10-28, cross-engine suite, thinking on)

Aggregate decode tok/s at N = 1 / 2 / 4 / 8 / 16 / 32; results `~/engine-baselines/tensorfold/<setting>/<run>/` on the box.

| Model, setting | Size | Streams, window, cache | N 1-32 | Box, run |
|---|---|---|---|---|
| Qwen3.8-27B, `qwen3.8-27b-maxfit` (1.1.7 binary, NVFP4 + DFlash2, fp8 KV) | 8k | 32, 12,288, 26 GiB | 44.5 / 82.5 / 124.3 / 186.1 / 202.5 / WindowTooWide | taiwan-home 20261028-0851 |
| | 16k | 32, 20,480, 22 GiB | 47.5 / 70.0 / 112.3 / 158.2 / 167.2 / WindowTooWide | taiwan-home 20261028-0701 |
| | 32k | 16, 38,912, 38 GiB | 44.0 / 73.5 / 102.7 / 128.3 / 145.1 / 16 streams | spark-e108 20261028-0757 |
| Nemotron 3.5 Bolt, `nemotron-3.5-bolt-maxfit` (MiaAI recipe 2cea422, 1.1.6 + libbolt) | 8k | 16, 40,960, 24 GiB | 83.5 / 124.9 / 181.7 / 238.0 / 292.4 / 16 streams | spark4 20261028-0700 |
| | 16k | 16, 40,960, 24 GiB | 83.7 / 118.1 / 179.9 / 236.3 / 276.9 / 16 streams | spark5 20261028-0656 |
| | 32k | 16, 40,960, 24 GiB | 78.7 / 124.8 / 165.8 / 223.0 / 265.2 / 16 streams | spark4 20261028-0737 |
| Nemotron 3.5 Super, `nemotron-3.5-super-maxfit` (1.1.7 binary, NVFP4, MTP) | 8k | 32, 12,288, 16 GiB (default) | 47.8 / 60.0 / 81.2 / 104.1 / 100.4 / 102.1 | spark-2f9d 20261028-0656 |
| | 16k | 32, 22,528, 24 GiB | 49.0 / 60.7 / 80.7 / 96.2 / 98.2 / 98.2 | spark-2f9d 20261028-0844 |
| | 32k | 16, 38,912, 24 GiB | 46.2 / 56.9 / 77.1 / 94.5 / 95.9 / 16 streams | spark5 20261028-0752 |

What limits each row:
- 27B: at 32 streams every request fails with `WindowTooWide` (the shared DFlash2 verify window is wider than 1.1.7
  takes), so 16 is its most. A kept prompt holds 2 full states (~0.6 / 1.2 / 2.2 GiB at 8k / 16k / 32k); cache hit
  88-89% (resumes at a 4,096-row chunk start).
- Bolt: libbolt is compiled for 16 sequences (`SEG_MAX`); memory is not the limit. libbolt maps KV out of free memory
  only, so the setting drops the page cache before a start.
- Super: the engine's default prompt cache is 16 GiB at most (49 snapshot slots); a kept prompt takes 2 slots, so 32
  need `--prompt-cache-gib 24` (64 slots). Throughput is flat from N 8 (shared rounds slow as streams are added).
  Windows must hold the largest prompt in Nemotron's tokenizer + 1,040: 16k's largest is 20,044 tokens.

## 7. Agent replay

`sharar-eval efficiency agent-replay` (method: `docs/agent-replay.md`) runs against TensorFold like any other engine.
The agent-replay size `agents` serves 16 streams with 65,536-token windows (the largest recorded turn is 57,587 tokens);
at more agents than streams the rest wait for a stream. Flash-Next uses its own setting:

```sh
TF_PARALLEL=16 TF_CONTEXT=65536 TF_RESERVE_GIB=9 TF_ENV=TF_FLASHNEXT_HEAD_KV=fp8 \
  engines/tensorfold/serve.sh up qwen3.8-flash-next-p32 ~/tf-agents
uv run sharar-eval efficiency agent-replay --engine tensorfold --engine-version "TensorFold 1.1.4 ..." \
  --url http://127.0.0.1:8888 --out ~/tf-agents/agents --agents 1,8,16 --tool-mode none
engines/tensorfold/serve.sh down ~/tf-agents
```

27B, Bolt and Super use their max-fit setting at size `agents`: `CONTEXT=agents engines/tensorfold/serve.sh up <model>-maxfit DIR`.

What decides its cache hit:
- TensorFold 1.1.4 (Flash-Next) and the Bolt / Super builds resume a conversation only at the exact end of the previous
  request. AA's SWE-bench harness ends every request with a "Tool-call turn k of 100" message that the next turn
  deletes, so those turns re-read their whole prompt (0% cached on them; 45-51% over all 8 tasks). Without that
  message they cache about 93%.
- TensorFold 1.1.7 on Qwen3.8-27B resumes only at multiples of 4,096 tokens, so it caches about 79% with or without it.
- The prompt cache is sized from the memory left after the model and the streams, so it differs by box (Flash-Next:
  8.8 GiB on spark3, 5.3 GiB on spark4, 1.9 GiB on spark-911c). With fewer conversation states than agents it evicts.

Measured numbers for all four models are in `campaigns/2026-10-29-agent-replay/` (campaign branch).

