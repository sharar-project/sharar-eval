# Engines: how to start each one for the benchmarks

sharar-eval measures any OpenAI-compatible server already running at `--url`; it never starts or stops an engine.
This folder is what starts them: one folder per engine with every pinned version, its settings (one per model plus
variants), and the script that starts and stops it, plus one runner, [`common/run.sh`](common/run.sh), that
benchmarks every engine the same way on a DGX Spark (GB10, about 121 GiB of shared memory).

```sh
engines/common/run.sh ENGINE SETTING [SUITE]          # SUITE defaults to cross-engine
engines/common/run.sh vllm nemotron-3.5-bolt                       # warm decode, suite cross-engine
engines/common/run.sh sharar nemotron-3.5-bolt-fast cross-engine-n4   # Hibana --fast: N = 1, 2, 4
engines/common/run.sh vllm qwen3.8-27b-d4 smoke                    # a few-minute check
CONTEXT=16k engines/common/run.sh tensorfold qwen3.8-flash-next    # the suite's 16k cells only (suite --context)
engines/common/run.sh sharar qwen3.8-flash-next cross-engine-agents   # the agent replay instead of warm decode
```

What the suites measure: [`docs/warm-decode.md`](../docs/warm-decode.md) (decode with every prompt cached; suites
`cross-engine`, `cross-engine-n4`, `cross-engine-n64`) and [`docs/agent-replay.md`](../docs/agent-replay.md)
(recorded coding-agent tasks; suite `cross-engine-agents`: 1, 8, 16 and 32 agents, the thinking-off setting).

`run.sh` reserves the Spark (`~/.sp-claim`, the file the `sp` driver uses), starts the engine with its memory guard,
sends one request with the suite's sampling and logs whether the engine thought, runs `sharar-eval efficiency suite`,
writes the report, stops the engine and releases the Spark. Results go to
`~/engine-baselines/ENGINE/SETTING/<UTC time>-<host>/` (`OUT_ROOT`): `run.log`, the engine's logs, `disk.log` (SSD
reads, GPU use, SM clock and free memory every 2 s), `suite/` (sharar-eval's run directory) and `report.txt`.
Environment: `CONTEXT` (warm decode at only these sizes, e.g. `16k`; TensorFold's `-maxfit` settings also size their
server by it, `agents` for the agent replay), `OUT=<an earlier run's directory>` (resume; finished cells are skipped),
`RUN_ON_ANY_HOST=1` (see the rules). An engine's own variables (`TF_PARALLEL`, `SHARAR_W3_CHECKPOINT`, ...) are read
from the environment too.

## Baselines

The settings behind the published cross-engine numbers. Warm decode: suite `cross-engine` unless marked; figures take
each engine's fastest setting at each N. Agent replay: suite `cross-engine-agents`.

| Model | Warm decode: Hibana | vLLM | TensorFold | llama.cpp (16k) | Agent replay: Hibana | vLLM | TensorFold | llama.cpp |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | `qwen3.8-27b`; `-fast` (n4); `-b64` (n64) | `qwen3.8-27b-d4` (`qwen3.8-27b`, 8 drafts, also run); `-d4-s64` (n64) | `qwen3.8-27b-maxfit`, one `CONTEXT` per run | `qwen3.8-27b-dflash2`, `warm.sh` | `qwen3.8-27b` | `qwen3.8-27b-d4` | `qwen3.8-27b-maxfit`, `CONTEXT=agents` | `qwen3.8-27b-dflash2`, `agents.sh` |
| Qwen3.8-Flash-Next | `qwen3.8-flash-next`; `-fast` (n4); `-b64` (n64) | `qwen3.8-flash-next-d2` | `qwen3.8-flash-next` (budget) and `-maxfit`, one `CONTEXT` per run | `qwen3.8-flash-next-p`, `warm.sh` (N 1-8) | `qwen3.8-flash-next` | `qwen3.8-flash-next-d2` | `qwen3.8-flash-next-p32` with `TF_PARALLEL=16 TF_CONTEXT=65536 TF_RESERVE_GIB=9 TF_ENV=TF_FLASHNEXT_HEAD_KV=fp8` | `qwen3.8-flash-next-p`, `agents.sh` (32 agents on 16 slots) |
| Nemotron 3.5 Bolt | `nemotron-3.5-bolt` and `-w3` (3-bit copies; ran as `nemotron-3.5-bolt-w3hf`, 20261029-2311-spark-e108), fastest per N; `-fast` (n4); `-b64` (n64) | `nemotron-3.5-bolt`; `-s64` (n64) | `nemotron-3.5-bolt-maxfit`, one `CONTEXT` per run | `nemotron-3.5-bolt-gguf`, `warm.sh` | `nemotron-3.5-bolt-w3` | `nemotron-3.5-bolt-m086` | `nemotron-3.5-bolt-maxfit`, `CONTEXT=agents` | `nemotron-3.5-bolt-gguf`, `agents.sh` |
| Nemotron 3.5 Super | `nemotron-3.5-super`; `-fast` (n4); `-b64` (n64) | `nemotron-3.5-super-d3` (`nemotron-3.5-super`, 7 drafts, also run) | `nemotron-3.5-super-maxfit`, one `CONTEXT` per run | `nemotron-3.5-super-gguf`, `warm.sh` (N 1-16) | `nemotron-3.5-super` | `nemotron-3.5-super-d3` (the note-removed replay ran `-d3-m084`: the box could not free 0.86) | `nemotron-3.5-super-maxfit`, `CONTEXT=agents` | `nemotron-3.5-super-gguf`, `agents.sh` (32 agents on 16 slots) |

(n4) = suite `cross-engine-n4`, (n64) = suite `cross-engine-n64`. llama.cpp runs one server per N
(`engines/llamacpp/warm.sh`, 16k only) and per agent count (`engines/llamacpp/agents.sh`): see its README. Every other setting in the engine folders is an
experiment or a fallback for a Spark with less free memory; each README says which. Copy-paste:

```sh
uv run sharar-eval efficiency prepare --context 8k,16k,32k        # once per Spark, with no engine serving
engines/common/run.sh sharar qwen3.8-27b                           # warm decode, N 1-32
engines/common/run.sh sharar qwen3.8-27b-fast cross-engine-n4
engines/common/run.sh sharar qwen3.8-27b-b64 cross-engine-n64
engines/common/run.sh vllm qwen3.8-27b-d4
for c in 8k 16k 32k; do CONTEXT=$c engines/common/run.sh tensorfold qwen3.8-27b-maxfit; done
engines/common/run.sh sharar qwen3.8-27b cross-engine-agents        # agent replay
engines/common/run.sh vllm qwen3.8-27b-d4 cross-engine-agents
CONTEXT=agents engines/common/run.sh tensorfold qwen3.8-27b-maxfit cross-engine-agents
TF_PARALLEL=16 TF_CONTEXT=65536 TF_RESERVE_GIB=9 TF_ENV=TF_FLASHNEXT_HEAD_KV=fp8 \
  engines/common/run.sh tensorfold qwen3.8-flash-next-p32 cross-engine-agents
CONTEXT=16k engines/llamacpp/warm.sh qwen3.8-27b-dflash2           # llama.cpp: one server per N
engines/llamacpp/agents.sh qwen3.8-27b-dflash2                     # llama.cpp agent replay, 1/8/16/32 agents
```

`run.sh` replays the published traces only. The replay with the SWE-bench turn messages removed (docs/agent-replay.md)
is run by hand: `serve.sh up SETTING DIR`, then `uv run sharar-eval efficiency agent-replay ... --replay <that copy>
--thinking off --tool-choice unset`, then `serve.sh down DIR`.

## The engines and models

| Model | [Hibana](sharar/README.md) (then Sharar; main `7387a45`) | [vLLM](vllm/README.md) (eugr's recipes `b62168e`) | [TensorFold](tensorfold/README.md) | [llama.cpp](llamacpp/README.md) (`547208d`) |
|---|---|---|---|---|
| Qwen3.8-27B | RadixArk/Qwen3.8-27B-NVFP4 `319f741` + 3-bit copies, DFlash2 | nvidia/Qwen3.8-27B-NVFP4 `482ca0f`, z-lab DFlash2 `50307d4` | v1.1.7 release binary, nvidia `482ca0f` + z-lab DFlash2 `50307d4` | ggml-org Q4_K_M GGUF `a316ea5` + z-lab DFlash2 `50307d4` converted to GGUF |
| Qwen3.8-Flash-Next | RadixArk/Qwen3.8-Flash-Next-NVFP4 `7b71922` (NVFP4 throughout, 126 GB) + 3-bit copies, MTP | nvidia/Qwen3.8-Flash-Next-FP4-Local `b9ce18a` (mixed MXFP8 / NVFP4, 96 GB), MTP | v1.1.4 via MiaAI-Lab's recipe `fd3f43e`, nvidia FP4-Local `b9ce18a`, MTP | ISTA-DASLab GSQ-RCO IQ3_S GGUF `ed59f92`, ggml-org MTP head `92796c0` |
| Nemotron 3.5 Bolt | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB `cf2effc`, LiLiCorr DFlash; `-w3`: + 3-bit copies | the same `cf2effc`, LiLiCorr `449d241` | v1.1.6 via MiaAI-Lab's Bolt recipe `2cea422`, revision `a326822`, LiLiCorr `4c6a561` | nvidia's IQ2 GGUF `e0fecb6`, LiLiCorr GGUF `182d298` |
| Nemotron 3.5 Super | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 `6aae0d6` + 3-bit copies, MTP | the same `6aae0d6`, LiLiCorr `292bf1e` | v1.1.7 release binary, the same `6aae0d6`, MTP | ggml-org Q4_0 GGUF `99144d1`, its LiLiCorr NVFP4 drafter |

Weights: the 27B, Bolt and Super rows serve the same weights in every engine except llama.cpp, which cannot load these
NVFP4 checkpoints and serves the GGUFs named above (Bolt `a326822` and `cf2effc` have
identical weight files; only README.md and explainability.md differ). Flash-Next does not: Hibana serves RadixArk's
all-NVFP4 build, vLLM and TensorFold NVIDIA's FP4-Local (MXFP8 attention, mixer and most experts, NVFP4 elsewhere), so
speed differences there include the quantization. Hibana's 3-bit copies are extra decode-time copies of the trunk
(and, for Flash-Next and Super, experts), calibrated per target; Hibana fetches them for 27B, Flash-Next and Super, and
publishes none for Bolt (hence `nemotron-3.5-bolt-w3`).

KV cache: Hibana NVFP4 (27B, Flash-Next) or FP8 (Bolt, Super); vLLM FP8; TensorFold FP8 (Flash-Next, 27B), BF16
(Bolt, the recipe's), the engine's default (Super); llama.cpp q8_0.

Every engine decodes at least 32 requests at once, the suite's largest N: vLLM's recipes run 8 sequences and Hibana's
`--throughput` 16 (Bolt 4) at this commit, so both are raised to 32. Beyond that, each setting names its changes from
the recipe or profile (vLLM: memory share and draft count; TensorFold: streams, window, reserve).

## Rules every baseline follows

- **One engine on one Spark at a time.** Two engines on one GB10 run it out of memory. `run.sh` refuses a Spark that
  someone reserved, or whose GPU is busy without a reservation; `serve.sh up` refuses a busy GPU.
- **Nothing else heavy on the Spark, above all no download.** A download fills the page cache, which on GB10 is the
  engine's memory: vLLM then refuses its memory share, TensorFold sizes a smaller prompt cache, and a running Hibana
  was killed by its guard. Download checkpoints before a run, not beside one.
- **A memory guard starts before the engine loads.** On GB10 an engine's GPU memory is not in its RSS: the kernel's
  OOM killer cannot free it, and the machine can hang until it is power-cycled.
- **Stop an engine with its `serve.sh down`.** Hibana and TensorFold's native binary run under `setsid`, vLLM and the
  TensorFold recipes in a container: all outlive a `run.sh` that is killed (e.g. by `sp stop`). Then run
  `engines/ENGINE/serve.sh down <the run's directory>`, and check `nvidia-smi` is empty before the next start.
- **Engines that size caches from free memory differ by box.** TensorFold's prompt cache and vLLM's KV share depend on
  `MemAvailable` at start (spark-e108 has 119.7 GiB, others about 121); label results with the machine.
- **The same suite for every engine.** `cross-engine`: SPEED-Bench throughput prompts (low entropy) at 8k, 16k and 32k
  tokens, the same 32 balanced prompts at every N = 1-32, primed then sent together, 1,024 tokens with `ignore_eos`,
  only the stretch where all N decode timed, at least three bursts per cell. A row ends at the first N the engine
  cannot run as asked, and the report says why.
- **`sharar-eval check` must pass** (the suite runs it first). An engine that ignores `ignore_eos` cannot be measured
  this way.
- **Not on spark-911c.** It runs the desktop and the Claude Code sessions that drive the other Sparks, so its free
  memory and page cache are not the engine's own (`AVOID_HOSTS`; `RUN_ON_ANY_HOST=1` overrides).
- **Label every number with its engine, setting and machine**, and keep raw results out of `/tmp`.

## One-time setup on a Spark (every engine)

- `uv` on `PATH`, and this checkout synced (`uv sync`); then the prompt sets, with no engine serving:
  `uv run sharar-eval efficiency prepare --context 8k,16k,32k`. The agent-replay traces are fetched on first use.
- Passwordless `sudo` is used only to drop the page cache before a start (settings with `DROP_PAGE_CACHE=1`); without
  it the start goes on and logs that it could not.
- Disk: checkpoints 27B ~21 GB, Flash-Next 126 GB (RadixArk, Hibana) / 96 GB (FP4-Local, vLLM and TensorFold), Bolt
  ~23 GB, Super ~70 GB; vLLM images ~20 GB each; TensorFold's Flash-Next image ~50 GB.
- Then the engine's README, section 2.

## Layout

```
engines/
  README.md                this index
  common/
    run.sh                 the runner: reserve, serve.sh up, sharar-eval efficiency suite, report, serve.sh down, release
    defaults.env           the suite, its sampling, the memory guard, hosts to avoid, result folders
    lib.sh                 reservation, memory guards, health wait, sampling probe
    dlog.sh                SSD reads, GPU use, SM clock and free memory every 2 s
    ENGINE_FOLDER.md       what an engine folder holds
  <engine>/
    README.md              the launch guide
    versions.env           what every setting shares: commits, image digests
    settings/<model>[-variant].env   one base per model, named the same in every engine, plus variants
    serve.sh               serve.sh up SETTING DIR / serve.sh down DIR
```

## Adding an engine or a model

An engine: copy the folder that resembles it, follow [`common/ENGINE_FOLDER.md`](common/ENGINE_FOLDER.md), run
`common/run.sh NEW MODEL smoke` once to check it, and add a column above. A model: add `settings/<model>.env` to each
engine that serves it, with the same name everywhere, and a row above.
