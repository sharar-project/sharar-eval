# Engines: how to start each one for the benchmarks

sharar-eval measures any OpenAI-compatible server already running at `--url`; it never starts or stops an engine.
This folder is what starts them: one folder per engine with every pinned version, one setting per model, and the
script that starts and stops it, plus one runner, [`common/run.sh`](common/run.sh), that benchmarks every engine the
same way on a DGX Spark (GB10, about 121 GiB of shared memory).

```sh
engines/common/run.sh ENGINE SETTING [SUITE]
engines/common/run.sh vllm nemotron-3.5-bolt            # vLLM, eugr's recipe for Bolt, suite cross-engine
engines/common/run.sh sharar nemotron-3.5-bolt          # Sharar on the same model
engines/common/run.sh vllm qwen3.8-27b smoke            # a few-minute check
```

`run.sh` reserves the Spark (`~/.sp-claim`, the file the `sp` driver uses), starts the engine with its memory guard,
sends one request with the suite's sampling and logs whether the engine thought, runs `sharar-eval efficiency suite`
(default `cross-engine`), writes the report, stops the engine and releases the Spark. Results go to
`~/engine-baselines/ENGINE/SETTING/<UTC time>-<host>/`: `run.log`, the engine's logs, `disk.log` (SSD reads, GPU use,
SM clock and free memory every 2 s), `suite/` (sharar-eval's run directory) and `report.txt`. `OUT=<that directory>`
resumes a run; finished cells are skipped. Read the engine's README first: setup is per engine, and `uv` must be on
`PATH`.

## The engines and models

| Model | [vLLM](vllm/README.md) (eugr's recipe) | [Sharar](sharar/README.md) |
|---|---|---|
| `qwen3.8-27b` | nvidia/Qwen3.8-27B-NVFP4, DFlash2 8 drafts | RadixArk/Qwen3.8-27B-NVFP4, `--throughput`, max batch 32, drafts 15/10/6 by batch |
| `qwen3.8-flash-next` | nvidia/Qwen3.8-Flash-Next-FP4-Local, MTP 4 drafts (B12X image) | RadixArk/Qwen3.8-Flash-Next-NVFP4, `--throughput` as shipped |
| `nemotron-3.5-bolt` | Bolt 74B-A9B-24GB, LiLiCorr DFlash 7 drafts | the same checkpoint, `--throughput`, max batch 32 |
| `nemotron-3.5-super` | Super VL 120B-A12B NVFP4, LiLiCorr DFlash 7 drafts | the same checkpoint, `--throughput`, max batch 32 |

Every engine is set to decode at least 32 requests at once, the suite's largest N: vLLM's recipes run 8 sequences
and Sharar's `--throughput` 16 for these models, so both are raised to 32 and nothing else is changed. The
Flash-Next rows serve different checkpoints (NVIDIA's FP4-Local in vLLM's recipe, RadixArk's NVFP4 in Sharar), so
speed differences there include the quantization.

## Rules every baseline follows

- **One engine on one Spark at a time.** Two engines on one GB10 run it out of memory. `run.sh` refuses a Spark that
  someone reserved, or whose GPU is busy without a reservation.
- **A memory guard starts before the engine loads.** On GB10 an engine's GPU memory is not in its RSS: the kernel's
  OOM killer cannot free it, and the machine can hang until it is power-cycled.
- **The same suite for every engine.** `cross-engine`: SPEED-Bench throughput prompts (low entropy) at 8k, 16k and 32k
  tokens, the same 32 balanced prompts at every N = 1-32, primed then sent together, 1,024 tokens with `ignore_eos`,
  only the stretch where all N decode timed, at least three bursts per cell. A row ends at the first N the engine
  cannot run as asked, and the report says why.
- **`sharar-eval check` must pass** (the suite runs it first). An engine that ignores `ignore_eos` cannot be measured
  this way.
- **Not on spark-911c.** It runs the desktop and the Claude Code sessions that drive the other Sparks, so its free
  memory and page cache are not the engine's own (`AVOID_HOSTS`; `RUN_ON_ANY_HOST=1` overrides).
- **Label every number with its engine, setting and machine**, and keep raw results out of `/tmp`.

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
    settings/<model>.env   one per model, named the same in every engine
    serve.sh               serve.sh up SETTING DIR / serve.sh down DIR
```

## Adding an engine or a model

An engine: copy the folder that resembles it, follow [`common/ENGINE_FOLDER.md`](common/ENGINE_FOLDER.md), run
`common/run.sh NEW MODEL smoke` once to check it, and add a column above. A model: add `settings/<model>.env` to each
engine that serves it, with the same name everywhere, and a row above.
