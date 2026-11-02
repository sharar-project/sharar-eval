# Warm decode: how we measure decode speed with the prompts already cached

`sharar-eval efficiency warm-decode` measures how fast an engine decodes N requests together when every prompt is
already in its prefix cache, at a fixed prompt length. Prefill and time to first token are left out by design: the
timed window is the stretch where all N decode. This page is the method; per-engine launch is
[`engines/`](../engines/README.md), and every flag is listed in the README's Efficiency section.

## The cell

A cell is one (context, prompt kind, N):

1. **Flush** the engine's prefix cache (its flush endpoint from `bench_v1.json`, or `--flush-cmd`, a restart, for an
   engine without one).
2. **Prime**: send the N prompts one at a time with `max_tokens 1`, so the engine caches them (`--prime-passes 2` for
   an engine that keeps a prompt only once it has seen it twice, as vLLM's Nemotron recipes do).
3. **Time**: send the same N together through AIPerf, each forced to `--output-tokens` (default 1,024) with
   `ignore_eos`. The window runs from the last request's first token to the first request's last token.
4. **Check**: every request answered in full, none read less than `--min-hit` of its prompt from cache, and all N
   decoded at the same moment at least once. A failed check re-primes and runs once more.

Reported per cell: aggregate decode tok/s (tokens streamed in the window over its length), per-request decode tok/s
(median and slowest), the window's share of each request's decode, the lowest cache hit, the longest TTFT, and the
longest pause between two chunks of one request inside the window. Tokens are counted per streamed chunk with the
pinned tokenizer (scaled to the server's own count), so an engine that streams in bursts is counted when tokens arrive.

## When a row stops

A row (context, kind) climbs N = 1, 2, 4, ... and ends at the first N the engine cannot run as asked:

- **"decodes at most k of N at once"**: fewer than N requests were ever decoding together; the engine queued the rest
  behind a batch, slot or memory limit.
- **"stopped"**: the cache check failed twice; the engine could not keep N primed prompts at that length.

Queueing is checked before the cache hit (a queued request also loses its cache). Larger N are skipped.

## The cross-engine protocol (suite `cross-engine`)

`evals/workloads/efficiency_v1.json`, the same cells for every engine:

| Choice | Value |
|---|---|
| Prompts | SPEED-Bench's throughput split, `low_entropy` (code completion from repositories, text sorting) |
| Lengths | 8k, 16k and 32k tokens |
| N | 1, 2, 4, 8, 16, 32 |
| Prompt set | the same 32 prompts at every N (`--fixed-prompts`), drawn evenly from RepoBench Python, RepoBench Java and text sorting (`--balance`, seed 0; JetBrains LCA left out) |
| Bursts | N = 1 runs 32 bursts of one, N = 32 one burst of 32; every cell is timed at least 3 times in whole passes (`--min-bursts 3`) |
| Output | 1,024 tokens per request with `ignore_eos` |
| Sampling | the checkpoint's own: temperature 1.0, top-p 0.95, top-k 20, min-p 0, thinking on at effort `xhigh` |
| Cache-hit floor | 60% (vLLM's prefix cache keeps 80-95% of these hybrid-model prompts) |

Variants: `cross-engine-thinking-off` (thinking off), `cross-engine-n4` (N = 1, 2, 4, for a setting that decodes at
most 4 at once), `cross-engine-n64` (N = 64 on 64 prompts whose first 32 are cross-engine's), `smoke` (a few minutes).
`suite --context 16k` runs only some of a suite's lengths; an engine without a flush endpoint (TensorFold) runs each
length on a fresh server this way.

## Running it

```sh
uv run sharar-eval efficiency prepare --context 8k,16k,32k          # once per machine, no engine serving
engines/common/run.sh sharar qwen3.8-flash-next                     # reserve, start, check, suite cross-engine, stop
CONTEXT=16k engines/common/run.sh tensorfold qwen3.8-flash-next-maxfit
```

`run.sh` writes `~/engine-baselines/ENGINE/SETTING/<UTC time>-<host>/` with the engine's logs, `disk.log` (SSD reads,
GPU use, SM clock and free memory every 2 s), `suite/` (sharar-eval's run directory: `efficiency.json`, `run.json`,
`checks.json`, `warm/<context>-<kind>/c<N>/` with AIPerf's exports and `cell.json`) and `report.txt`. A rerun into
the same `OUT` skips finished cells; `report --recompute` rebuilds cells with the current maths.

Before comparing engines, check:

- **Each engine at its fastest setting per N.** Memory share changes how many requests fit, not their speed, so a
  figure takes each engine's best setting at each N (Sharar `--fast` or `--throughput`; vLLM's recipe at more memory
  or with fewer drafts; TensorFold's budget run or its max-fit run).
- **Checkpoints can differ.** On Qwen3.8-Flash-Next, Sharar serves RadixArk's NVFP4 build while vLLM and TensorFold
  serve NVIDIA's FP4-Local; speed differences there include the quantization.
- **The page cache.** Sharar and TensorFold read n-gram tables from disk on the first pass over new prompts; a run
  right after a reboot or next to a download measures the disk.
- **Label every number** with engine, setting, machine and suite.
