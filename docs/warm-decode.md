# Warm decode: how we measure decode speed with the prompts already cached

`sharar-eval efficiency warm-decode` measures how fast an engine decodes N requests together when every prompt is
already in its prefix cache, at a fixed prompt length. Prefill and time to first token are left out by design: the
timed window is the stretch where all N decode. This page is the method; how each engine is started is
[`engines/`](../engines/README.md), and every flag is in `uv run sharar-eval efficiency warm-decode --help` and the
README's Efficiency section.

## Quick start

```sh
uv run sharar-eval efficiency prepare --context 8k,16k,32k       # once per machine, with NO engine serving
engines/common/run.sh vllm qwen3.8-27b-d4                        # one engine x setting (engines/<engine>/settings/), suite cross-engine
less ~/engine-baselines/vllm/qwen3.8-27b-d4/<UTC time>-<host>/report.txt
```

`prepare` resolves SPEED-Bench's throughput files through AIPerf (it needs a Hugging Face login that accepted
[cais/hle](https://huggingface.co/datasets/cais/hle)'s terms; see the README's Bench suite section) and loads large
sources, so run it before an engine starts: on a GB10 it can push a serving engine under its memory guard. Its default
is `--context 8k,16k`; the suites need `32k` too. `run.sh` needs `uv` on `PATH`, refuses spark-911c, and reserves the
Spark (`~/.sp-claim`, the `sp` driver's file) for the whole run.

## The cell

A cell is one (context, prompt kind, N). Each burst of N prompts goes through:

1. **Flush** the engine's prefix cache: its flush endpoint from `evals/workloads/bench_v1.json` (Hibana
   `/flush_cache`, vLLM `/reset_prefix_cache`), or `--flush-cmd` (a shell command, e.g. a restart; sharar-eval then
   waits up to 15 minutes for `/v1/models`). An engine with neither (TensorFold) is not flushed: prompts primed for
   earlier bursts stay cached until the engine evicts them.
2. **Prime**: send the N prompts one at a time with one output token and the timed requests' sampling, so the engine
   caches them. `--prime-passes K` sends them K times, for an engine that keeps a prompt only once it has seen it
   more than once (vLLM's Nemotron recipes).
3. **Time**: send the same N together through AIPerf, each forced to `--output-tokens` (default 1,024) with
   `ignore_eos`. The window runs from the last request's first token to the first request's last token.
4. **Check**, in this order: every request answered (no errors); all N decoded at the same moment at least once; no
   request read less than `--min-hit` of its prompt from cache (default 0.95; the suites use 0.6); no answer came back
   short of `--output-tokens`. Any failure but the second (queueing) flushes, primes and times the burst once more;
   queueing and a priming error stop the row at once.

A cell's bursts: by default one burst of the first N prompts in file order. `--questions K` runs ceil(K/N) bursts of N
different prompts and reports their mean (each burst under `rounds` in `cell.json`). `--fixed-prompts` keeps every N
on the same K prompts (N above K is not run); `--min-bursts M` (needs `--fixed-prompts`) repeats whole passes over them
until the cell has at least M bursts. `--balance` draws the prompts evenly from the kind's source datasets: each source
shuffled with `--prompt-seed` (default 0), the sources taken in turn, so every length gets the same mix;
`--exclude-sources` leaves some out. `--max-tokens-in-flight 256k` skips cells whose N x context is larger.

Sampling by default is the checkpoint's own: temperature 1.0, top-p 0.95, top-k 20, min-p 0, thinking on at
`reasoning_effort` `xhigh`. `--temperature` (0 is greedy: no top-p, top-k or min-p sent), `--top-p` and `--reasoning
off|low|medium|xhigh` change it; priming and timed requests send the same.

## What is reported

One row per cell (`report.txt`, `efficiency report`), from AIPerf's per-request and raw (per-chunk) exports:

| Column | Definition |
|---|---|
| aggregate | tokens streamed inside the window / window length (tok/s, all N together) |
| per stream, slowest | each request's (output tokens - 1) / its first to last token, median and slowest (its whole decode, not only the window) |
| window s, share | the window's length, and that over the median request's first-to-last-token time |
| hit min | lowest prompt share read from cache (`usage.prompt_tokens_details.cached_tokens`); "-" when the server does not report it, and then nothing confirms prefill was skipped |
| TTFT max | longest time to first token in the timed burst |
| pause max | longest gap between two chunks of one request inside the window (an engine that sets a request aside shows it here; it already lowers both rates) |
| prompt, out min | mean prompt tokens and fewest output tokens, as the server counts them |

Tokens are counted per streamed chunk: each chunk's text in the pinned tokenizer, scaled so a request's chunks add up
to the server's count, so an engine that streams in bursts is counted when the tokens arrive. A request whose visible
text holds under 90% of the server's count (special tokens after the model's own end, not streamed) or whose stream is
held back to its end is spread evenly over its whole decode instead; the row then ends "(k held back)". A cell of
several bursts reports the mean of their rates and windows, the minimum hit and slowest stream, and the maximum TTFT
and pause over all bursts. Where the server reports speculation counters (Hibana's `/get_server_info` `spec_stats`),
`cell.json` also keeps the share of drafted tokens accepted and the tokens emitted per request per verify round.

## When a row stops

A row (context, kind) runs its N in order and ends at the first N the engine cannot run as asked; larger N of that row
are not run, the other rows are. The report prints `N stopped: <reason>`. The two that mean an engine limit:

- **"decodes at most k of N at once: the engine queued the rest"**: fewer than N requests were ever decoding together
  (a batch, slot or memory admission limit). This stops the row at once, without a retry, and is checked before the
  cache hit (a queued request often loses its cache while it waits; the reason then says so).
- **"cache hit x% under y%: the primed prompts did not stay cached"**, on both tries: the engine could not keep N primed
  prompts at that length.

Otherwise: "k of N requests failed", "answers came back short (ignore_eos not honoured?)", "priming failed" or "decode
failed". If the server stops answering, the run ends there.

## The cross-engine protocol (suite `cross-engine`)

`evals/workloads/efficiency_v1.json` (version `efficiency-v1`), the same cells for every engine:

| Choice | Value |
|---|---|
| Prompts | SPEED-Bench's throughput split, `low_entropy` (code completion from repositories, text sorting) |
| Lengths | 8k, 16k and 32k (SPEED-Bench's buckets; the `prompt` column shows each model's own count) |
| N | 1, 2, 4, 8, 16, 32 |
| Prompt set | the same 32 prompts at every N (`--questions 32 --fixed-prompts`), drawn evenly from RepoBench Java, RepoBench Python and text sorting (`--balance`, `--prompt-seed 0`, JetBrains LCA excluded: 6 prompts at 8k against 92 at 32k) |
| Bursts | a pass over the 32 is 32/N bursts; `--min-bursts 3` repeats passes to at least 3 bursts: N = 1: 32, 2: 16, 4: 8, 8: 4, 16: 4 (two passes), 32: 3 (three passes) |
| Output | 1,024 tokens per request with `ignore_eos` |
| Sampling | the checkpoint's own: temperature 1.0, top-p 0.95, top-k 20, min-p 0, thinking on at `xhigh` |
| Cache-hit floor | 60% (`--min-hit 0.6`; vLLM's prefix cache keeps 80-95% of these hybrid-model prompts) |

Variants, the same cells otherwise:

| Suite | Change |
|---|---|
| `cross-engine-thinking-off` | thinking off (`reasoning off`), sampling unchanged |
| `cross-engine-n4` | N = 1, 2, 4 only, for a setting that decodes at most 4 at once (Hibana's `-fast` settings) |
| `cross-engine-n64` | N = 64 only, one burst of 64 balanced prompts whose first 32 are cross-engine's; needs a 64-sequence setting (vLLM `-s64`, Hibana `-b64`) |
| `smoke` | a few-minute check: 8k, N = 1 and 4, 256 tokens, the first prompts in file order, one burst, floor 60% |

`suite --context 16k` (or `8k,32k`) runs only those of a suite's lengths, the same cells; `suite.json` records it.
`suite --prime-passes`, `--min-hit` (overrides the suite's floor) and `--flush-cmd` pass to warm-decode: they say how
the engine caches, not what is measured.

## How each engine is run

`engines/common/run.sh ENGINE SETTING [SUITE]` (default `cross-engine`) starts the engine in one setting
(`engines/ENGINE/settings/SETTING.env`, named after the model), sends one request with the suite's sampling and logs
whether the engine thought (`sampling-probe.json`), then runs

llama.cpp fixes its slot count at start, so it runs one server per N with exactly N slots
(`engines/llamacpp/warm.sh SETTING`, the same `cross-engine` options; 16k only in the published rows).

```sh
sharar-eval efficiency suite --suite SUITE --engine $EVAL_ENGINE --engine-version "<versions.env label>; setting SETTING: <its description> (<host>)" \
  --url $URL --out $OUT/suite [--context $CONTEXT] [--flush-cmd ...] [--prime-passes $PRIME_PASSES] [--min-hit $MIN_HIT]
```

and `efficiency report $OUT/suite > $OUT/report.txt`. The knobs that change the procedure come from the setting file
or the environment:

| Knob | Set by | Effect |
|---|---|---|
| `EVAL_ENGINE` | `versions.env` | `sharar` and `vllm` flush through their endpoint before every burst; `tensorfold` has none and is not flushed |
| `PRIME_PASSES` | vLLM `nemotron-3.5-bolt*`, `nemotron-3.5-super` (2), `nemotron-3.5-super-d3` (3) | `--prime-passes`: the recipes' Mamba shared-prefix checkpoints keep a prompt only once vLLM has seen it twice |
| `MIN_HIT` | vLLM `nemotron-3.5-bolt*` and `nemotron-3.5-super-d3` (0.5) | `--min-hit`: they keep Mamba state only every few thousand tokens (Bolt: 54% of an 8k prompt), under the suite's 60% |
| `CONTEXT` | environment, e.g. `CONTEXT=16k` | `suite --context`: TensorFold runs one length per run, so each length gets a fresh server (no flush keeps one length's prompts out of the next) and `-maxfit` settings are sized per length |
| `FLUSH_RESTART=1` | environment | `--flush-cmd` restarting the engine before every burst; not used for the published runs (TensorFold ran unflushed) |
| `OUT` | environment | resume an earlier run directory: finished cells and stopped rows are skipped; the options (suite, `CONTEXT`, host in the version label) must match |

```sh
engines/common/run.sh sharar qwen3.8-flash-next                                  # Hibana --throughput
engines/common/run.sh sharar qwen3.8-flash-next-fast cross-engine-n4             # Hibana --fast
engines/common/run.sh vllm nemotron-3.5-bolt                                     # primes twice, floor 0.5
for c in 8k 16k 32k; do CONTEXT=$c engines/common/run.sh tensorfold qwen3.8-flash-next-maxfit; done
```

Each engine's README says which settings the cross-engine runs used and why.

## Results

`run.sh` writes `~/engine-baselines/ENGINE/SETTING/<UTC time>-<host>/`:

- `run.log`, `suite.txt` (sharar-eval's output), the engine's logs, `sampling-probe.json`, `disk.log` (SSD reads, GPU
  use, SM clock and free memory every 2 s) and `report.txt`;
- `suite/suite.json` (suite, version, `--context`) and `suite/warm-decode/`: `efficiency.json` (the options; a rerun
  must match them), `run.json` (engine, version label, server description, resolved prompt-file hashes, code hash),
  `checks.json` (the pre-flight `sharar-eval check`: a failure is printed and the cells run anyway, so read it),
  and `warm/<context>-<kind>/`: `c<N>/cell.json` (the cell's summary) with the first burst's AIPerf exports,
  `c<N>-r<k>/` for later bursts, `prime-c<N>/` (`prime2-...` for extra passes) and `stopped.json` where the row ended.

`efficiency report DIR [DIR ...]` prints the tables again. `report --recompute DIR` rebuilds each `cell.json` from its
AIPerf records with the current maths: every burst of a cell (`c<N>/`, `c<N>-r<k>/`) is rebuilt and the bursts are
merged as the run merged them (speculation counters, which come from the server, are kept per burst).

## Before comparing engines

- **Each engine at its fastest setting per N.** Memory share changes how many requests fit, not their speed, so a
  figure takes each engine's best setting at each N (Hibana `--fast` at N <= 4 or `--throughput`; vLLM's recipe at
  more memory or with fewer drafts; TensorFold's budget setting or its max-fit one).
- **Checkpoints can differ.** On Qwen3.8-Flash-Next, Hibana serves RadixArk's NVFP4 build while vLLM and TensorFold
  serve NVIDIA's FP4-Local; speed differences there include the quantization.
- **The page cache.** Hibana and TensorFold read n-gram tables from disk on the first pass over new prompts, so the
  first bursts can be slower while rows come off the SSD (`disk.log`). A run right after a reboot, or next to a
  download, measures the disk.
- **The box.** One engine per Spark, never next to a download (it also starves the engine's memory), and not on
  spark-911c (desktop and Claude Code; `AVOID_HOSTS`).
- **Check the run.** `checks.json` passed (above all `ignore_eos`), `sampling-probe.json` shows the thinking you
  expect, and hit min is not "-".
- **Label every number** with engine, setting, machine and suite.
