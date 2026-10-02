# sharar-eval

Engine-agnostic evaluations for local inference servers on the GB10. Everything here
talks HTTP (OpenAI chat completions) and imports nothing from the engine, so the same
commands measure Sharar, SGLang, vLLM or anything else with that API. It was split out
of `abdelfattah-lab/sharar` on 2026-09-26 (sharar#53). The engine keeps its in-process
benchmarks and gates, and runs these evals against its served arms.

## Evaluate a server in one command

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m evals.datasets fetch          # pinned datasets; GPQA is gated (HF_TOKEN)

    .venv/bin/python -m evals.suite run --url http://127.0.0.1:30004 --label fast --suite quick \
        --out runs/fast --engine "sharar:main-20261001 sha256:b615f4ecfda0"
    .venv/bin/python -m evals.suite compare runs/exact runs/fast

    # any tasks, first N items of each (a later, longer run into the same directory only adds the rest)
    .venv/bin/python -m evals.suite run --label fast --tasks gpqa,mmlupro --limit 50 --out runs/fast

| Suite | Tasks | Time per arm (Flash-Next, 4 streams) |
|---|---|---|
| `quick` | GSM8K first 300, HumanEval 164, MGSM 24 per language (264), needles at 16K and 64K | ~1 h |
| `full` | GSM8K 1,319, HumanEval 164, MBPP 500, MGSM 2,750 | ~4 h |
| `hard` | MMLU-Pro 1,400 (100 per category), GPQA Diamond 198, IFEval 541 (strict), WMT24++ 1,000 (English into 10 languages, chrF++) | 4-6 h |
| `all` | `full` + `hard` + needles | ~10 h |

- **Tasks:** gsm8k, humaneval, mbpp, mgsm, mgsm-pl24, mmlupro, gpqa, ifeval, wmt, needles.
  `--tasks` picks any of them instead of a suite, `--limit N` takes the first N items of each,
  and `--skip` leaves one out.
- **One run directory per server configuration.** It holds 100-item shards per task, so
  an interrupted run resumes where it stopped, and `full` reuses the GSM8K shards `quick` wrote.
- **`run.json` records the run.** It holds the server's `/get_server_info` and the engine
  version given with `--engine`. It also holds this repository's commit, the sha256 of every
  dataset file read, and each task's token cap. A later `run` into the same directory must match it.
- **`compare` prints one scorecard, paired item by item.** It covers every task both runs have.
  - Binary tasks: McNemar and a 95% normal interval of the paired difference.
  - WMT24++: a paired bootstrap of chrF++.
  - Needles: counts.
  - Pooled: all binary tasks together.
  - `--detail` adds per-category and per-language breakdowns; `--json` writes the scorecard.
- **Requests are greedy with thinking off.** Every arm sends identical prompts. By default
  the arm's own profile sets the acceptance rule (`--accept server`); `--accept 1:0` pins exact.
  For another engine, pass `--plain --model NAME`.
- **Repeatability:** at 4 streams a Sharar server is not item-for-item repeatable, even with
  exact acceptance, because batch composition changes the numerics. At 1 stream it is. Two
  runs of the same configuration therefore differ on a few items; that is part of the noise
  the paired tests measure.

## Modules

| Module | What it measures |
|---|---|
| `evals.suite` | The suites above: `run` and `compare` |
| `evals.accuracy_eval` | GSM8K and MGSM (11 languages): exact match. HumanEval and MBPP: pass@1, sandboxed |
| `evals.hard_eval` | MMLU-Pro, GPQA Diamond, IFEval (reference checker vendored in `evals/ifeval/`), WMT24++ (sharar#28) |
| `evals.long_context` | Needle retrieval and prefill/decode rates at 10K-262K |
| `evals.h2h`, `evals.h2h_report` | Head-to-head serving benchmark: public probes, 22 decode prompts (greedy, t=0.7, t=1.0), long-context TTFT and needles, four-stream aggregate (`evals/workloads/h2h_v1.json`) |
| `evals.identity` | Exact greedy output identity between two servers (the refactor gate: `run` twice, then `compare`) |
| `evals.tts_eval` | Test-time scaling on MATH-500: greedy against self-consistency with n choices |
| `evals.datasets` | `fetch` and `check` the pinned datasets |

Each task module also runs alone, e.g. `python -m evals.hard_eval --task gpqa --label x --output DIR`,
and has a `compare` subcommand that works on its own result files.

## Datasets

Data stays out of git, under `$SHARAR_RESULTS/datasets/` (default `/home/yke/sharar-results/datasets/`).
`evals/datasets.py` names each file's pinned source, and [datasets.sha256](datasets.sha256) pins
its hash. Every suite run checks the files it reads.

- **MBPP:** the test split is task ids 11-510. The reference solution of one task fails its own asserts.
- **Needle corpus:** CPython 3.12.7's `Lib/*.py` in path order, cut at about 3 MB (about 832K tokens).
  The tokenizer is Qwen3.8-27B's `tokenizer.json`.

## Recording results

Every result names both versions: this repository's commit and the engine's (image tag and
commit, or the other engine's pinned image digest), plus the model revision and the serving
configuration. `evals.suite` records all of these in `run.json`; pass `--engine`.
