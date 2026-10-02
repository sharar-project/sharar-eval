# sharar-eval

Engine-agnostic evaluations for local inference servers on the GB10: everything
here talks HTTP (OpenAI chat completions) and imports nothing from the engine, so
the same scripts measure Sharar, SGLang, vLLM or anything else with that API.
Split out of `abdelfattah-lab/sharar` on 2026-09-26 (sharar#53); the engine keeps
its in-process benchmarks and the family gate.

| Module | What it measures |
|---|---|
| `evals.h2h` | Head-to-head serving benchmark: public probes, 22 decode prompts (greedy, t=0.7, t=1.0), long-context cold/warm TTFT and needles, four-stream aggregate (`evals/workloads/h2h_v1.json`) |
| `evals.h2h_report` | Tabulate h2h result files side by side |
| `evals.identity` | Exact greedy output identity between two servers (the refactor gate: `run` twice, then `compare`) |
| `evals.accuracy_eval` | GSM8K and MGSM (11 languages) exact match, HumanEval and MBPP pass@1 (sandboxed), acceptance pinned per request or left to the server's profile (`--accept server`); `compare` pairs arms with McNemar |
| `evals.tts_eval` | Test-time scaling on MATH-500: greedy, self-consistency with n choices |
| `evals.long_context` | Needle retrieval and prefill/decode rates at 10K-262K |
| `evals.hard_eval` | Harder, less saturated tasks (sharar#28): MMLU-Pro (1,400 items, 14 categories), GPQA Diamond, IFEval (strict, reference checker vendored in `evals/ifeval/`), WMT24++ English into 10 languages (chrF++); paired McNemar / bootstrap `compare`. Needs `.venv-evals` (`requirements-hard.txt`); `fetch` writes the pinned datasets |

Run from the repository root, e.g.

    python -m evals.h2h --url http://127.0.0.1:30004 --engine sharar-flashnext --workloads probes,decode --modes greedy,t07 --out runs/x.jsonl
    python -m evals.identity run --url http://127.0.0.1:30004 --out a.json
    python -m evals.accuracy_eval --url http://127.0.0.1:30004 --task gsm8k --limit 200 --label exact --accept 1:0 --output out/

Requirements: Python 3.10+, `tokenizers` for the long-context and h2h long workloads.

## Datasets

Data stays out of git. Scripts read `$SHARAR_RESULTS/datasets/` (default
`/home/yke/sharar-results/datasets/`); pinned files and hashes are in
[datasets.sha256](datasets.sha256). Sources: MGSM `mgsm/mgsm_<lang>.tsv` from google-research/url-nlp@452a21ad,
MBPP `mbpp.jsonl` from google-research/google-research@f82046ba (test split: task ids 11-510; the
reference solution of one task fails its own asserts). Check with `cd $SHARAR_RESULTS/datasets && sha256sum -c /path/to/datasets.sha256`.

## Recording results

Every result must name both versions: this repository's commit and the engine's
(image tag and commit, or the other engine's pinned image digest), plus the model
revision and the serving configuration.
