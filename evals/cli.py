"""The sharar-eval command: one entry point for every evaluation and benchmark.

    sharar-eval run --url URL --suite quick --label fast --out runs/fast    # evals.suite run
    sharar-eval compare runs/exact runs/fast                                # evals.suite compare
    sharar-eval datasets fetch                                              # evals.datasets fetch
    sharar-eval speed --url URL --workload all                              # evals.speed
    sharar-eval bench run --engine sharar --engine-version V --url URL --out runs/bench/x   # evals.bench

Each command runs its module as `python -m <module>` would, with the rest of the arguments; `sharar-eval
<command> --help` shows them.
"""
from __future__ import annotations

import runpy
import sys

# command: (module, arguments put first, one line of help)
COMMANDS = {
    "run": ("evals.suite", ["run"], "run a suite or tasks against a server into a run directory"),
    "compare": ("evals.suite", ["compare"], "paired scorecard of two run directories"),
    "show": ("evals.suite", ["show"], "one run directory's scores"),
    "datasets": ("evals.datasets", [], "fetch or check the pinned datasets"),
    "speed": ("evals.speed", [], "AIPerf throughput and latency on the frozen prompts"),
    "sharegpt": ("evals.speed.sharegpt", [], "ShareGPT first turns in length buckets at several concurrencies"),
    "competitive": ("evals.speed.competitive", [], "engine-neutral native /generate cells with frozen token inputs"),
    "engine-matrix": ("evals.speed.engine_matrix", [], "Sharar / SGLang / Atlas token-input matrix"),
    "local": ("evals.speed.local", [], "frozen local-use requests (cache variants, mixed arrival, cancellation)"),
    "agent": ("evals.speed.agent", [], "frozen agent-shaped episodes"),
    "efficiency": ("evals.efficiency", [], "decode speed of any running engine through AIPerf: warm-decode, aggregated-decode"),
    "bench": ("evals.bench", [], "the bench suite through AIPerf, any engine: decode, long context, prefill, SPEED-Bench"),
    "check": ("evals.checks", [], "pre-flight check of the request features the benchmarks rely on"),
    "tbench": ("evals.tbench", [], "Terminal-Bench 2.1 through Harbor (oracle, run)"),
    "accuracy": ("evals.accuracy_eval", [], "GSM8K, MGSM, HumanEval and MBPP on their own"),
    "hard": ("evals.hard_eval", [], "MMLU-Pro, GPQA Diamond, IFEval, IFBench and WMT24++ on their own"),
    "ruler2": ("evals.ruler2", [], "RULER v2 on its own"),
    "long-context": ("evals.long_context", [], "needle retrieval and prefill/decode rates at long context"),
    "h2h": ("evals.h2h", [], "head-to-head serving benchmark"),
    "h2h-report": ("evals.h2h_report", [], "tabulate h2h result files"),
    "identity": ("evals.identity", [], "exact greedy output identity between two servers"),
    "tts": ("evals.tts_eval", [], "test-time scaling on MATH-500"),
    "agent-traffic": ("evals.agent_traffic", [], "record an agent's traffic and check its tool calls"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: sharar-eval <command> [arguments]", "", "commands:"]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, _, help_) in COMMANDS.items()]
    lines += ["", "sharar-eval <command> --help shows a command's arguments."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"sharar-eval: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module, first, _ = COMMANDS[command]
    sys.argv = [f"sharar-eval {command}", *first, *rest]
    try:
        runpy.run_module(module, run_name="__main__")  # sys.argv[0] stays "sharar-eval <command>"
    except SystemExit as exc:  # the module's own exit: a status, or a message (printed, status 1)
        if exc.code is None or isinstance(exc.code, int):
            return exc.code or 0
        print(exc.code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
