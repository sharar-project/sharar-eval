"""The bench suite: the same requests to any engine, sent by AIPerf, so results stay comparable across engines and
over time.

    sharar-eval bench prepare --suite quick                     # with no engine serving: resolve SPEED-Bench sets
    sharar-eval bench run --engine sharar --url http://127.0.0.1:30004 --suite quick --out runs/bench/sharar-fast
    sharar-eval bench report runs/bench/sharar-fast [runs/bench/vllm ...]
    sharar-eval bench list

Fixed and versioned together as bench-v1 (evals/workloads/bench_v1.json):
- the workloads, as AIPerf configs (evals/workloads/bench/*.yaml): SPEED-Bench throughput and qualitative cells
  through AIPerf's public loader, and the fixed-prompt decode, longctx and prefill tests (evals/speed/decode.py),
  written per cell;
- the suites (cells: dataset, concurrency, request or session count) and the AIPerf version, run through uvx so
  nothing is installed;
- per engine, only what is not the chat API: how to flush its prefix cache before every cell, so cells never warm
  each other, and where it describes itself. Any other OpenAI-compatible server runs as `openai`; a new engine with
  a flush endpoint gets an entry.
The server must already be serving and warm. Before the cells, `sharar-eval check` (evals/checks.py) checks the request
features the cells rely on, into checks.json. run.json records the suite version, the engine (`--engine-version`),
the server's self-description and sharar-eval's code hash; a later run into the same directory must match it and
skips finished cells. SPEED-Bench needs a Hugging Face login whose account accepted cais/hle's
terms (AIPerf refetches the rows the published files mask; cais/hle is in every config).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from evals import datasets

PACKAGE = Path(__file__).resolve().parent
SUITE = json.loads((PACKAGE / "workloads" / "bench_v1.json").read_text())
CONFIGS = PACKAGE / "workloads" / "bench"
MATCH = ("suite", "suite_name", "aiperf", "engine", "engine_version", "model")  # run.json keys a rerun keeps


def http(url: str, method: str = "GET", timeout: float = 60):
    request = urllib.request.Request(url, method=method, data=b"{}" if method == "POST" else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def describe(url: str, engine: str):
    path = SUITE["engines"][engine].get("info")
    if not path:
        return None
    try:
        return http(url + path)
    except Exception as exc:  # noqa: BLE001  recorded, not fatal
        return {"error": repr(exc)}


def flush(url: str, engine: str) -> None:
    if path := SUITE["engines"][engine].get("flush"):
        http(url + path, "POST", timeout=600)


FIXED = ("decode", "longctx", "prefill")  # workloads whose requests evals/speed/decode.py writes per cell


def cells(suite: str):
    """(cell name, spec) for every cell of a suite, in run order; an {"include": suite} entry adds that suite's cells.
    Fixed-prompt cells send max(min_requests, requests_per_stream x concurrency) requests."""
    for spec in SUITE["suites"][suite]:
        workload = spec.get("workload")
        if "include" in spec:
            yield from cells(spec["include"])
        elif workload == "speedbench":
            for isl in spec["isl"]:
                for entropy in spec["entropy"]:
                    for c in spec["concurrency"]:
                        yield (f"speedbench/{isl}-{entropy}/c{c}",
                               {**spec, "dataset": f"speed_bench_throughput_{isl}_{entropy}", "concurrency": c})
        elif workload == "speedbench-qual":
            for category in spec["categories"]:
                for c in spec["concurrency"]:
                    yield f"speedbench-qual/{category}/c{c}", {**spec, "dataset": f"speed_bench_{category}", "concurrency": c}
        elif workload in ("decode", "longctx"):
            for test in spec.get("tests", ["longctx"]):
                for t in spec["temperatures"]:
                    for c in spec["concurrency"]:
                        yield (f"{workload}/{test}-t{t:g}/c{c}" if workload == "decode" else f"longctx/t{t:g}/c{c}",
                               {**spec, "test": test, "temperature": t, "concurrency": c,
                                "requests": max(spec["min_requests"], spec["requests_per_stream"] * c)})
        elif workload == "prefill":
            for n in spec["lengths"]:
                yield f"prefill/{n}", {**spec, "test": f"prefill-{n}", "temperature": 0, "concurrency": 1}
        else:
            raise ValueError(f"suite {suite}: unknown workload {workload!r}")


def cell_seed(name: str, salt: str = "") -> int:
    """AIPerf's sampling seed for a SPEED-Bench cell: cells draw different rows, and on a server with no flush endpoint
    each run does too (salted), so no request finds its prompt cached."""
    return int(hashlib.sha256(f"{name}{salt}".encode()).hexdigest()[:8], 16)


WORK = datasets.ROOT / "aiperf"  # AIPerf's working directory: it caches resolved public datasets under it
SPEEDBENCH = ("speedbench", "speedbench-qual")


def resolved_path(dataset: str) -> Path:
    """AIPerf's resolved SPEED-Bench file behind a dataset (one per throughput length, one for the qualitative split)."""
    config = re.match(r"speed_bench_(throughput_\w+?)_(low_entropy|mixed|high_entropy)$", dataset)
    return WORK / ".cache" / "aiperf" / "datasets" / "speed-bench" / f"{config.group(1) if config else 'qualitative'}.jsonl"


def resolved(dataset: str) -> dict:
    """sha256 of AIPerf's resolved SPEED-Bench file behind a dataset: its 14 sources are fetched, not pinned."""
    path = resolved_path(dataset)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else {}


def alive(url: str) -> bool:
    try:
        http(url + "/v1/models", timeout=30)
        return True
    except Exception:  # noqa: BLE001
        return False


def aiperf_command(explicit: str | None) -> list[str]:
    return [explicit] if explicit else ["uvx", "--python", "3.12", "--from", f"aiperf=={SUITE['aiperf']}", "aiperf"]


def prepare(args) -> int:
    """Resolve a suite's SPEED-Bench sets before any engine serves. Resolving downloads and loads the sources (about 40
    GB for the qualitative split), which on a GB10's shared memory can push a serving engine under its memory floor
    (Sharar's guard stops the server). One AIPerf request per missing set, against a local stand-in server."""
    from evals import mock_server
    from evals.speed import model as snapshot_model
    from http.server import ThreadingHTTPServer
    import tempfile
    import threading
    need = {}
    for _, spec in cells(args.suite):
        if spec["workload"] in SPEEDBENCH:
            need.setdefault(resolved_path(spec["dataset"]), spec)
    missing = [(p, s) for p, s in need.items() if not p.is_file()]
    if missing:
        server = ThreadingHTTPServer(("127.0.0.1", 0), mock_server.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        WORK.mkdir(parents=True, exist_ok=True)
        try:
            for path, spec in missing:
                print(f"resolve {path.name} (AIPerf: {spec['dataset']})", flush=True)
                with tempfile.TemporaryDirectory() as tmp:
                    env = {"BENCH_URL": f"http://127.0.0.1:{server.server_address[1]}", "BENCH_MODEL": mock_server.MODEL,
                           "BENCH_TOKENIZER": str(snapshot_model.snapshot_dir(args.snapshot)), "BENCH_DATASET": spec["dataset"],
                           "BENCH_REQUESTS": "1", "BENCH_SESSIONS": "1", "BENCH_CONCURRENCY": "1", "BENCH_SEED": "1",
                           "BENCH_ARTIFACTS": str(Path(tmp) / "aiperf")}
                    config = CONFIGS / ("speedbench.yaml" if spec["workload"] == "speedbench" else "speedbench_qual.yaml")
                    with (Path(tmp) / "aiperf.log").open("w") as log:
                        code = subprocess.run([*aiperf_command(args.aiperf), "profile", "-f", str(config), "--ui", "none"], cwd=WORK,
                                              stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
                    if code or not path.is_file():
                        tail = [line for line in (Path(tmp) / "aiperf.log").read_text().splitlines() if line.strip()][-8:]
                        raise SystemExit(f"AIPerf could not resolve {spec['dataset']} (exit {code}):\n  " + "\n  ".join(tail))
        finally:
            server.shutdown()
    for path in need:
        print(f"{path.name:24s} {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


def summarize(artifacts: Path, prefill: bool = False) -> dict:
    """The cell's headline numbers from AIPerf's exports (seconds, tokens per second). decode_tok_s is all output tokens
    over first token to last (engine-bench's aggregate, which leaves out the first wait; AIPerf's output_tok_s starts
    at the first request sent). Prefill cells add the prompt tokens per second of their second request (the first
    warms the server's shapes)."""
    s = json.loads((artifacts / "profile_export_aiperf.json").read_text())
    get = lambda k, stat="avg": (s.get(k) or {}).get(stat)  # noqa: E731
    ms = lambda k, stat: None if get(k, stat) is None else get(k, stat) / 1000  # noqa: E731
    isl, cached = get("input_sequence_length"), get("usage_prompt_cache_read_tokens")
    out = {"requests": get("request_count"), "errors": sum((e.get("count") or 0) for e in s.get("error_summary") or []),
           "output_tok_s": get("output_token_throughput"), "per_user_tok_s_p50": get("output_token_throughput_per_user", "p50"),
           "ttft_s_p50": ms("time_to_first_token", "p50"), "ttft_s_p99": ms("time_to_first_token", "p99"),
           "itl_s_p50": ms("inter_token_latency", "p50"), "isl_avg": isl, "osl_avg": get("output_sequence_length"),
           "cache_read_share": cached / isl if cached is not None and isl else None,
           "duration_s": get("benchmark_duration")}
    short, records = 0, []
    for line in (artifacts / "profile_export.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else None
        if r and not r.get("error"):
            short += ((r.get("metrics") or {}).get("osl_mismatch_diff_pct") or {}).get("value", 0) < 0
            records.append(r)
    # Replies shorter than requested: a failure where a cell forces exact lengths (ignore_eos), where a server
    # ignoring it would otherwise look faster; expected where replies stop naturally.
    out["short_outputs"] = short
    value = lambda r, k: ((r.get("metrics") or {}).get(k) or {}).get("value")  # noqa: E731
    timed = [r for r in records if value(r, "time_to_first_token") is not None and value(r, "output_sequence_length")]
    if timed:
        first = min(r["metadata"]["request_start_ns"] + value(r, "time_to_first_token") * 1e6 for r in timed)
        last = max(r["metadata"]["request_end_ns"] for r in timed)
        out["decode_tok_s"] = sum(value(r, "output_sequence_length") for r in timed) / ((last - first) / 1e9) if last > first else None
    if prefill and timed:
        r = max(timed, key=lambda r: r["metadata"]["request_start_ns"])
        prompt, ttft = value(r, "usage_prompt_tokens") or value(r, "input_sequence_length"), value(r, "time_to_first_token") / 1000
        out.update(prefill_prompt_tokens=prompt, prefill_ttft_s=ttft, prefill_tok_s=prompt / ttft if prompt and ttft else None,
                   prefill_cached_tokens=value(r, "usage_prompt_cache_read_tokens"))
    return out


def fmt(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def row(name: str, s: dict) -> str:
    share = s.get("cache_read_share")
    return (f"{name:44s} {fmt(s['requests'], '.0f'):>5s} {fmt(s['errors'], '.0f'):>4s} {fmt(s.get('short_outputs'), '.0f'):>5s} "
            f"{fmt(s['output_tok_s']):>9s} {fmt(s.get('decode_tok_s')):>9s} "
            f"{fmt(s['per_user_tok_s_p50']):>8s} {fmt(s['ttft_s_p50'], '.2f'):>8s} {fmt(s['ttft_s_p99'], '.2f'):>8s} "
            f"{'-' if share is None else f'{100 * share:.0f}%':>6s}"
            + (f"  prefill {fmt(s['prefill_tok_s'], '.0f')} tok/s ({fmt(s['prefill_prompt_tokens'], '.0f')} tokens, TTFT "
               f"{fmt(s['prefill_ttft_s'], '.2f')} s)" if "prefill_tok_s" in s else ""))


HEADER = (f"{'cell':44s} {'reqs':>5s} {'err':>4s} {'short':>5s} {'out tok/s':>9s} {'dec tok/s':>9s} {'user p50':>8s} "
          f"{'TTFT p50':>8s} {'TTFT p99':>8s} {'cached':>6s}")


def run(args) -> int:
    if args.engine not in SUITE["engines"]:
        raise SystemExit(f"unknown engine {args.engine!r}; known: {', '.join(SUITE['engines'])} (or add one to bench_v1.json)")
    if args.suite not in SUITE["suites"]:
        raise SystemExit(f"unknown suite {args.suite!r}; known: {', '.join(SUITE['suites'])}")
    from evals.speed import model as snapshot_model
    from evals.suite import evals_tree
    url = args.url.rstrip("/")
    if not alive(url):
        raise SystemExit(f"no server answers at {url}/v1/models: start the engine first")
    model = args.model or http(url + "/v1/models")["data"][0]["id"]
    tokenizer = str(snapshot_model.snapshot_dir(args.snapshot))
    plan = list(cells(args.suite))
    record = {"suite": SUITE["version"], "suite_name": args.suite, "aiperf": args.aiperf or SUITE["aiperf"],
              "engine": args.engine, "engine_version": args.engine_version, "url": url, "model": model,
              "cache_isolation": "flush before each cell" if SUITE["engines"][args.engine].get("flush")
              else "none: unique request tags (fixed-prompt cells) and per-run seeds (SPEED-Bench)",
              "evals_tree": evals_tree(), "server": describe(url, args.engine), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    unresolved = sorted({resolved_path(s["dataset"]).name for _, s in plan
                         if s["workload"] in SPEEDBENCH and not resolved_path(s["dataset"]).is_file()})
    if unresolved and not args.resolve:
        raise SystemExit(f"SPEED-Bench sets not resolved yet ({', '.join(unresolved)}): run `sharar-eval bench prepare "
                         f"--suite {args.suite}` first, with no engine serving, since resolving loads up to about 40 GB of "
                         "sources; or pass --resolve to let the cells resolve them while the engine serves")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_checks:  # recorded, not fatal: a failed check says which comparisons the cells cannot support
        from evals import checks
        try:
            result = checks.run(url, model)
        except checks.CannotRun as exc:
            raise SystemExit(f"sharar-eval check: {exc}") from None
        (args.out / "checks.json").write_text(json.dumps(result, indent=1))
        for c in result["checks"]:
            print(checks.line(c), flush=True)
        if not result["passed"]:
            print(f"some checks did not pass ({args.out / 'checks.json'}); the cells run anyway", file=sys.stderr, flush=True)
        record["checks"] = {c["name"]: c["status"] for c in result["checks"]}
    if (args.out / "run.json").is_file():
        old = json.loads((args.out / "run.json").read_text())
        if (diff := [k for k in MATCH if old.get(k) != record[k]]):
            raise SystemExit(f"{args.out}/run.json differs in {', '.join(diff)}: use another --out")
    else:
        (args.out / "run.json").write_text(json.dumps(record, indent=1))
    aiperf = aiperf_command(args.aiperf)
    WORK.mkdir(parents=True, exist_ok=True)
    print(HEADER, flush=True)
    failed = []
    for name, spec in plan:
        cell = args.out / name
        if (cell / "cell.json").is_file():
            print(row(name, json.loads((cell / "cell.json").read_text())["summary"]), "(done)", flush=True)
            continue
        shutil.rmtree(cell, ignore_errors=True)
        cell.mkdir(parents=True)
        env = {"BENCH_URL": url, "BENCH_MODEL": model, "BENCH_TOKENIZER": tokenizer, "BENCH_ARTIFACTS": str(cell / "aiperf"),
               "BENCH_CONCURRENCY": str(spec["concurrency"])}
        nonce = os.urandom(6).hex()
        if spec["workload"] in ("speedbench", "speedbench-qual"):
            salt = "" if SUITE["engines"][args.engine].get("flush") else nonce
            env.update(BENCH_DATASET=spec["dataset"], BENCH_SEED=str(cell_seed(name, salt)))
            if spec["workload"] == "speedbench":
                config = CONFIGS / "speedbench.yaml"
                env.update(BENCH_REQUESTS=str(spec["requests"]))
            else:
                config = CONFIGS / "speedbench_qual.yaml"
                env.update(BENCH_SESSIONS=str(spec["sessions"]))
        else:  # a fixed-prompt test: fresh request tags per cell and run, so nothing is cached on any engine
            from evals.speed import decode
            config = CONFIGS / "trace.yaml"
            n = decode.write(cell / "input.jsonl", spec["test"], spec.get("requests", 0), spec["temperature"], snapshot=args.snapshot)
            env.update(BENCH_INPUT=str(cell / "input.jsonl"), BENCH_SESSIONS=str(n))
        try:
            flush(url, args.engine)
        except Exception as exc:  # noqa: BLE001  a cell after a failed flush would not start cold
            print(f"{name:44s} not run: flushing the cache failed ({exc!r})"
                  + ("" if alive(url) else f"; the server at {url} is not answering"), file=sys.stderr, flush=True)
            return 2
        before, t0 = describe(url, args.engine), time.time()
        with (cell / "aiperf.log").open("w") as log:
            code = subprocess.run([*aiperf, "profile", "-f", str(config), "--ui", "none"], cwd=WORK, stdout=log,
                                  stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
        result = {"cell": name, "spec": spec, "env": env, "nonce": nonce, "exit": code, "seconds": time.time() - t0,
                  "resolved": resolved(spec["dataset"]) if spec["workload"] in SPEEDBENCH else {},
                  "server_before": before, "server_after": describe(url, args.engine)}
        if code or not (cell / "aiperf" / "profile_export_aiperf.json").is_file():
            (cell / "failed.json").write_text(json.dumps(result, indent=1))
            tail = [line for line in (cell / "aiperf.log").read_text().splitlines() if line.strip()][-6:]
            print(f"{name:44s} FAILED (exit {code}); {cell / 'aiperf.log'}:\n  " + "\n  ".join(tail), flush=True)
            failed.append(name)
            if not alive(url):
                print(f"the server at {url} stopped answering; the remaining cells are not run", file=sys.stderr, flush=True)
                return 2
            continue
        result["summary"] = summarize(cell / "aiperf", prefill=spec["workload"] == "prefill")
        (cell / "cell.json").write_text(json.dumps(result, indent=1))
        print(row(name, result["summary"]), flush=True)
    if failed:
        print(f"{len(failed)} of {len(plan)} cells failed; rerun the same command to retry them", file=sys.stderr)
    return 1 if failed else 0


def report(args) -> int:
    runs = []
    for d in args.runs:
        meta = json.loads((d / "run.json").read_text())
        summaries = {json.loads(p.read_text())["cell"]: json.loads(p.read_text())["summary"] for p in sorted(d.glob("**/cell.json"))}
        runs.append((d, meta, summaries))
        print(f"\n{d}: {meta['engine']} {meta['engine_version']} ({meta['model']}), suite {meta['suite']}/{meta['suite_name']}")
        print(HEADER)
        for name, s in summaries.items():
            print(row(name, s))
    if len(runs) > 1:
        base = runs[0]
        print(f"\nagainst {base[0]}: output tok/s ratios (other / first); failed calls beside them, since a call that fails\n"
              "(e.g. a reply streamed as nothing) counts no tokens and no time")
        for d, meta, summaries in runs[1:]:
            print(f"  {d}:")
            for name, s in summaries.items():
                b = base[2].get(name)
                if not b:
                    continue
                tps = s["output_tok_s"] / b["output_tok_s"] if s["output_tok_s"] and b["output_tok_s"] else None
                errors = f"  errors {fmt(s['errors'], '.0f')} vs {fmt(b['errors'], '.0f')}" if s["errors"] or b["errors"] else ""
                print(f"    {name:44s} tok/s x{fmt(tps, '.2f')}" + errors)
    return 0


def list_(args) -> int:
    print(f"{SUITE['version']} (AIPerf {SUITE['aiperf']})\nengines: {', '.join(SUITE['engines'])}")
    for suite in SUITE["suites"]:
        names = [n for n, _ in cells(suite)]
        print(f"\n{suite}: {len(names)} cells")
        for n in names:
            print(f"  {n}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite's cells against a serving engine")
    r.add_argument("--engine", required=True, help="an engine of bench_v1.json (openai for any other server)")
    r.add_argument("--engine-version", required=True, help="what is serving: version, commit, profile; recorded in run.json")
    r.add_argument("--url", required=True, help="the server's base URL, e.g. http://127.0.0.1:30004")
    r.add_argument("--suite", default="quick")
    r.add_argument("--model", help="served model name (default: the first of /v1/models)")
    r.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    r.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={SUITE['aiperf']})")
    r.add_argument("--no-checks", action="store_true", help="skip the pre-flight checks (sharar-eval check)")
    r.add_argument("--resolve", action="store_true", help="let SPEED-Bench cells resolve missing sets while the engine serves")
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report", help="tables of run directories; with several, ratios against the first")
    p.add_argument("runs", type=Path, nargs="+")
    sub.add_parser("list", help="engines, suites and their cells")
    q = sub.add_parser("prepare", help="resolve a suite's SPEED-Bench sets with no engine serving")
    q.add_argument("--suite", default="quick")
    q.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    q.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={SUITE['aiperf']})")
    args = parser.parse_args()
    return {"run": run, "report": report, "list": list_, "prepare": prepare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
