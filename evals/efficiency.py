"""Efficiency of a serving engine, any engine, through AIPerf: the bench suite's cells chosen by flags, not suite names.

    sharar-eval efficiency aggregated-decode --engine vllm --engine-version V --url http://127.0.0.1:8000 --out DIR
    sharar-eval efficiency aggregated-decode ... --tests code --concurrency 1,4,16,64 --slots 8
    sharar-eval efficiency warm-decode --engine sharar --engine-version V --url http://127.0.0.1:30004 --out DIR \
        [--context 8k,16k] [--categories low_entropy,mixed,high_entropy] [--concurrency 1,2,4,8,16,32,64]
    sharar-eval efficiency report DIR [DIR ...]

aggregated-decode: short-prompt decode at N requests in flight. Each cell is one (test, temperature, N): the test's
fixed prompts (evals/speed/decode.py: chat, prose, code, math, ~35-55 tokens in, capped at 256-1,024 out, natural
stops), each request tagged so nothing is cached, sent by AIPerf as a closed loop: N in flight, each stream sending
its next request when its answer ends, max(8, N x --requests-per-stream) requests in all. Per cell:
- aggregate decode tok/s: all output tokens over first token to last (prefill waves at the start left out);
- per-stream tok/s: AIPerf's per-user output rate, median over requests (decode only: queueing and prefill left out);
- TTFT, errors and short replies.
--slots is how many requests the engine decodes at once (its batch or slot limit, if it has one). Above it the
extra requests queue, and per-stream speed says nothing about what they waited, so the report leaves it out there.
warm-decode: decode only, at a long context, with every request's prompt already in the engine's cache. SPEED-Bench's
throughput split (prompts of a fixed length in three kinds: low entropy = code completion and text sorting, mixed =
few-shot exams and answer selection, high entropy = book and dialogue continuation, writing). Per (context, kind, N): the
cache is flushed, the first N prompts are primed one at a time (1 output token each), then the same N prompts are sent
together, each forced to --output-tokens with ignore_eos. Priming per cell, not once for the largest N: an engine
that keeps fewer prompts cached would otherwise evict the earliest ones, the very prompts the small cells reuse. Per cell:
- window: from the moment the last request starts decoding to the moment the first one ends, all N are decoding and
  nobody prefills;
- aggregate decode tok/s: tokens streamed inside the window over its length (a request's tokens are spread evenly over
  its streamed chunks: exact at one token a chunk, an average where a chunk carries a speculative burst);
- per-stream decode tok/s: each request's (tokens - 1) over its first to last token, median and slowest;
- longest pause: the longest gap between two chunks of one request inside the window (an engine that sets a request
  aside while the others decode shows it here; it already lowers both rates);
- cache hit: prompt tokens read from cache over prompt tokens, lowest over the requests (where the server reports it).
A cell whose requests fail or miss the cache (lowest hit under --min-hit) gets one re-prime and retry; if it fails
again, the row stops there ("stopped": the engine recomputed prompts it could not keep cached at that N, which is a
cache capacity limit, not a failure) and larger N are not run, nor are cells over --max-tokens-in-flight (N x context).
A cell in which fewer than N requests were ever decoding at the same moment stops the row at once: the engine queued
the rest (a batch, slot or memory admission limit), so it does not decode N together.
The engine must already be serving at --url; nothing here starts or stops it. It needs prefix caching, and should report
cached prompt tokens (usage.prompt_tokens_details.cached_tokens): without them the hit column is "-" and nothing
confirms prefill was skipped. A flush endpoint (--engine) is optional.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evals import bench
from evals.speed import decode

TESTS = ("chat", "prose", "code", "math")
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64, 128)


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def decode_spec(args) -> dict:
    """The cells of an aggregated-decode run, in the bench suite's own spec form (bench.cells expands it)."""
    unknown = [t for t in args.tests if t not in TESTS]
    if unknown:
        raise SystemExit(f"unknown test {', '.join(unknown)}; known: {', '.join(TESTS)}")
    return {"workload": "decode", "tests": args.tests, "temperatures": args.temperatures, "concurrency": args.concurrency,
            "requests_per_stream": args.requests_per_stream, "min_requests": 8}


def aggregated_decode(args) -> int:
    spec = decode_spec(args)
    record = {"mode": "aggregated-decode", "spec": spec, "slots": args.slots}
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / "efficiency.json"
    if saved.is_file() and json.loads(saved.read_text()) != record:  # finished cells are skipped by name: keep them the same
        raise SystemExit(f"{saved} was written with other options: use another --out, or the same options to resume")
    saved.write_text(json.dumps(record, indent=1))
    bench.SUITE["suites"]["aggregated-decode"] = [spec]  # one suite for every flag choice; efficiency.json says which
    code = bench.run(argparse.Namespace(**{**vars(args), "suite": "aggregated-decode", "resolve": False}))
    print()
    table(args.out)
    return code


def cells(run: Path) -> list[dict]:
    """The finished cells of a run, each with its test, temperature, N and summary."""
    out = []
    for p in sorted(run.glob("decode/*/c*/cell.json")):
        c = json.loads(p.read_text())
        out.append({"test": c["spec"]["test"], "temperature": c["spec"]["temperature"], "n": c["spec"]["concurrency"],
                    **c["summary"]})
    return sorted(out, key=lambda c: (TESTS.index(c["test"]), c["temperature"], c["n"]))


def table(run: Path) -> None:
    meta = json.loads((run / "run.json").read_text())
    slots = json.loads((run / "efficiency.json").read_text()).get("slots")
    print(f"{run}: {meta['engine']} {meta['engine_version']}" + (f", {slots} slots" if slots else ""))
    print(f"{'test':6s} {'temp':>4s} {'N':>4s} {'reqs':>5s} {'err':>4s} {'aggregate tok/s':>15s} {'per stream tok/s':>16s} "
          f"{'TTFT p50 s':>10s}")
    for c in cells(run):
        queued = slots is not None and c["n"] > slots
        per = "queued" if queued else bench.fmt(c["per_user_tok_s_p50"])
        print(f"{c['test']:6s} {c['temperature']:>4g} {c['n']:>4d} {bench.fmt(c['requests'], '.0f'):>5s} "
              f"{bench.fmt(c['errors'], '.0f'):>4s} {bench.fmt(c.get('decode_tok_s')):>15s} {per:>16s} "
              f"{bench.fmt(c['ttft_s_p50'], '.2f'):>10s}")


CONTEXTS = ("1k", "2k", "8k", "16k", "32k")
CATEGORIES = ("low_entropy", "mixed", "high_entropy")


def tokens(text: str) -> int:
    return int(float(text[:-1]) * 1000) if text.endswith("k") else int(text)


def speedbench_prompts(context: str, category: str, n: int) -> list[list[dict]]:
    """The first n prompts of a kind in AIPerf's resolved SPEED-Bench throughput file for that length, in file order."""
    path = bench.resolved_path(f"speed_bench_throughput_{context}_{category}")
    if not path.is_file():
        raise SystemExit(f"{path.name} is not resolved yet: run `sharar-eval efficiency prepare --context {context}` with no "
                         "engine serving (it loads the SPEED-Bench sources)")
    out = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["category"] == category:
                out.append(row["messages"])
                if len(out) == n:
                    return out
    raise SystemExit(f"{path.name} holds fewer than {n} {category} prompts")


def rows(prompts, output_tokens: int, prime: bool, tag: str) -> list[dict]:
    extra = decode.sampling(0.0)
    if not prime:
        extra = {**extra, "ignore_eos": True}
    return [{"session_id": f"{tag}-{k}", "messages": m, "output_length": 1 if prime else output_tokens, "extra": extra}
            for k, m in enumerate(prompts)]


def run_aiperf(cell: Path, rows_, concurrency: int, ctx: dict) -> Path:
    """One AIPerf pass over rows (bench's trace.yaml); returns its artifacts directory, or raises RuntimeError."""
    import shutil
    shutil.rmtree(cell, ignore_errors=True)
    cell.mkdir(parents=True)
    (cell / "input.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows_))
    env = {"BENCH_URL": ctx["url"], "BENCH_MODEL": ctx["model"], "BENCH_TOKENIZER": ctx["tokenizer"],
           "BENCH_ARTIFACTS": str(cell / "aiperf"), "BENCH_CONCURRENCY": str(concurrency),
           "BENCH_INPUT": str(cell / "input.jsonl"), "BENCH_SESSIONS": str(len(rows_))}
    with (cell / "aiperf.log").open("w") as log:
        code = subprocess.run([*ctx["aiperf"], "profile", "-f", str(bench.CONFIGS / "trace.yaml"), "--ui", "none"], cwd=bench.WORK,
                              stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
    if code or not (cell / "aiperf" / "profile_export.jsonl").is_file():
        tail = [x for x in (cell / "aiperf.log").read_text().splitlines() if x.strip()][-6:]
        raise RuntimeError(f"AIPerf exit {code}: " + " | ".join(tail))
    return cell / "aiperf"


def warm_summary(artifacts: Path, expected: int, want: int) -> dict:
    """Decode-only numbers of one warmed burst from AIPerf's per-request records (times in ms, timestamps in ns)."""
    value = lambda r, k: ((r.get("metrics") or {}).get(k) or {}).get("value")  # noqa: E731
    reqs, errors = [], 0
    for line in (artifacts / "profile_export.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or value(r, "time_to_first_token") is None:
            errors += 1
            continue
        first = r["metadata"]["request_start_ns"] / 1e9 + value(r, "time_to_first_token") / 1e3
        gaps = value(r, "inter_chunk_latency") or []
        times, t = [first], first
        for g in gaps:
            t += g / 1e3
            times.append(t)
        osl = value(r, "output_sequence_length") or 0
        prompt, cached = value(r, "usage_prompt_tokens"), value(r, "usage_prompt_cache_read_tokens")
        reqs.append({"times": times, "osl": osl, "ttft_s": value(r, "time_to_first_token") / 1e3, "prompt": prompt,
                     "hit": cached / prompt if cached is not None and prompt else None})
    out = {"requests": len(reqs), "errors": errors + max(0, expected - len(reqs) - errors),
           "short": sum(r["osl"] < want for r in reqs)}
    if not reqs:
        return out
    start, end = max(r["times"][0] for r in reqs), min(r["times"][-1] for r in reqs)
    # tokens each request streamed inside the window (its tokens spread evenly over its chunks)
    inside = [r["osl"] / len(r["times"]) * sum(start < t <= end for t in r["times"]) for r in reqs]
    rates = [(r["osl"] - 1) / (r["times"][-1] - r["times"][0]) for r in reqs if r["times"][-1] > r["times"][0]]
    # the most requests decoding at the same moment (an end sorts before a start at the same instant)
    edges = sorted([(r["times"][0], 1) for r in reqs] + [(r["times"][-1], -1) for r in reqs])
    level, peak = 0, 0
    for _, step in edges:
        level += step
        peak = max(peak, level)
    pauses = [b - a for r in reqs for a, b in zip(r["times"], r["times"][1:]) if start < b <= end]
    hits = [r["hit"] for r in reqs if r["hit"] is not None]
    out.update(window_s=end - start if end > start else 0.0,
               window_share=(end - start) / statistics.median(r["times"][-1] - r["times"][0] for r in reqs) if end > start else 0.0,
               aggregate_decode_tok_s=sum(inside) / (end - start) if end > start else None,
               per_stream_tok_s_p50=statistics.median(rates) if rates else None,
               per_stream_tok_s_min=min(rates) if rates else None,
               peak_decoding=peak, pause_s_max=max(pauses) if pauses else None,
               output_tokens_min=min(r["osl"] for r in reqs), prompt_tokens_avg=statistics.mean(r["prompt"] or 0 for r in reqs),
               cache_hit_min=min(hits) if hits else None, ttft_s_max=max(r["ttft_s"] for r in reqs),
               chunks_per_request=statistics.mean(len(r["times"]) for r in reqs))
    return out


def warm_decode(args) -> int:
    from evals.speed import model as snapshot_model
    from evals.suite import evals_tree
    url = args.url.rstrip("/")
    if args.engine not in bench.SUITE["engines"]:
        raise SystemExit(f"unknown engine {args.engine!r}; known: {', '.join(bench.SUITE['engines'])}")
    if not bench.alive(url):
        raise SystemExit(f"no server answers at {url}/v1/models: start the engine first")
    model = args.model or bench.http(url + "/v1/models")["data"][0]["id"]
    ctx = {"url": url, "model": model, "tokenizer": str(snapshot_model.snapshot_dir(args.snapshot)),
           "aiperf": bench.aiperf_command(args.aiperf)}
    record = {"mode": "warm-decode", "engine": args.engine, "engine_version": args.engine_version, "model": model,
              "context": args.context, "categories": args.categories, "concurrency": args.concurrency,
              "output_tokens": args.output_tokens, "min_hit": args.min_hit, "max_tokens_in_flight": args.max_tokens_in_flight,
              "aiperf": args.aiperf or bench.SUITE["aiperf"]}
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / "efficiency.json"
    if saved.is_file() and json.loads(saved.read_text()) != record:
        raise SystemExit(f"{saved} was written with other options: use another --out, or the same options to resume")
    saved.write_text(json.dumps(record, indent=1))
    (args.out / "run.json").write_text(json.dumps({**record, "url": url, "evals_tree": evals_tree(),
                                                   "server": bench.describe(url, args.engine),
                                                   "resolved": {c: bench.resolved(f"speed_bench_throughput_{c}_mixed") for c in args.context},
                                                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    if not args.no_checks:
        from evals import checks
        result = checks.run(url, model)
        (args.out / "checks.json").write_text(json.dumps(result, indent=1))
        for c in result["checks"]:
            print(checks.line(c), flush=True)
    bench.WORK.mkdir(parents=True, exist_ok=True)
    print(WARM_HEADER, flush=True)
    unverified = False
    for context in args.context:
        for category in args.categories:
            group = args.out / "warm" / f"{context}-{category}"
            levels = [n for n in args.concurrency
                      if not args.max_tokens_in_flight or n * tokens(context) <= tokens(args.max_tokens_in_flight)]
            todo = [n for n in levels if not (group / f"c{n}" / "cell.json").is_file()]
            if (group / "stopped.json").is_file() or not todo:
                for n in levels:
                    if (group / f"c{n}" / "cell.json").is_file():
                        print(warm_row(context, category, n, json.loads((group / f"c{n}" / "cell.json").read_text())["summary"]),
                              "(done)", flush=True)
                continue
            prompts = speedbench_prompts(context, category, max(levels))
            for n in levels:
                cell = group / f"c{n}"
                if (cell / "cell.json").is_file():
                    continue
                summary, reason = None, None
                for attempt in range(2):
                    try:  # this cell's N prompts and nothing else in the cache
                        bench.flush(url, args.engine)
                        run_aiperf(group / f"prime-c{n}", rows(prompts[:n], 1, True, f"{context}-{category}"), 1, ctx)
                    except Exception as exc:  # noqa: BLE001
                        reason = f"priming failed: {exc}"
                        break
                    try:
                        art = run_aiperf(cell, rows(prompts[:n], args.output_tokens, False, f"{context}-{category}"), n, ctx)
                        summary = warm_summary(art, n, args.output_tokens)
                    except Exception as exc:  # noqa: BLE001
                        reason, summary = f"decode failed: {exc}", None
                        if not bench.alive(url):
                            break
                        continue
                    hit = summary.get("cache_hit_min")
                    if hit is None and not unverified:
                        unverified = True
                        print("the server reports no cached prompt tokens: cache hits are not verified (hit column \"-\")",
                              file=sys.stderr, flush=True)
                    if summary["errors"] or summary["requests"] < n:
                        reason = f"{summary['errors']} of {n} requests failed"
                    elif summary.get("peak_decoding", n) < n:  # an engine limit: a retry would queue the same way
                        reason = f"decodes at most {summary['peak_decoding']} of {n} at once: the engine queued the rest"
                        break
                    elif summary["short"]:
                        reason = f"{summary['short']} of {n} answers came back short of {args.output_tokens} tokens (ignore_eos not honoured?)"
                    elif hit is not None and hit < args.min_hit:
                        reason = f"cache hit {hit:.0%} under {args.min_hit:.0%}: the primed prompts did not stay cached"
                    else:
                        reason = None
                        break
                if reason is None:
                    (cell / "cell.json").write_text(json.dumps({"context": context, "category": category, "n": n,
                                                                 "summary": summary}, indent=1))
                    print(warm_row(context, category, n, summary), flush=True)
                    continue
                (group / "stopped.json").write_text(json.dumps({"n": n, "reason": reason, "summary": summary}, indent=1))
                print(f"{context:>4s} {category:12s} {n:>4d}  stopped: {reason}", flush=True)
                if not bench.alive(url):
                    print(f"the server at {url} stopped answering; the remaining cells are not run", file=sys.stderr)
                    return 2
                break
    return 0


WARM_HEADER = (f"{'ctx':>4s} {'kind':12s} {'N':>4s} {'aggregate':>10s} {'per stream':>10s} {'slowest':>8s} {'window s':>8s} "
               f"{'share':>6s} {'hit min':>7s} {'TTFT max':>8s} {'pause max':>9s} {'prompt':>7s} {'out min':>7s}")


def warm_row(context, category, n, s) -> str:
    f = bench.fmt
    hit = s.get("cache_hit_min")
    return (f"{context:>4s} {category:12s} {n:>4d} {f(s.get('aggregate_decode_tok_s')):>10s} {f(s.get('per_stream_tok_s_p50')):>10s} "
            f"{f(s.get('per_stream_tok_s_min')):>8s} {f(s.get('window_s')):>8s} {f(100 * s.get('window_share', 0), '.0f'):>5s}% "
            f"{'-' if hit is None else f'{100 * hit:.0f}%':>7s} {f(s.get('ttft_s_max'), '.2f'):>8s} "
            f"{f(s.get('pause_s_max'), '.2f'):>9s} {f(s.get('prompt_tokens_avg'), '.0f'):>7s} {f(s.get('output_tokens_min'), '.0f'):>7s}")


def prepare(args) -> int:
    """Resolve SPEED-Bench throughput sets with no engine serving (bench prepare for these lengths)."""
    bench.SUITE["suites"]["warm-decode"] = [{"workload": "speedbench", "isl": args.context, "entropy": ["mixed"],
                                             "concurrency": [1], "requests": 1}]
    return bench.prepare(argparse.Namespace(suite="warm-decode", snapshot=args.snapshot, aiperf=args.aiperf))


def report(args) -> int:
    for run in args.runs:
        mode = json.loads((run / "efficiency.json").read_text())["mode"]
        if mode == "warm-decode":
            meta = json.loads((run / "run.json").read_text())
            print(f"{run}: engine {meta['engine']} ({meta['engine_version']}), warm decode, {meta['output_tokens']} tokens per answer")
            print(WARM_HEADER)
            for group in sorted((g for g in (run / "warm").iterdir() if g.is_dir()),
                                key=lambda g: (tokens(g.name.split("-", 1)[0]), g.name)):
                context, category = group.name.split("-", 1)
                for p in sorted(group.glob("c*/cell.json"), key=lambda p: int(p.parent.name[1:])):
                    print(warm_row(context, category, int(p.parent.name[1:]), json.loads(p.read_text())["summary"]))
                if (group / "stopped.json").is_file():
                    st = json.loads((group / "stopped.json").read_text())
                    print(f"{context:>4s} {category:12s} {st['n']:>4d}  stopped: {st['reason']}")
        else:
            table(run)
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("aggregated-decode", help="short-prompt decode at N requests in flight")
    d.add_argument("--engine", required=True, help="an engine of bench_v1.json (openai for any other server)")
    d.add_argument("--engine-version", required=True, help="what is serving: version, commit, profile; recorded in run.json")
    d.add_argument("--url", required=True, help="the running engine's base URL, e.g. http://127.0.0.1:30004")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--tests", type=lambda s: s.split(","), default=list(TESTS), help="comma-separated: chat,prose,code,math")
    d.add_argument("--concurrency", type=ints, default=list(CONCURRENCY), help="requests in flight, comma-separated")
    d.add_argument("--temperatures", type=lambda s: [float(x) for x in s.split(",")], default=[0.0])
    d.add_argument("--requests-per-stream", type=int, default=4, help="requests per cell = max(8, N x this)")
    d.add_argument("--slots", type=int, help="requests the engine decodes at once; per-stream speed is left out above it")
    d.add_argument("--model", help="served model name (default: the first of /v1/models)")
    d.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    d.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={bench.SUITE['aiperf']})")
    d.add_argument("--no-checks", action="store_true", help="skip the pre-flight checks (sharar-eval check)")
    w = sub.add_parser("warm-decode", help="decode only at a long context, every prompt already cached (SPEED-Bench throughput)")
    w.add_argument("--engine", required=True, help="an engine of bench_v1.json (openai for any other server)")
    w.add_argument("--engine-version", required=True, help="what is serving: version, commit, profile; recorded in run.json")
    w.add_argument("--url", required=True, help="the running engine's base URL, e.g. http://127.0.0.1:30004")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--context", type=lambda s: s.split(","), default=["8k", "16k"], help=f"of {', '.join(CONTEXTS)}")
    w.add_argument("--categories", type=lambda s: s.split(","), default=list(CATEGORIES))
    w.add_argument("--concurrency", type=ints, default=[1, 2, 4, 8, 16, 32, 64],
                   help="requests decoding together, comma-separated (no cap here: the engine's batch size and memory are the limit)")
    w.add_argument("--output-tokens", type=int, default=1024)
    w.add_argument("--min-hit", type=float, default=0.95, help="lowest cache hit a cell accepts (default 0.95)")
    w.add_argument("--max-tokens-in-flight", help="skip cells whose N x context exceeds this, e.g. 256k")
    w.add_argument("--model", help="served model name (default: the first of /v1/models)")
    w.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    w.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={bench.SUITE['aiperf']})")
    w.add_argument("--no-checks", action="store_true", help="skip the pre-flight checks (sharar-eval check)")
    q = sub.add_parser("prepare", help="resolve SPEED-Bench throughput lengths with no engine serving")
    q.add_argument("--context", type=lambda s: s.split(","), default=["8k", "16k"])
    q.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    q.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={bench.SUITE['aiperf']})")
    r = sub.add_parser("report", help="tables of run directories")
    r.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    return {"aggregated-decode": aggregated_decode, "warm-decode": warm_decode, "prepare": prepare,
            "report": report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
