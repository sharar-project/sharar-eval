"""Head-to-head serving benchmark (#26): the same chat requests against any
OpenAI-compatible server (Sharar, SGLang, vLLM, Atlas, llama.cpp).

Workloads (evals/workloads/h2h_v1.json):
  decode  every prompt per sampling mode and thinking setting, one at a time;
          rate = (completion_tokens - 1) / (last token - first token), reasoning
          tokens included, time to first token reported apart
  long    cold time to first token at each length (a needle, unique text per
          length), then a warm follow-up on the same context
  conc    N streams over a prompt list; aggregate completion tokens / wall time. With --streams N1,N2,...: a
          closed loop per N (each stream sends its next request when the last ends, the 22 prompts in turn by
          category, numbered so none share a prefix); tokens counted in a window after a warm-up, so the server holds
          N requests throughout (no shrinking tail)
  probes  the public SGLang 27B figure's four probes, verbatim (engine-default template)

    sharar-eval h2h --url http://127.0.0.1:30001 --engine sharar-27b --out runs/sharar-27b.jsonl
    sharar-eval h2h --url ... --engine sglang-27b --api-key KEY --workloads decode --modes greedy
    sharar-eval h2h --url ... --engine sharar-fn --workloads conc --streams 8,16,32,64,128 --modes greedy,t07
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from pathlib import Path

SPEC = Path(__file__).parent / "workloads/h2h_v1.json"


def stream(url, body, key=None, timeout=3600):
    """(ttft, t_first→t_last span, completion tokens, text, finish) for one streamed chat request."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers)
    start = time.monotonic()
    first = last = None
    usage, finish, text, chunks = None, None, [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    now = time.monotonic()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
                    chunks += 1
                finish = choice.get("finish_reason") or finish
    n = (usage or {}).get("completion_tokens") or chunks
    return {"t_first": first, "t_last": last,
            "ttft_s": round((first or time.monotonic()) - start, 4), "span_s": round((last or 0) - (first or 0), 4),
            "completion_tokens": n, "tps": round((n - 1) / (last - first), 2) if first and last and last > first and n > 1 else None,
            "finish": finish, "text": "".join(text)}


def body(args, spec, prompt, mode, thinking, max_tokens=None):
    b = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens or spec["max_tokens"],
         **spec["sampling"][mode], "chat_template_kwargs": {"enable_thinking": thinking}}
    if args.seed is not None and mode != "greedy":
        b["seed"] = args.seed
    return {**b, **args.extra}


def run_decode(args, spec, out):
    prompts = [p for p in spec["decode"] if not args.only or any(p["id"].startswith(o) or p["cat"] == o for o in args.only)]
    for thinking in args.thinking:
        for mode in args.modes:
            for rep in range(args.repeats):
                for p in prompts:
                    r = stream(args.url, body(args, spec, p["prompt"], mode, thinking), args.api_key)
                    row = {"engine": args.engine, "workload": "decode", "id": p["id"], "cat": p["cat"], "mode": mode,
                           "thinking": thinking, "rep": rep, **{k: v for k, v in r.items() if k not in ("text", "t_first", "t_last")}, "head": r["text"][:120]}
                    out(row)


def long_prompts(args, spec):
    from evals.long_context import DEFAULT_CORPUS, DEFAULT_TOKENIZER, corpus_ids
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer or DEFAULT_TOKENIZER)
    lengths = [n for n in spec["long"]["lengths"] if not args.max_context or n <= args.max_context]
    ids = corpus_ids(args.corpus or DEFAULT_CORPUS, tok, sum(lengths) + 1000)
    rng, off = random.Random(26), 0
    for n in lengths:
        body_ids = ids[off:off + n - 600]
        off += n
        code = str(rng.randrange(10**7, 10**8))
        cut = len(body_ids) // 2
        text = tok.decode(body_ids[:cut]) + f"\n\n# The special magic number is: {code}.\n\n" + tok.decode(body_ids[cut:])
        yield n, code, text


def run_long(args, spec, out):
    q = spec["long"]["question"]
    for n, code, text in long_prompts(args, spec):
        cold = stream(args.url, body(args, spec, text + q, "greedy", False, spec["long"]["max_tokens"]), args.api_key)
        warm = stream(args.url, {**body(args, spec, text + q, "greedy", False, spec["long"]["max_tokens"]),
                                 "messages": [{"role": "user", "content": text + q}, {"role": "assistant", "content": cold["text"].strip()},
                                              {"role": "user", "content": "Repeat that number, then say how confident you are in one sentence."}]},
                      args.api_key)
        out({"engine": args.engine, "workload": "long", "length": n, "cold_ttft_s": cold["ttft_s"], "correct": code in cold["text"],
             "answer": cold["text"].strip()[:40], "warm_ttft_s": warm["ttft_s"], "prefill_tps": round(n / cold["ttft_s"], 1)})


def interleaved(prompts):
    """The prompts in turn by category (chat, code, math, ...), so any run of consecutive requests mixes them."""
    by_cat = {}
    for p in prompts:
        by_cat.setdefault(p["cat"], []).append(p)
    queues = [by_cat[c] for c in sorted(by_cat)]
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def window_tokens(r, w0, w1):
    """A request's completion tokens inside [w0, w1], spread evenly between its first and last streamed token."""
    first, last, n = r.get("t_first"), r.get("t_last"), r.get("completion_tokens") or 0
    if first is None or last is None or n <= 0:
        return 0.0
    if last <= first:
        return float(n) if w0 <= first <= w1 else 0.0
    overlap = max(0.0, min(last, w1) - max(first, w0))
    return n * overlap / (last - first)


def run_streams(args, spec, out):
    """Closed-loop concurrency sweep: for each N, N streams each send their next request as soon as the last one ends
    (the prompts in turn by category, each opened with a request number so none share a prefix), so the server holds
    N requests. After --warmup-s, tokens are counted for --duration-s (each request's tokens spread over its first to
    last streamed token); streams stop sending at the window's end and the requests in flight finish uncounted past it."""
    prompts = interleaved([p for p in spec["decode"] if not args.only or any(p["id"].startswith(o) or p["cat"] == o for o in args.only)])
    for mode in args.modes:
        for thinking in args.thinking:
            for n in args.streams:
                counter, lock, results = iter(range(10 ** 9)), threading.Lock(), []
                t0 = time.monotonic()
                w0, w1 = t0 + args.warmup_s, t0 + args.warmup_s + args.duration_s

                def worker():
                    while time.monotonic() < w1:
                        with lock:
                            i = next(counter)
                        p = prompts[i % len(prompts)]
                        r = stream(args.url, body(args, spec, f"(Request {i}) " + p["prompt"], mode, thinking), args.api_key)
                        with lock:
                            results.append({**r, "cat": p["cat"]})
                threads = [threading.Thread(target=worker) for _ in range(n)]
                [t.start() for t in threads]
                [t.join() for t in threads]
                toks = sum(window_tokens(r, w0, w1) for r in results)
                inside = [r for r in results if r.get("t_first") and r["t_first"] >= w0 and r["t_last"] <= w1]
                out({"engine": args.engine, "workload": "streams", "streams": n, "mode": mode, "thinking": thinking,
                     "window_s": args.duration_s, "warmup_s": args.warmup_s, "requests": len(results),
                     "tokens": round(toks, 1), "aggregate_tps": round(toks / args.duration_s, 2),
                     "median_stream_tps": statistics.median([r["tps"] for r in results if r["tps"]]) if any(r["tps"] for r in results) else None,
                     "median_ttft_s": statistics.median([r["ttft_s"] for r in results]) if results else None,
                     "requests_inside_window": len(inside)})


def run_conc(args, spec, out):
    if args.streams:
        return run_streams(args, spec, out)
    c = spec["concurrency"]
    by_id = {p["id"]: p for p in spec["decode"]}
    for mode in args.modes:
        for thinking in args.thinking:
            jobs = list(c["prompts"])
            results, lock = [], threading.Lock()

            def worker():
                while True:
                    with lock:
                        if not jobs:
                            return
                        pid = jobs.pop(0)
                    r = stream(args.url, body(args, spec, by_id[pid]["prompt"], mode, thinking), args.api_key)
                    with lock:
                        results.append(r)
            t0 = time.monotonic()
            threads = [threading.Thread(target=worker) for _ in range(c["streams"])]
            [t.start() for t in threads]
            [t.join() for t in threads]
            wall = time.monotonic() - t0
            toks = sum(r["completion_tokens"] for r in results)
            out({"engine": args.engine, "workload": "conc", "streams": c["streams"], "mode": mode, "thinking": thinking,
                 "requests": len(results), "tokens": toks, "wall_s": round(wall, 2), "aggregate_tps": round(toks / wall, 2),
                 "median_stream_tps": statistics.median([r["tps"] for r in results if r["tps"]])})


def run_probes(args, spec, out):
    """The public 27B figure's own probes: the engine's default template, no thinking override."""
    for rep in range(2):
        for p in spec["probes"]["items"]:
            b = {"model": args.model, "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": p["max_tokens"],
                 "temperature": p["temperature"], **args.extra}
            r = stream(args.url, b, args.api_key)
            out({"engine": args.engine, "workload": "probes", "id": p["id"], "rep": rep, "median": p["median"],
                 **{k: v for k, v in r.items() if k not in ("text", "t_first", "t_last")}, "head": r["text"][:120]})
    tps = [r["tps"] for r in _rows if r.get("workload") == "probes" and r.get("median") and r.get("tps")]
    if tps:
        print(f"probes greedy median {statistics.median(tps):.1f} tok/s over {len(tps)} runs", flush=True)


_rows = []


def summarize(rows):
    dec = [r for r in rows if r["workload"] == "decode" and r["tps"]]
    groups = {}
    for r in dec:
        groups.setdefault((r["mode"], r["thinking"]), []).append(r)
        groups.setdefault((r["mode"], r["thinking"], r["cat"]), []).append(r)
    for key in sorted(groups, key=str):
        rs = groups[key]
        tps = [r["tps"] for r in rs]
        print(f"{'/'.join(map(str, key)):32s} n={len(rs):3d} median {statistics.median(tps):6.1f} tok/s  "
              f"p10 {sorted(tps)[len(tps) // 10]:6.1f}  ttft {statistics.median(r['ttft_s'] for r in rs):.3f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", required=True, help="label for the rows")
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workloads", default="decode,long,conc")
    ap.add_argument("--modes", default="greedy,t07,t10")
    ap.add_argument("--thinking", default="off", help="off, on or off,on")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of prompt id prefixes or categories")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body (engine-specific knobs)")
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--streams", default=None, help="conc as a closed-loop sweep over these stream counts (e.g. 4,8,16,32,64,128); default: the workload's 4 streams over its 8 prompts")
    ap.add_argument("--warmup-s", type=float, default=30.0, help="--streams: seconds before the counted window")
    ap.add_argument("--duration-s", type=float, default=60.0, help="--streams: the counted window, seconds")
    args = ap.parse_args()
    args.url = args.url.rstrip("/")
    args.modes = args.modes.split(",")
    args.thinking = [t == "on" for t in args.thinking.split(",")]
    args.only = args.only.split(",") if args.only else None
    args.streams = [int(v) for v in args.streams.split(",")] if args.streams else None
    args.extra = json.loads(args.extra)
    spec = json.loads(SPEC.read_text())
    rows = _rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        def out(row):
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "head"}, ensure_ascii=False)[:300], flush=True)
        for w in args.workloads.split(","):
            {"decode": run_decode, "long": run_long, "conc": run_conc, "probes": run_probes}[w](args, spec, out)
    summarize(rows)


if __name__ == "__main__":
    main()
