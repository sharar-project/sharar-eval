"""Efficiency of a serving engine, any engine, through AIPerf: the bench suite's cells chosen by flags, not suite names.

    sharar-eval efficiency aggregated-decode --engine vllm --engine-version V --url http://127.0.0.1:8000 --out DIR
    sharar-eval efficiency aggregated-decode ... --tests code --concurrency 1,4,16,64 --slots 8
    sharar-eval efficiency warm-decode --engine sharar --engine-version V --url http://127.0.0.1:30004 --out DIR \
        [--context 8k,16k] [--categories low_entropy,mixed,high_entropy] [--concurrency 1,2,4,8,16,32,64]
    sharar-eval efficiency suite --suite cross-engine --engine E --engine-version V --url URL --out DIR
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
- token counts per chunk: the timed pass keeps AIPerf's raw export (every chunk's arrival time and text), and each
  chunk's tokens are its text's tokens in the pinned tokenizer, scaled so a request's chunks add up to the server's
  count. A server that streams in bursts (a pause, then one chunk carrying many tokens) or holds a stream back to its
  end is counted when the tokens arrive. Cells without the raw export (`report --recompute` of an older run) spread a
  request's tokens evenly over its chunks, and a held-back stream (its final chunk has no latency in AIPerf's records)
  over its span, up to the real end from AIPerf's full_decode_duration;
- cache hit: prompt tokens read from cache over prompt tokens, lowest over the requests (where the server reports it).
A cell whose requests fail or miss the cache (lowest hit under --min-hit) gets one re-prime and retry; if it fails
again, the row stops there ("stopped": the engine recomputed prompts it could not keep cached at that N, which is a
cache capacity limit, not a failure) and larger N are not run, nor are cells over --max-tokens-in-flight (N x context).
A cell in which fewer than N requests were ever decoding at the same moment stops the row at once: the engine queued
the rest (a batch, slot or memory admission limit), so it does not decode N together.
The engine must already be serving at --url; nothing here starts or stops it. It needs prefix caching, and should report
cached prompt tokens (usage.prompt_tokens_details.cached_tokens): without them the hit column is "-" and nothing
confirms prefill was skipped. A flush endpoint (--engine) is optional.
suite: every part of a named suite (workloads/efficiency_v1.json) against one URL, into OUT/<part>.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


def prompt_source(row: dict) -> str:
    """The dataset a SPEED-Bench throughput prompt comes from: low entropy mixes RepoBench Python and Java, JetBrains
    Long Code Arena and AdaLEval text sorting in shares that change with the length; other kinds by sub-category."""
    for key, name in (("repobench_python", "repobench_python"), ("repobench_java", "repobench_java"),
                      ("lca-project-level", "jetbrains_lca"), ("textsort", "textsort")):
        if key in row.get("src_id", ""):
            return name
    return row.get("sub_category") or "?"


def speedbench_prompts(context: str, category: str, n: int, balanced: bool = False, exclude: tuple = (),
                       seed: int = 0, fewer: bool = False) -> list[list[dict]]:
    """n prompts of a kind from AIPerf's resolved SPEED-Bench throughput file for that length. Default: the first n in
    file order. `balanced`: grouped by source dataset (prompt_source), each group shuffled with `seed`, the groups then
    taken in turn (A, B, C, A, ...; a used-up one skipped), so every length gets the same mix of sources whatever the
    file's own shares, from across the whole file. `exclude` drops sources. `fewer`: return what there is when the
    file holds fewer than n (instead of stopping)."""
    path = bench.resolved_path(f"speed_bench_throughput_{context}_{category}")
    if not path.is_file():
        raise SystemExit(f"{path.name} is not resolved yet: run `sharar-eval efficiency prepare --context {context}` with no "
                         "engine serving (it loads the SPEED-Bench sources)")
    groups: dict[str, list] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["category"] == category and prompt_source(row) not in exclude:
                groups.setdefault(prompt_source(row) if balanced else "", []).append(row["messages"])
    queues = []
    for name in sorted(groups):
        if balanced:
            random.Random(f"{seed}-{context}-{category}-{name}").shuffle(groups[name])
        queues.append(groups[name])
    out = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    if len(out) < n and not fewer:
        raise SystemExit(f"{path.name} holds fewer than {n} {category} prompts")
    return out


REASONING = ("off", "low", "medium", "xhigh")  # thinking off, or on at the template's reasoning_effort


def sampling(temperature: float = 1.0, reasoning: str = "xhigh", top_p: float = 0.95) -> dict:
    """A request's sampling and template switches. The default is the checkpoint's own (generation_config.json, and
    TensorFold's): temperature 1.0, top-p 0.95, top-k 20, min-p 0, thinking on at reasoning effort xhigh (the chat
    template's default). Temperature 0 is greedy (no top-p, top-k or min-p sent); reasoning "off" turns thinking off.
    Every request of a run sends the same, so priming and timed requests render the same prompt."""
    extra = {"temperature": float(temperature)}
    if temperature:
        extra.update(top_p=float(top_p), top_k=20, min_p=0.0)
    extra["chat_template_kwargs"] = ({"enable_thinking": True, "reasoning_effort": reasoning} if reasoning != "off"
                                     else {"enable_thinking": False})
    return extra


def add_sampling(p) -> None:
    p.add_argument("--temperature", type=float, default=1.0, help="default 1.0, the checkpoint's own; 0 is greedy")
    p.add_argument("--top-p", type=float, default=0.95, help="with a temperature above 0 (top-k 20, min-p 0 always)")
    p.add_argument("--reasoning", choices=REASONING, default="xhigh",
                   help="thinking on at this reasoning_effort (default xhigh, the template's), or off")


def rows(prompts, output_tokens: int, prime: bool, tag: str, extra: dict | None = None) -> list[dict]:
    extra = sampling() if extra is None else extra
    if not prime:
        extra = {**extra, "ignore_eos": True}
    return [{"session_id": f"{tag}-{k}", "messages": m, "output_length": 1 if prime else output_tokens, "extra": extra}
            for k, m in enumerate(prompts)]


def run_aiperf(cell: Path, rows_, concurrency: int, ctx: dict, raw: bool = False) -> Path:
    """One AIPerf pass over rows (bench's trace.yaml); returns its artifacts directory, or raises RuntimeError. `raw`
    also keeps every streamed chunk with its arrival time and text (profile_export_raw.jsonl)."""
    import shutil
    shutil.rmtree(cell, ignore_errors=True)
    cell.mkdir(parents=True)
    (cell / "input.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows_))
    env = {"BENCH_URL": ctx["url"], "BENCH_MODEL": ctx["model"], "BENCH_TOKENIZER": ctx["tokenizer"],
           "BENCH_ARTIFACTS": str(cell / "aiperf"), "BENCH_CONCURRENCY": str(concurrency),
           "BENCH_INPUT": str(cell / "input.jsonl"), "BENCH_SESSIONS": str(len(rows_)), "BENCH_RAW": str(raw).lower()}
    with (cell / "aiperf.log").open("w") as log:
        code = subprocess.run([*ctx["aiperf"], "profile", "-f", str(bench.CONFIGS / "trace.yaml"), "--ui", "none"], cwd=bench.WORK,
                              stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
    if code or not (cell / "aiperf" / "profile_export.jsonl").is_file():
        tail = [x for x in (cell / "aiperf.log").read_text().splitlines() if x.strip()][-6:]
        raise RuntimeError(f"AIPerf exit {code}: " + " | ".join(tail))
    return cell / "aiperf"


HELD_BACK_S = 1.0  # a timeline that ends this much before the response does is a held-back stream
VISIBLE_SHARE = 0.9  # raw chunks whose text counts under this share of the server's tokens: a held-back stream


def token_counter(snapshot: str):
    """Tokens in each of a list of texts, by the pinned tokenizer (texts counted apart, no special tokens)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
    return lambda texts: [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)] if texts else []


def raw_chunks(artifacts: Path) -> dict[str, list[tuple[float, str]]]:
    """Each request's streamed chunks that carry output (content, reasoning or a tool call), as (arrival time in s,
    text), from AIPerf's raw export; {} without one."""
    path = artifacts / "profile_export_raw.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t0 = r["metadata"]["request_start_ns"] / 1e9 - r["start_perf_ns"] / 1e9
        chunks = []
        for resp in r.get("responses") or []:
            text = ""
            for packet in resp.get("packets") or []:
                if packet.get("name") != "data" or packet.get("value", "").strip() == "[DONE]":
                    continue
                try:
                    delta = ((json.loads(packet["value"]).get("choices") or [{}])[0].get("delta") or {})
                except (json.JSONDecodeError, AttributeError):
                    continue
                text += (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                for call in delta.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    text += (fn.get("name") or "") + (fn.get("arguments") or "")
            if text:
                chunks.append((t0 + resp["perf_ns"] / 1e9, text))
        out[r["metadata"]["x_request_id"]] = chunks
    return out


def warm_summary(artifacts: Path, expected: int, want: int, count=None) -> dict:
    """Decode-only numbers of one warmed burst from AIPerf's per-request records (times in ms, timestamps in ns). With
    AIPerf's raw export and `count` (token_counter), every chunk carries its own token count (its text's tokens, scaled
    so a request's chunks add up to the server's count); without, a request's tokens are spread evenly over its chunks."""
    value = lambda r, k: ((r.get("metrics") or {}).get(k) or {}).get("value")  # noqa: E731
    raw = raw_chunks(artifacts) if count else {}
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
        # a server that holds a stream back and sends the rest in its final chunk leaves that chunk without a latency,
        # so the timeline stops early (vLLM, 8k N=8: 1.4 s of an 86 s request); full_decode_duration (first token to
        # the end of the response) gives the real end, and such a request is counted over its whole span
        full = value(r, "full_decode_duration")
        held = full is not None and first + full / 1e3 - times[-1] > HELD_BACK_S
        if held:
            times.append(first + full / 1e3)
        osl = value(r, "output_sequence_length") or 0
        weights = None if held else [osl / len(times)] * len(times)
        chunks = raw.get(r["metadata"].get("x_request_id"))
        if chunks:  # the chunks as they arrived, each with its own tokens
            times, held = [t for t, _ in chunks], False
            tokens_ = count([text for _, text in chunks])
            weights = [k * osl / sum(tokens_) for k in tokens_] if sum(tokens_) else [osl / len(times)] * len(times)
            # with ignore_eos, the tokens after the model's own end are mostly special tokens a server may not stream
            # (vLLM, Flare), so the visible chunks stop or pause while decoding goes on to the real end: a request
            # whose text holds under VISIBLE_SHARE of the server's count is spread over its whole span (first token to
            # full_decode_duration), as a held-back stream
            if full is not None and sum(tokens_) < VISIBLE_SHARE * osl:
                times, weights, held = [first, first + full / 1e3], None, True
        prompt, cached = value(r, "usage_prompt_tokens"), value(r, "usage_prompt_cache_read_tokens")
        reqs.append({"times": times, "weights": weights, "osl": osl, "ttft_s": value(r, "time_to_first_token") / 1e3,
                     "prompt": prompt, "hit": cached / prompt if cached is not None and prompt else None, "held": held})
    out = {"requests": len(reqs), "errors": errors + max(0, expected - len(reqs) - errors),
           "short": sum(r["osl"] < want for r in reqs)}
    if not reqs:
        return out
    start, end = max(r["times"][0] for r in reqs), min(r["times"][-1] for r in reqs)
    # tokens each request streamed inside the window: its chunks' tokens, or spread over its span if held back
    inside = [r["osl"] * max(0.0, min(end, r["times"][-1]) - max(start, r["times"][0])) / (r["times"][-1] - r["times"][0])
              if r["held"] else sum(w for t, w in zip(r["times"], r["weights"]) if start < t <= end) for r in reqs]
    rates = [(r["osl"] - 1) / (r["times"][-1] - r["times"][0]) for r in reqs if r["times"][-1] > r["times"][0]]
    # the most requests decoding at the same moment (an end sorts before a start at the same instant)
    edges = sorted([(r["times"][0], 1) for r in reqs] + [(r["times"][-1], -1) for r in reqs])
    level, peak = 0, 0
    for _, step in edges:
        level += step
        peak = max(peak, level)
    pauses = [b - a for r in reqs if not r["held"] for a, b in zip(r["times"], r["times"][1:]) if start < b <= end]
    hits = [r["hit"] for r in reqs if r["hit"] is not None]
    out.update(window_s=end - start if end > start else 0.0,
               window_share=(end - start) / statistics.median(r["times"][-1] - r["times"][0] for r in reqs) if end > start else 0.0,
               aggregate_decode_tok_s=sum(inside) / (end - start) if end > start else None,
               per_stream_tok_s_p50=statistics.median(rates) if rates else None,
               per_stream_tok_s_min=min(rates) if rates else None,
               peak_decoding=peak, pause_s_max=max(pauses) if pauses else None, held_back=sum(r["held"] for r in reqs),
               output_tokens_min=min(r["osl"] for r in reqs), prompt_tokens_avg=statistics.mean(r["prompt"] or 0 for r in reqs),
               cache_hit_min=min(hits) if hits else None, ttft_s_max=max(r["ttft_s"] for r in reqs),
               chunks_per_request=statistics.mean(len(r["times"]) for r in reqs),
               chunk_tokens_max=max((max(r["weights"]) for r in reqs if r["weights"]), default=None),
               token_counts="per chunk" if raw else "spread over chunks")
    return out


SPEC_KEYS = ("rounds", "proposed", "accepted", "emitted")


def spec_counters(url: str) -> dict | None:
    """The engine's cumulative speculation counters, where it reports them (Sharar: /get_server_info spec_stats), else
    None: rounds (batched verify steps), proposed and accepted draft tokens, emitted tokens, verify depth."""
    try:
        st = bench.http(url + "/get_server_info", timeout=10).get("spec_stats")
    except Exception:  # noqa: BLE001  an engine without the endpoint
        return None
    if not isinstance(st, dict) or not all(k in st for k in SPEC_KEYS):
        return None
    return {**{k: st[k] for k in SPEC_KEYS}, "depth": st.get("verify_depth")}


def spec_delta(before: dict | None, after: dict | None) -> dict | None:
    """Speculation over one burst from two counter snapshots: per request and verify round, the draft tokens accepted
    (accept_rate = accepted / proposed) and the tokens emitted (accept_length, the bonus token included)."""
    if not before or not after:
        return None
    d = {k: after[k] - before[k] for k in SPEC_KEYS}
    return spec_rates({**d, "depth": after.get("depth")})


def spec_rates(d: dict) -> dict:
    # Each request emits its accepted drafts plus one token of its own per verify round, so emitted - accepted counts
    # request-rounds whatever depth each round verified (an engine may pick the depth per batch size).
    req_rounds = d["emitted"] - d["accepted"]
    return {**d, "accept_rate": d["accepted"] / d["proposed"] if d["proposed"] else None,
            "accept_length": d["emitted"] / req_rounds if req_rounds > 0 else None}


def flush(args, url: str) -> None:
    """Empty the engine's prefix cache before a burst: its flush endpoint, or --flush-cmd for an engine without one
    (a restart), after which the server must answer again within 15 minutes."""
    command = getattr(args, "flush_cmd", None)
    if not command:
        bench.flush(url, args.engine)
        return
    subprocess.run(command, shell=True, check=True)
    deadline = time.time() + 900
    while not bench.alive(url):
        if time.time() > deadline:
            raise RuntimeError(f"no server answers at {url} 15 minutes after --flush-cmd")
        time.sleep(3)


def warm_burst(args, ctx, record, url, group, n, prompts, name, context, category, count):
    """Flush the cache, prime these n prompts, then time them decoding together (up to two tries): (summary, None), or
    (summary or None, the reason the cell stops)."""
    summary, reason = None, None
    for attempt in range(2):
        try:  # these n prompts and nothing else in the cache
            flush(args, url)
            run_aiperf(group / f"prime-{name}", rows(prompts, 1, True, f"{context}-{category}", record["sampling"]), 1, ctx)
        except Exception as exc:  # noqa: BLE001
            return None, f"priming failed: {exc}"
        try:
            before = spec_counters(url)
            art = run_aiperf(group / name, rows(prompts, args.output_tokens, False, f"{context}-{category}", record["sampling"]),
                             n, ctx, raw=True)
            summary = warm_summary(art, n, args.output_tokens, count)
            spec = spec_delta(before, spec_counters(url))
            if spec:
                summary["spec"] = spec
        except Exception as exc:  # noqa: BLE001
            reason, summary = f"decode failed: {exc}", None
            if not bench.alive(url):
                break
            continue
        hit = summary.get("cache_hit_min")
        if summary["errors"] or summary["requests"] < n:
            reason = f"{summary['errors']} of {n} requests failed"
        elif summary.get("peak_decoding", n) < n:  # an engine limit: a retry would queue the same way
            return summary, f"decodes at most {summary['peak_decoding']} of {n} at once: the engine queued the rest"
        elif summary["short"]:
            reason = f"{summary['short']} of {n} answers came back short of {args.output_tokens} tokens (ignore_eos not honoured?)"
        elif hit is not None and hit < args.min_hit:
            reason = f"cache hit {hit:.0%} under {args.min_hit:.0%}: the primed prompts did not stay cached"
        else:
            return summary, None
    return summary, reason


def merge_rounds(parts: list[dict]) -> dict:
    """One cell from several bursts of different prompts (--questions): throughputs and per-stream medians are the mean
    of the bursts, minimums and maximums over all of them, counts summed; `rounds` keeps each burst's own numbers."""
    mean = lambda k: statistics.mean(p[k] for p in parts) if all(p.get(k) is not None for p in parts) else None  # noqa: E731
    low = lambda k: min((p[k] for p in parts if p.get(k) is not None), default=None)  # noqa: E731
    high = lambda k: max((p[k] for p in parts if p.get(k) is not None), default=None)  # noqa: E731
    out = {**parts[0]}
    out.update({k: sum(p[k] for p in parts) for k in ("requests", "errors", "short", "held_back")},
               **{k: mean(k) for k in ("window_s", "window_share", "aggregate_decode_tok_s", "per_stream_tok_s_p50",
                                       "prompt_tokens_avg", "chunks_per_request")},
               **{k: low(k) for k in ("per_stream_tok_s_min", "peak_decoding", "output_tokens_min", "cache_hit_min")},
               **{k: high(k) for k in ("pause_s_max", "ttft_s_max", "chunk_tokens_max")},
               rounds=[{k: p.get(k) for k in ("aggregate_decode_tok_s", "per_stream_tok_s_p50", "cache_hit_min",
                                              "prompt_tokens_avg")} | ({"spec": p["spec"]} if p.get("spec") else {})
                       for p in parts])
    out.pop("spec", None)
    if all(p.get("spec") for p in parts):
        out["spec"] = spec_rates({**{k: sum(p["spec"][k] for p in parts) for k in SPEC_KEYS}, "depth": parts[0]["spec"]["depth"]})
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
    if getattr(args, "min_bursts", 1) > 1 and not getattr(args, "fixed_prompts", False):
        raise SystemExit("--min-bursts needs --fixed-prompts (extra bursts repeat the same prompts)")
    record = {"mode": "warm-decode", "engine": args.engine, "engine_version": args.engine_version, "model": model,
              "context": args.context, "categories": args.categories, "concurrency": args.concurrency,
              "output_tokens": args.output_tokens, "min_hit": args.min_hit, **({"questions": args.questions} if getattr(args, "questions", 1) > 1 else {}),
              **({"flush_cmd": args.flush_cmd} if getattr(args, "flush_cmd", None) else {}),
              **({"balance": True, "exclude_sources": args.exclude_sources or [], "prompt_seed": args.prompt_seed}
                 if getattr(args, "balance", False) else {}), **({"fixed_prompts": True} if getattr(args, "fixed_prompts", False) else {}),
              **({"min_bursts": args.min_bursts} if getattr(args, "min_bursts", 1) > 1 else {}),
              "max_tokens_in_flight": args.max_tokens_in_flight,
              "aiperf": args.aiperf or bench.SUITE["aiperf"]}
    record["sampling"] = sampling(args.temperature, args.reasoning, args.top_p)
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
    count = token_counter(ctx["tokenizer"])
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
            rounds = lambda n: max(1, -(-getattr(args, "questions", 1) // n))  # noqa: E731  bursts of n different prompts per cell
            pick = ({"balanced": True, "exclude": tuple(args.exclude_sources or ()), "seed": args.prompt_seed, "fewer": True}
                    if getattr(args, "balance", False) else {})
            fixed = getattr(args, "fixed_prompts", False)
            prompts = speedbench_prompts(context, category, args.questions if fixed else max(n * rounds(n) for n in levels), **pick)
            if fixed:  # the same prompts at every N: whole passes of ceil(K/n) bursts, at least --min-bursts in all
                rounds = lambda n, k=len(prompts), m=getattr(args, "min_bursts", 1): (  # noqa: E731
                    lambda b: b * -(-m // b))(max(1, -(-k // n)))
                if any(n > len(prompts) for n in levels):
                    print(f"{context} {category}: cells above N {len(prompts)} (the --questions set) are not run", flush=True)
                    levels = [n for n in levels if n <= len(prompts)]
            elif len(prompts) < max(n * rounds(n) for n in levels):  # a small source: as many bursts as its prompts allow
                rounds = lambda n, k=len(prompts), r=rounds: min(r(n), k // n)  # noqa: E731
                print(f"{context} {category}: {len(prompts)} prompts after --exclude-sources; cells above N {len(prompts)} "
                      "are not run, smaller ones use fewer bursts", flush=True)
                levels = [n for n in levels if n <= len(prompts)]
            for n in levels:
                cell = group / f"c{n}"
                if (cell / "cell.json").is_file():
                    continue
                parts, reason = [], None
                for r in range(rounds(n)):  # round 0 in c{n} as before, round r in c{n}-r{r}
                    summary, reason = warm_burst(args, ctx, record, url, group, n, [prompts[(r * n + i) % len(prompts)] for i in range(n)],
                                                 f"c{n}" if r == 0 else f"c{n}-r{r}", context, category, count)
                    if reason is not None:
                        break
                    parts.append(summary)
                    if summary.get("cache_hit_min") is None and not unverified:
                        unverified = True
                        print("the server reports no cached prompt tokens: cache hits are not verified (hit column \"-\")",
                              file=sys.stderr, flush=True)
                if reason is None:
                    summary = parts[0] if len(parts) == 1 else merge_rounds(parts)
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
            f"{f(s.get('pause_s_max'), '.2f'):>9s} {f(s.get('prompt_tokens_avg'), '.0f'):>7s} {f(s.get('output_tokens_min'), '.0f'):>7s}"
            + (f"  ({s['held_back']} held back)" if s.get("held_back") else ""))


def prepare(args) -> int:
    """Resolve SPEED-Bench throughput sets with no engine serving (bench prepare for these lengths)."""
    bench.SUITE["suites"]["warm-decode"] = [{"workload": "speedbench", "isl": args.context, "entropy": ["mixed"],
                                             "concurrency": [1], "requests": 1}]
    return bench.prepare(argparse.Namespace(suite="warm-decode", snapshot=args.snapshot, aiperf=args.aiperf))


# ---------------------------------------------------------------- suite

SUITES = json.loads((bench.PACKAGE / "workloads" / "efficiency_v1.json").read_text())
PARTS = ("warm-decode",)  # in run order
DEFAULTS = {"warm-decode": {"min_hit": 0.95, "max_tokens_in_flight": None, "questions": 1, "balance": False, "exclude_sources": None, "prompt_seed": 0,
                            "fixed_prompts": False, "min_bursts": 1, "flush_cmd": None}}  # what the mode's flags default to


def suite(args) -> int:
    """Every part of a named suite against one engine, into OUT/<part>: one call per engine, the same cells for all."""
    if args.suite not in SUITES["suites"]:
        raise SystemExit(f"unknown suite {args.suite!r}; known: {', '.join(SUITES['suites'])}")
    common = {"engine": args.engine, "engine_version": args.engine_version, "url": args.url, "model": args.model,
              "snapshot": args.snapshot, "aiperf": args.aiperf, "no_checks": args.no_checks}
    worst = 0
    for part in PARTS:
        options = SUITES["suites"][args.suite].get(part)
        if options is None:
            continue
        print(f"\n== {args.suite}: {part}", flush=True)
        flush_cmd = {"flush_cmd": args.flush_cmd} if part == "warm-decode" and getattr(args, "flush_cmd", None) else {}
        ns = argparse.Namespace(**common, **{**DEFAULTS[part], **options, **flush_cmd}, out=args.out / part)  # the suite overrides defaults
        worst = max(worst, warm_decode(ns))
        if not bench.alive(args.url.rstrip("/")):
            print(f"the server at {args.url} stopped answering; the remaining parts are not run", file=sys.stderr)
            return 2
    (args.out / "suite.json").write_text(json.dumps({"suite": args.suite, "version": SUITES["version"],
                                                     "parts": SUITES["suites"][args.suite]}, indent=1))
    return worst


def report(args) -> int:
    count = None
    for run in args.runs:
        if (run / "suite.json").is_file():  # a suite run: its parts
            report(argparse.Namespace(runs=[run / p for p in PARTS if (run / p / "efficiency.json").is_file()],
                                      recompute=getattr(args, "recompute", False)))
            continue
        mode = json.loads((run / "efficiency.json").read_text())["mode"]
        if mode == "warm-decode":
            meta = json.loads((run / "run.json").read_text())
            print(f"{run}: engine {meta['engine']} ({meta['engine_version']}), warm decode, {meta['output_tokens']} tokens per answer")
            print(WARM_HEADER)
            for group in sorted((g for g in (run / "warm").iterdir() if g.is_dir()),
                                key=lambda g: (tokens(g.name.split("-", 1)[0]), g.name)):
                context, category = group.name.split("-", 1)
                for p in sorted(group.glob("c*/cell.json"), key=lambda p: int(p.parent.name[1:])):
                    cell = json.loads(p.read_text())
                    if args.recompute:  # the summary again from the cell's AIPerf records, with this version's maths
                        if count is None and (p.parent / "aiperf" / "profile_export_raw.jsonl").is_file():
                            from evals.speed import model as snapshot_model
                            count = token_counter(str(snapshot_model.snapshot_dir(None)))
                        cell["summary"] = warm_summary(p.parent / "aiperf", cell["n"], meta["output_tokens"], count)
                        p.write_text(json.dumps(cell, indent=1))
                    print(warm_row(context, category, cell["n"], cell["summary"]))
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
    add_sampling(w)
    w.add_argument("--min-hit", type=float, default=0.95, help="lowest cache hit a cell accepts (default 0.95)")
    w.add_argument("--flush-cmd", help="shell command that empties the engine's prefix cache before each burst, for an "
                                       "engine with no flush endpoint (e.g. a restart); the server must answer again after it")
    w.add_argument("--questions", type=int, default=1,
                   help="prompts each cell averages over: N=n runs ceil(questions/n) bursts of n different prompts, "
                        "each flushed and primed on its own, and reports their mean (default 1: one burst, the first n)")
    w.add_argument("--fixed-prompts", action="store_true",
                   help="every N on the same first --questions prompts: N=n runs ceil(questions/n) bursts; N above "
                        "questions is not run (default: N above questions takes new prompts)")
    w.add_argument("--min-bursts", type=int, default=1,
                   help="with --fixed-prompts: at least this many bursts per cell, in whole passes over the prompts, so a "
                        "large N that one burst covers is timed more than once (default 1)")
    w.add_argument("--balance", action="store_true",
                   help="instead of the first prompts in file order: each source dataset shuffled (--prompt-seed), the sources "
                        "taken in turn, so every length gets the same mix (low entropy: repobench_java, repobench_python, "
                        "textsort, jetbrains_lca)")
    w.add_argument("--exclude-sources", type=lambda s: s.split(","), help="with --balance: sources left out, comma-separated")
    w.add_argument("--prompt-seed", type=int, default=0, help="with --balance: the shuffle's seed (default 0)")
    w.add_argument("--max-tokens-in-flight", help="skip cells whose N x context exceeds this, e.g. 256k")
    w.add_argument("--model", help="served model name (default: the first of /v1/models)")
    w.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    w.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={bench.SUITE['aiperf']})")
    w.add_argument("--no-checks", action="store_true", help="skip the pre-flight checks (sharar-eval check)")
    q = sub.add_parser("prepare", help="resolve SPEED-Bench throughput lengths with no engine serving")
    q.add_argument("--context", type=lambda s: s.split(","), default=["8k", "16k"])
    q.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    q.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={bench.SUITE['aiperf']})")
    u = sub.add_parser("suite", help="every part of a named suite (workloads/efficiency_v1.json) against one engine")
    u.add_argument("--suite", required=True, help=f"of {', '.join(SUITES['suites'])}")
    u.add_argument("--engine", required=True, help="an engine of bench_v1.json (openai for any other server)")
    u.add_argument("--engine-version", required=True, help="what is serving: version, commit, profile; recorded in run.json")
    u.add_argument("--url", required=True, help="the running engine's base URL, e.g. http://127.0.0.1:30004")
    u.add_argument("--out", type=Path, required=True)
    u.add_argument("--model", help="served model name (default: the first of /v1/models)")
    u.add_argument("--snapshot", help="Hugging Face cache holding the pinned tokenizer snapshot")
    u.add_argument("--aiperf", help=f"AIPerf executable (default: uvx aiperf=={bench.SUITE['aiperf']})")
    u.add_argument("--no-checks", action="store_true", help="skip the pre-flight checks (sharar-eval check)")
    u.add_argument("--flush-cmd", help="warm-decode's --flush-cmd, for an engine without a flush endpoint")
    r = sub.add_parser("report", help="tables of run directories")
    r.add_argument("runs", type=Path, nargs="+")
    r.add_argument("--recompute", action="store_true",
                   help="warm-decode: rebuild each cell's summary from its AIPerf records with this version (rewrites cell.json)")
    args = parser.parse_args()
    return {"aggregated-decode": aggregated_decode, "warm-decode": warm_decode,
            "suite": suite, "prepare": prepare, "report": report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
