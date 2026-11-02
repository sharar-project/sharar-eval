import argparse
import json

import pytest

from evals import bench, efficiency


def record(start_s, ttft_ms, gaps_ms, osl, prompt=1000, cached=990, cid="s", full_ms=None):
    full_ms = sum(gaps_ms) if full_ms is None else full_ms
    end = start_s + (ttft_ms + full_ms) / 1e3
    return {"metadata": {"conversation_id": cid, "request_start_ns": int(start_s * 1e9), "request_end_ns": int(end * 1e9)},
            "metrics": {"time_to_first_token": {"value": ttft_ms}, "inter_chunk_latency": {"value": gaps_ms},
                        "full_decode_duration": {"value": full_ms},
                        "output_sequence_length": {"value": osl}, "usage_prompt_tokens": {"value": prompt},
                        "usage_prompt_cache_read_tokens": {"value": cached}}}


def write(tmp_path, records):
    (tmp_path / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return tmp_path


def test_warm_summary_times_only_the_window_where_every_request_decodes(tmp_path):
    # a: first token at 1.0 s, then a chunk every 100 ms to 1.9 s; b: first token at 1.55 s, last at 2.45 s.
    # Both decode in (1.55, 1.9]: a's chunks at 1.6-1.9 and b's at 1.65-1.85, one token each, over 0.35 s.
    art = write(tmp_path, [record(0.0, 1000, [100] * 9, 10, cid="a"), record(0.0, 1550, [100] * 9, 10, cid="b")])
    s = efficiency.warm_summary(art, expected=2, want=10)
    assert s["requests"] == 2 and s["errors"] == 0 and s["short"] == 0
    assert s["window_s"] == pytest.approx(0.35)
    assert s["aggregate_decode_tok_s"] == pytest.approx(7 / 0.35)
    assert s["per_stream_tok_s_p50"] == pytest.approx(9 / 0.9)
    assert s["window_share"] == pytest.approx(0.35 / 0.9)
    assert s["cache_hit_min"] == pytest.approx(0.99) and s["ttft_s_max"] == pytest.approx(1.55)
    assert s["peak_decoding"] == 2 and s["pause_s_max"] == pytest.approx(0.1)


def test_warm_summary_takes_a_held_back_streams_end_from_its_full_decode(tmp_path):
    # b's chunks stop at 1.2 s with no latency for the final chunk, which ends the response at 3.05 s: without the real
    # end, b would look finished before a and the cell queued
    art = write(tmp_path, [record(0.0, 1000, [100] * 20, 21, cid="a"), record(0.0, 1000, [100, 100], 20, cid="b", full_ms=2050)])
    s = efficiency.warm_summary(art, expected=2, want=20)
    assert s["held_back"] == 1 and s["peak_decoding"] == 2
    assert s["window_s"] == pytest.approx(2.0)  # 1.0 s to a's last chunk at 3.0 s
    # a: 20 chunks of one token in (1.0, 3.0]; b: 20 tokens spread over its 2.05 s span, 2.0 s of it inside
    assert s["aggregate_decode_tok_s"] == pytest.approx((20 + 20 * 2.0 / 2.05) / 2.0)
    assert s["per_stream_tok_s_min"] == pytest.approx(19 / 2.05) and s["pause_s_max"] == pytest.approx(0.1)


def raw_record(rid, start_s, chunks):
    """AIPerf's raw export of one request: chunks as (arrival s, text)."""
    resp = [{"perf_ns": int((start_s + t) * 1e9), "packets": [{"name": "data", "value": json.dumps(
        {"choices": [{"index": 0, "delta": {"content": text}}]})}]} for t, text in chunks]
    resp.append({"perf_ns": resp[-1]["perf_ns"], "packets": [{"name": "data", "value": "[DONE]"}]})
    return {"metadata": {"x_request_id": rid, "request_start_ns": int(start_s * 1e9)}, "start_perf_ns": int(start_s * 1e9),
            "responses": resp}


def test_warm_summary_counts_each_raw_chunks_own_tokens(tmp_path):
    # a: one word a chunk, 1.0-3.0 s; b: one word at 1.0 s, a pause, then 19 words in one chunk at 2.0 s. Spread evenly
    # over b's chunks, the burst would count 10 tokens; counted by its text it carries 19 of b's 20.
    a = record(0.0, 1000, [100] * 20, 21, cid="a")
    b = record(0.0, 1000, [1000, 1050], 20, cid="b")
    a["metadata"]["x_request_id"], b["metadata"]["x_request_id"] = "ra", "rb"
    art = write(tmp_path, [a, b])
    (art / "profile_export_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        raw_record("ra", 0.0, [(1.0 + 0.1 * k, f"w{k} ") for k in range(21)]),
        raw_record("rb", 0.0, [(1.0, "w0 "), (2.0, " ".join(f"w{k}" for k in range(1, 20)) + " "), (3.05, "")]))))
    count = lambda texts: [len(t.split()) for t in texts]  # noqa: E731
    s = efficiency.warm_summary(art, expected=2, want=20, count=count)
    assert s["token_counts"] == "per chunk" and s["chunk_tokens_max"] == pytest.approx(19)
    # window (1.0, 2.0]: b's last chunk with text is at 2.0 s; a has 10 chunks in it, b its 19-token burst
    assert s["window_s"] == pytest.approx(1.0) and s["aggregate_decode_tok_s"] == pytest.approx(10 + 19)
    s = efficiency.warm_summary(art, expected=2, want=20)  # no counter: tokens spread evenly over chunks
    assert s["token_counts"] == "spread over chunks"


def test_warm_summary_spreads_a_stream_whose_text_stops_early_over_its_whole_span(tmp_path):
    # b's text ends after 5 of its 20 tokens at 1.4 s (the model's own end; the server goes on to 20 without streaming
    # the rest) and its response ends at 3.05 s. Counted by its chunks, b would carry 5 tokens and stop decoding at
    # 1.4 s; spread over its whole span it decodes with a to the end.
    a = record(0.0, 1000, [100] * 20, 21, cid="a")
    b = record(0.0, 1000, [100] * 4, 20, cid="b", full_ms=2050)
    a["metadata"]["x_request_id"], b["metadata"]["x_request_id"] = "ra", "rb"
    art = write(tmp_path, [a, b])
    (art / "profile_export_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        raw_record("ra", 0.0, [(1.0 + 0.1 * k, f"w{k} ") for k in range(21)]),
        raw_record("rb", 0.0, [(1.0 + 0.1 * k, f"w{k} ") for k in range(5)] + [(3.05, "")]))))
    count = lambda texts: [len(t.split()) for t in texts]  # noqa: E731
    s = efficiency.warm_summary(art, expected=2, want=20, count=count)
    assert s["held_back"] == 1 and s["peak_decoding"] == 2
    assert s["window_s"] == pytest.approx(2.0)  # 1.0 s to a's last chunk at 3.0 s
    assert s["aggregate_decode_tok_s"] == pytest.approx((20 + 20 * 2.0 / 2.05) / 2.0)


def test_warm_summary_sees_queued_requests_and_pauses(tmp_path):
    # b starts decoding only after a has finished: never 2 at once, no window
    art = write(tmp_path, [record(0.0, 100, [100] * 4, 5, cid="a"), record(0.0, 600, [100] * 4, 5, cid="b")])
    s = efficiency.warm_summary(art, expected=2, want=5)
    assert s["peak_decoding"] == 1 and s["window_s"] == 0.0 and s["aggregate_decode_tok_s"] is None
    # three decode together; c is set aside for 0.8 s inside the window
    art = write(tmp_path, [record(0.0, 100, [100] * 20, 21, cid="a"), record(0.0, 100, [100] * 20, 21, cid="b"),
                           record(0.0, 100, [100, 800] + [100] * 10, 13, cid="c")])
    s = efficiency.warm_summary(art, expected=3, want=13)
    assert s["peak_decoding"] == 3 and s["pause_s_max"] == pytest.approx(0.8)


def test_warm_summary_spreads_a_speculative_burst_over_its_chunks(tmp_path):
    # 3 chunks carrying 12 tokens: 4 tokens a chunk is counted inside the window
    art = write(tmp_path, [record(0.0, 1000, [100, 100], 12, cid="a"), record(0.0, 1000, [100, 100], 12, cid="b")])
    s = efficiency.warm_summary(art, expected=2, want=12)
    assert s["window_s"] == pytest.approx(0.2)
    assert s["aggregate_decode_tok_s"] == pytest.approx(2 * 2 * 4 / 0.2)


def test_warm_summary_counts_missing_requests_and_short_answers(tmp_path):
    art = write(tmp_path, [record(0.0, 1000, [100] * 4, 5, cid="a")])
    s = efficiency.warm_summary(art, expected=2, want=10)
    assert s["requests"] == 1 and s["errors"] == 1 and s["short"] == 1


def test_rows_prime_with_one_token_and_force_the_answer_length():
    prompts = [[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]]
    prime = efficiency.rows(prompts, 1024, True, "8k-low")
    timed = efficiency.rows(prompts, 1024, False, "8k-low")
    assert [r["output_length"] for r in prime] == [1, 1] and "ignore_eos" not in prime[0]["extra"]
    assert [r["output_length"] for r in timed] == [1024, 1024] and timed[0]["extra"]["ignore_eos"] is True
    assert [r["messages"] for r in prime] == [r["messages"] for r in timed] == prompts  # the same bytes both times
    assert timed[0]["extra"]["temperature"] == 1.0  # the default: the checkpoint's own sampling, thinking on
    assert timed[0]["extra"]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}


def test_rows_send_the_aligned_sampling_with_thinking_on_both_when_priming_and_timed():
    prompts = [[{"role": "user", "content": "x"}]]
    extra = efficiency.sampling(1.0, "xhigh")
    prime, timed = efficiency.rows(prompts, 1, True, "8k", extra), efficiency.rows(prompts, 1024, False, "8k", extra)
    assert timed[0]["extra"] == {**prime[0]["extra"], "ignore_eos": True}
    assert prime[0]["extra"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                 "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
    assert efficiency.sampling(0.0, "off") == {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    assert efficiency.sampling(0.7, "off", 0.8)["top_p"] == 0.8


def test_speedbench_prompts_take_one_kind_in_file_order(tmp_path, monkeypatch):
    path = tmp_path / "throughput_8k.jsonl"
    path.write_text("".join(json.dumps({"category": c, "messages": [{"role": "user", "content": f"{c}{i}"}]}) + "\n"
                            for i, c in enumerate(["mixed", "low_entropy", "low_entropy", "high_entropy", "low_entropy"])))
    monkeypatch.setattr(bench, "resolved_path", lambda dataset: path)
    got = efficiency.speedbench_prompts("8k", "low_entropy", 2)
    assert [m[0]["content"] for m in got] == ["low_entropy1", "low_entropy2"]
    with pytest.raises(SystemExit, match="fewer than 4"):
        efficiency.speedbench_prompts("8k", "low_entropy", 4)


def test_report_lists_contexts_by_length_and_stopped_rows(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "warm").mkdir(parents=True)
    (run / "efficiency.json").write_text(json.dumps({"mode": "warm-decode"}))
    (run / "run.json").write_text(json.dumps({"engine": "openai", "engine_version": "x", "output_tokens": 1024}))
    summary = {"aggregate_decode_tok_s": 100.0, "per_stream_tok_s_p50": 100.0, "per_stream_tok_s_min": 100.0,
               "window_s": 1.0, "window_share": 1.0, "cache_hit_min": 1.0, "ttft_s_max": 0.1, "prompt_tokens_avg": 8000,
               "output_tokens_min": 1024}
    for context in ("16k", "8k"):
        cell = run / "warm" / f"{context}-low_entropy" / "c1"
        cell.mkdir(parents=True)
        (cell / "cell.json").write_text(json.dumps({"n": 1, "summary": summary}))
    (run / "warm" / "16k-low_entropy" / "stopped.json").write_text(json.dumps({"n": 2, "reason": "cache hit 0%"}))
    efficiency.report(argparse.Namespace(runs=[run], recompute=False))
    lines = [line for line in capsys.readouterr().out.splitlines() if "low_entropy" in line]
    assert [line.split()[0] for line in lines] == ["8k", "16k", "16k"]
    assert "stopped: cache hit 0%" in lines[-1]


def test_warm_decode_stops_a_row_where_the_engine_queues_without_a_retry(tmp_path, monkeypatch, capsys):
    from evals import suite
    from evals.speed import model
    monkeypatch.setattr(model, "snapshot_dir", lambda snapshot=None: tmp_path)
    monkeypatch.setattr(suite, "evals_tree", lambda: "test")
    monkeypatch.setattr(bench, "alive", lambda url: True)
    monkeypatch.setattr(bench, "describe", lambda url, engine: {})
    monkeypatch.setattr(bench, "resolved", lambda name: None)
    monkeypatch.setattr(bench, "aiperf_command", lambda a: ["aiperf"])
    monkeypatch.setattr(bench, "flush", lambda url, engine: None)
    monkeypatch.setattr(bench, "WORK", tmp_path / "work")
    monkeypatch.setattr(efficiency, "speedbench_prompts", lambda c, k, n: [[{"role": "user", "content": str(i)}] for i in range(n)])
    timed = []

    def run_aiperf(cell, rows_, n, ctx, raw=False):
        cell.mkdir(parents=True, exist_ok=True)
        if not cell.name.startswith("prime"):
            timed.append(n)
        return cell
    monkeypatch.setattr(efficiency, "run_aiperf", run_aiperf)
    monkeypatch.setattr(efficiency, "token_counter", lambda snapshot: None)
    good = {"requests": 0, "errors": 0, "short": 0, "cache_hit_min": 1.0, "aggregate_decode_tok_s": 1.0, "window_share": 1.0}
    # the engine decodes at most 4 at once
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want, count: {**good, "requests": n, "peak_decoding": min(n, 4)})
    args = argparse.Namespace(engine="openai", engine_version="x", url="http://x", model="m", snapshot=None, aiperf=None,
                              out=tmp_path / "run", context=["8k"], categories=["low_entropy"], concurrency=[1, 4, 8, 16],
                              output_tokens=16, temperature=0.0, top_p=0.95, reasoning="off", min_hit=0.95,
                              max_tokens_in_flight=None, no_checks=True)
    assert efficiency.warm_decode(args) == 0
    assert timed == [1, 4, 8]  # N = 8 ran once, not retried; N = 16 not run
    stopped = json.loads((args.out / "warm" / "8k-low_entropy" / "stopped.json").read_text())
    assert stopped["n"] == 8 and stopped["reason"].startswith("decodes at most 4 of 8 at once")

    # a queued request often loses its cached prompt while it waits: still the queue, not retried, and said
    timed.clear()
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want, count: {
        **good, "requests": n, "peak_decoding": min(n, 7), "cache_hit_min": 1.0 if n < 8 else 0.0})
    args.out = tmp_path / "lost"
    assert efficiency.warm_decode(args) == 0
    assert timed == [1, 4, 8]
    stopped = json.loads((args.out / "warm" / "8k-low_entropy" / "stopped.json").read_text())
    assert stopped["reason"] == ("decodes at most 7 of 8 at once: the engine queued the rest; the queued requests lost "
                                 "their cached prompts while waiting (lowest hit 0%)")


def test_warm_decode_questions_average_bursts_of_different_prompts(tmp_path, monkeypatch):
    from evals import suite
    from evals.speed import model
    monkeypatch.setattr(model, "snapshot_dir", lambda snapshot=None: tmp_path)
    monkeypatch.setattr(suite, "evals_tree", lambda: "test")
    monkeypatch.setattr(bench, "alive", lambda url: True)
    monkeypatch.setattr(bench, "describe", lambda url, engine: {})
    monkeypatch.setattr(bench, "resolved", lambda name: None)
    monkeypatch.setattr(bench, "aiperf_command", lambda a: ["aiperf"])
    monkeypatch.setattr(bench, "flush", lambda url, engine: None)
    monkeypatch.setattr(bench, "WORK", tmp_path / "work")
    monkeypatch.setattr(efficiency, "speedbench_prompts", lambda c, k, n: [[{"role": "user", "content": str(i)}] for i in range(n)])
    timed = []

    def run_aiperf(cell, rows_, n, ctx, raw=False):
        cell.mkdir(parents=True, exist_ok=True)
        if not cell.name.startswith("prime"):
            timed.append((cell.name, [r["messages"][0]["content"] for r in rows_]))
        return cell
    monkeypatch.setattr(efficiency, "run_aiperf", run_aiperf)
    monkeypatch.setattr(efficiency, "rows", lambda ps, *a: [{"messages": p} for p in ps])
    monkeypatch.setattr(efficiency, "token_counter", lambda snapshot: None)
    speed = {"c1": 10.0, "c1-r1": 20.0, "c1-r2": 30.0, "c2": 40.0, "c2-r1": 60.0, "c4": 100.0}
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want, count: {
        "requests": n, "errors": 0, "short": 0, "held_back": 0, "cache_hit_min": 1.0, "peak_decoding": n,
        "aggregate_decode_tok_s": speed[art.name], "per_stream_tok_s_p50": speed[art.name] / n, "window_share": 1.0})
    args = argparse.Namespace(engine="openai", engine_version="x", url="http://x", model="m", snapshot=None, aiperf=None,
                              out=tmp_path / "run", context=["8k"], categories=["low_entropy"], concurrency=[1, 2, 4],
                              output_tokens=16, temperature=0.0, top_p=0.95, reasoning="off", min_hit=0.95, questions=3,
                              max_tokens_in_flight=None, no_checks=True)
    assert efficiency.warm_decode(args) == 0
    # N = 1: 3 bursts of 1, N = 2: 2 bursts of 2 (4 prompts), N = 4: one burst; each burst on its own prompts
    assert timed == [("c1", ["0"]), ("c1-r1", ["1"]), ("c1-r2", ["2"]), ("c2", ["0", "1"]), ("c2-r1", ["2", "3"]),
                     ("c4", ["0", "1", "2", "3"])]
    cell = lambda n: json.loads((args.out / "warm" / "8k-low_entropy" / f"c{n}" / "cell.json").read_text())["summary"]  # noqa: E731
    assert cell(1)["aggregate_decode_tok_s"] == 20.0 and cell(1)["requests"] == 3 and len(cell(1)["rounds"]) == 3
    assert cell(2)["aggregate_decode_tok_s"] == 50.0 and "rounds" not in cell(4)


def test_warm_decode_fixed_prompts_are_the_same_set_at_every_n(tmp_path, monkeypatch):
    from evals import suite
    from evals.speed import model
    monkeypatch.setattr(model, "snapshot_dir", lambda snapshot=None: tmp_path)
    monkeypatch.setattr(suite, "evals_tree", lambda: "test")
    monkeypatch.setattr(bench, "alive", lambda url: True)
    monkeypatch.setattr(bench, "describe", lambda url, engine: {})
    monkeypatch.setattr(bench, "resolved", lambda name: None)
    monkeypatch.setattr(bench, "aiperf_command", lambda a: ["aiperf"])
    monkeypatch.setattr(bench, "flush", lambda url, engine: None)
    monkeypatch.setattr(bench, "WORK", tmp_path / "work")
    asked = []

    def prompts(c, k, n):
        asked.append(n)
        return [[{"role": "user", "content": str(i)}] for i in range(n)]
    monkeypatch.setattr(efficiency, "speedbench_prompts", prompts)
    timed = []

    def run_aiperf(cell, rows_, n, ctx, raw=False):
        cell.mkdir(parents=True, exist_ok=True)
        if not cell.name.startswith("prime"):
            timed.append((cell.name, [r["messages"][0]["content"] for r in rows_]))
        return cell
    monkeypatch.setattr(efficiency, "run_aiperf", run_aiperf)
    monkeypatch.setattr(efficiency, "rows", lambda ps, *a: [{"messages": p} for p in ps])
    monkeypatch.setattr(efficiency, "token_counter", lambda snapshot: None)
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want, count: {
        "requests": n, "errors": 0, "short": 0, "held_back": 0, "cache_hit_min": 1.0, "peak_decoding": n,
        "aggregate_decode_tok_s": 10.0, "per_stream_tok_s_p50": 10.0, "window_share": 1.0})
    args = argparse.Namespace(engine="openai", engine_version="x", url="http://x", model="m", snapshot=None, aiperf=None,
                              out=tmp_path / "run", context=["8k"], categories=["low_entropy"], concurrency=[1, 2, 4, 8],
                              output_tokens=16, temperature=0.0, top_p=0.95, reasoning="off", min_hit=0.95, questions=4,
                              fixed_prompts=True, min_bursts=3, max_tokens_in_flight=None, no_checks=True)
    assert efficiency.warm_decode(args) == 0
    # prompts 0-3 at every N, at least 3 bursts in whole passes (N 2: two passes, N 4: three); N 8 is not run
    p = ["0", "1", "2", "3"]
    assert asked == [4] and timed == [("c1", ["0"]), ("c1-r1", ["1"]), ("c1-r2", ["2"]), ("c1-r3", ["3"]),
                                      ("c2", p[:2]), ("c2-r1", p[2:]), ("c2-r2", p[:2]), ("c2-r3", p[2:]),
                                      ("c4", p), ("c4-r1", p), ("c4-r2", p)]
    saved = json.loads((args.out / "efficiency.json").read_text())
    assert saved["fixed_prompts"] is True and saved["min_bursts"] == 3
    args.min_bursts, args.fixed_prompts, args.out = 3, False, tmp_path / "other"
    with pytest.raises(SystemExit, match="--min-bursts needs --fixed-prompts"):
        efficiency.warm_decode(args)


def test_balanced_prompts_take_shuffled_sources_in_turn(tmp_path, monkeypatch):
    path = tmp_path / "throughput_8k.jsonl"
    srcs = ["tianyang/repobench_python_v1.1-in_file_1"] * 6 + ["textsort_1"] * 3 + ["JetBrains-Research/lca-project-level-x"] * 2
    path.write_text("".join(json.dumps({"category": "low_entropy", "sub_category": "x", "src_id": s,
                                        "messages": [{"role": "user", "content": f"{s}#{i}"}]}) + "\n"
                            for i, s in enumerate(srcs)))
    monkeypatch.setattr(bench, "resolved_path", lambda name: path)
    first = [p[0]["content"] for p in efficiency.speedbench_prompts("8k", "low_entropy", 3)]
    assert first == [f"{srcs[i]}#{i}" for i in range(3)]  # default: file order
    got = [p[0]["content"] for p in efficiency.speedbench_prompts("8k", "low_entropy", 8, balanced=True, exclude=("jetbrains_lca",))]
    kinds = [efficiency.prompt_source({"src_id": g.split("#")[0]}) for g in got]
    assert kinds == ["repobench_python", "textsort"] * 3 + ["repobench_python"] * 2  # in turn, a used-up source skipped
    assert len(set(got)) == 8 and got == [p[0]["content"] for p in efficiency.speedbench_prompts(
        "8k", "low_entropy", 8, balanced=True, exclude=("jetbrains_lca",))]  # same seed, same pick
    fewer = [p[0]["content"] for p in efficiency.speedbench_prompts("8k", "low_entropy", 5, balanced=True, exclude=("jetbrains_lca",))]
    assert got[:5] == fewer  # more questions extend the pick: cross-engine-n64's first 32 are cross-engine's 32


def test_spec_delta_gives_per_request_acceptance():
    a = {"rounds": 10, "proposed": 150, "accepted": 30, "emitted": 40, "depth": 15}
    b = {"rounds": 30, "proposed": 750, "accepted": 190, "emitted": 240, "depth": 15}
    d = efficiency.spec_delta(a, b)
    # 600 proposed at depth 15 = 40 request-rounds: 160 of 600 drafts accepted, 200 tokens over 40 rounds
    assert d["accept_rate"] == 160 / 600 and d["accept_length"] == 5.0 and d["rounds"] == 20
    assert efficiency.spec_delta(None, b) is None


def test_spec_delta_counts_request_rounds_when_the_depth_changes_with_the_batch():
    # one request-round at depth 15 (5 accepted, 6 emitted) and one at depth 6 (2 accepted, 3 emitted); the server
    # reports the last round's depth, 6, which would turn 21 proposed into 3.5 rounds
    a = {"rounds": 0, "proposed": 0, "accepted": 0, "emitted": 0, "depth": 15}
    b = {"rounds": 2, "proposed": 21, "accepted": 7, "emitted": 9, "depth": 6}
    d = efficiency.spec_delta(a, b)
    assert d["accept_length"] == 4.5 and d["accept_rate"] == 7 / 21


def test_balanced_small_source_runs_fewer_bursts(tmp_path, monkeypatch):
    from evals import suite
    from evals.speed import model
    monkeypatch.setattr(model, "snapshot_dir", lambda snapshot=None: tmp_path)
    monkeypatch.setattr(suite, "evals_tree", lambda: "test")
    for name, value in (("alive", lambda url: True), ("describe", lambda url, engine: {}), ("resolved", lambda name: None),
                        ("aiperf_command", lambda a: ["aiperf"]), ("flush", lambda url, engine: None), ("WORK", tmp_path / "w")):
        monkeypatch.setattr(bench, name, value)
    monkeypatch.setattr(efficiency, "speedbench_prompts",
                        lambda c, k, n, **kw: [[{"role": "user", "content": str(i)}] for i in range(min(n, 5))])
    timed = []

    def run_aiperf(cell, rows_, n, ctx, raw=False):
        cell.mkdir(parents=True, exist_ok=True)
        if not cell.name.startswith("prime"):
            timed.append(cell.name)
        return cell
    monkeypatch.setattr(efficiency, "run_aiperf", run_aiperf)
    monkeypatch.setattr(efficiency, "token_counter", lambda snapshot: None)
    counter = {"rounds": 0, "proposed": 0, "accepted": 0, "emitted": 0, "depth": 15}

    def spec_counters(url):
        counter.update(rounds=counter["rounds"] + 2, proposed=counter["proposed"] + 15, accepted=counter["accepted"] + 3,
                       emitted=counter["emitted"] + 4)
        return dict(counter)
    monkeypatch.setattr(efficiency, "spec_counters", spec_counters)
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want, count: {
        "requests": n, "errors": 0, "short": 0, "held_back": 0, "cache_hit_min": 1.0, "peak_decoding": n,
        "aggregate_decode_tok_s": 10.0, "per_stream_tok_s_p50": 10.0, "window_share": 1.0})
    args = argparse.Namespace(engine="openai", engine_version="x", url="http://x", model="m", snapshot=None, aiperf=None,
                              out=tmp_path / "run", context=["8k"], categories=["low_entropy"], concurrency=[1, 2, 4, 8],
                              output_tokens=16, temperature=0.0, top_p=0.95, reasoning="off", min_hit=0.95, questions=12,
                              balance=True, exclude_sources=["textsort"], prompt_seed=0, max_tokens_in_flight=None,
                              no_checks=True)
    assert efficiency.warm_decode(args) == 0
    # 5 prompts: N 1 five bursts, N 2 two, N 4 one, N 8 not run
    assert timed == ["c1", "c1-r1", "c1-r2", "c1-r3", "c1-r4", "c2", "c2-r1", "c4"]
    cell = json.loads((args.out / "warm" / "8k-low_entropy" / "c1" / "cell.json").read_text())["summary"]
    assert cell["spec"]["accept_length"] == 4.0 and cell["spec"]["accept_rate"] == 0.2


def test_flush_cmd_runs_the_command_instead_of_the_flush_endpoint(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(efficiency.bench, "flush", lambda url, engine: calls.append("endpoint"))
    monkeypatch.setattr(efficiency.bench, "alive", lambda url: True)
    marker = tmp_path / "restarted"
    efficiency.flush(argparse.Namespace(engine="openai", flush_cmd=f"echo once >> {marker}"), "http://x")
    assert marker.read_text() == "once\n" and calls == []
    efficiency.flush(argparse.Namespace(engine="sharar", flush_cmd=None), "http://x")
    assert calls == ["endpoint"]


def test_report_of_a_suite_run_reports_its_parts(tmp_path, capsys):
    run = tmp_path / "suite"
    (run / "warm-decode" / "warm").mkdir(parents=True)
    (run / "suite.json").write_text("{}")
    (run / "warm-decode" / "efficiency.json").write_text(json.dumps({"mode": "warm-decode"}))
    (run / "warm-decode" / "run.json").write_text(json.dumps({"engine": "vllm", "engine_version": "x", "output_tokens": 16}))
    assert efficiency.report(argparse.Namespace(runs=[run], recompute=False)) == 0
    assert "engine vllm (x), warm decode" in capsys.readouterr().out


def test_every_suite_keeps_the_cache_hit_floor_vllm_needs():
    assert all(s["warm-decode"]["min_hit"] == 0.6 for s in efficiency.SUITES["suites"].values() if "warm-decode" in s)


def test_prime_passes_send_each_burst_more_than_once_before_timing(tmp_path, monkeypatch):
    from evals import suite
    from evals.speed import model
    monkeypatch.setattr(model, "snapshot_dir", lambda snapshot=None: tmp_path)
    monkeypatch.setattr(suite, "evals_tree", lambda: "test")
    monkeypatch.setattr(bench, "alive", lambda url: True)
    monkeypatch.setattr(bench, "describe", lambda url, engine: {})
    monkeypatch.setattr(bench, "resolved", lambda name: None)
    monkeypatch.setattr(bench, "aiperf_command", lambda a: ["aiperf"])
    monkeypatch.setattr(bench, "flush", lambda url, engine: None)
    monkeypatch.setattr(bench, "WORK", tmp_path / "work")
    monkeypatch.setattr(efficiency, "speedbench_prompts", lambda c, k, n: [[{"role": "user", "content": str(i)}] for i in range(n)])
    calls = []

    def run_aiperf(cell, rows_, n, ctx, raw=False):
        cell.mkdir(parents=True, exist_ok=True)
        calls.append(cell.name)
        return cell
    monkeypatch.setattr(efficiency, "run_aiperf", run_aiperf)
    monkeypatch.setattr(efficiency, "rows", lambda ps, *a: [{"messages": p} for p in ps])
    monkeypatch.setattr(efficiency, "token_counter", lambda snapshot: None)
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want, count: {
        "requests": n, "errors": 0, "short": 0, "held_back": 0, "cache_hit_min": 1.0, "peak_decoding": n,
        "aggregate_decode_tok_s": 10.0, "per_stream_tok_s_p50": 10.0, "window_share": 1.0})
    args = argparse.Namespace(engine="openai", engine_version="x", url="http://x", model="m", snapshot=None, aiperf=None,
                              out=tmp_path / "run", context=["8k"], categories=["low_entropy"], concurrency=[1],
                              output_tokens=16, temperature=0.0, top_p=0.95, reasoning="off", min_hit=0.95, questions=1,
                              prime_passes=2, max_tokens_in_flight=None, no_checks=True)
    assert efficiency.warm_decode(args) == 0
    assert calls == ["prime-c1", "prime2-c1", "c1"]
    assert json.loads((args.out / "efficiency.json").read_text())["prime_passes"] == 2


def test_suite_context_runs_only_those_of_its_contexts(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(efficiency, "warm_decode", lambda ns: seen.append(ns.context) or 0)
    monkeypatch.setattr(efficiency.bench, "alive", lambda url: True)
    args = argparse.Namespace(suite="cross-engine", engine="openai", engine_version="x", url="http://x", model=None,
                              snapshot=None, aiperf=None, no_checks=True, out=tmp_path, flush_cmd=None, prime_passes=1,
                              min_hit=None, context=["16k"])
    assert efficiency.suite(args) == 0
    assert seen == [["16k"]] and json.loads((tmp_path / "suite.json").read_text())["context"] == ["16k"]
    args.context = ["4k"]
    with pytest.raises(SystemExit):
        efficiency.suite(args)
