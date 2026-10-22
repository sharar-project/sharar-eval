import argparse
import json

import pytest

from evals import bench, efficiency


def record(start_s, ttft_ms, gaps_ms, osl, prompt=1000, cached=990, cid="s"):
    end = start_s + (ttft_ms + sum(gaps_ms)) / 1e3
    return {"metadata": {"conversation_id": cid, "request_start_ns": int(start_s * 1e9), "request_end_ns": int(end * 1e9)},
            "metrics": {"time_to_first_token": {"value": ttft_ms}, "inter_chunk_latency": {"value": gaps_ms},
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
    assert timed[0]["extra"]["temperature"] == 0.0


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
        (cell / "cell.json").write_text(json.dumps({"summary": summary}))
    (run / "warm" / "16k-low_entropy" / "stopped.json").write_text(json.dumps({"n": 2, "reason": "cache hit 0%"}))
    efficiency.report(argparse.Namespace(runs=[run]))
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

    def run_aiperf(cell, rows_, n, ctx):
        cell.mkdir(parents=True, exist_ok=True)
        if not cell.name.startswith("prime"):
            timed.append(n)
        return cell
    monkeypatch.setattr(efficiency, "run_aiperf", run_aiperf)
    good = {"requests": 0, "errors": 0, "short": 0, "cache_hit_min": 1.0, "aggregate_decode_tok_s": 1.0, "window_share": 1.0}
    # the engine decodes at most 4 at once
    monkeypatch.setattr(efficiency, "warm_summary", lambda art, n, want: {**good, "requests": n, "peak_decoding": min(n, 4)})
    args = argparse.Namespace(engine="openai", engine_version="x", url="http://x", model="m", snapshot=None, aiperf=None,
                              out=tmp_path / "run", context=["8k"], categories=["low_entropy"], concurrency=[1, 4, 8, 16],
                              output_tokens=16, min_hit=0.95, max_tokens_in_flight=None, no_checks=True)
    assert efficiency.warm_decode(args) == 0
    assert timed == [1, 4, 8]  # N = 8 ran once, not retried; N = 16 not run
    stopped = json.loads((args.out / "warm" / "8k-low_entropy" / "stopped.json").read_text())
    assert stopped["n"] == 8 and stopped["reason"].startswith("decodes at most 4 of 8 at once")
