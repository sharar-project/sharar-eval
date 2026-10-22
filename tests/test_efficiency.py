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
