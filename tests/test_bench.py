import json
import re
import sys
import threading
from pathlib import Path

import pytest

from evals import bench


def test_every_suite_cell_names_a_known_workload():
    for suite in bench.SUITE["suites"]:
        names = [n for n, _ in bench.cells(suite)]
        assert len(names) == len(set(names)) > 0
        for name, spec in bench.cells(suite):
            assert spec["workload"] in ("speedbench", "speedbench-qual", *bench.FIXED)
            if spec["workload"] == "speedbench":
                assert re.fullmatch(r"speed_bench_throughput_(1k|2k|8k|16k|32k)_(low_entropy|mixed|high_entropy)", spec["dataset"])
            elif spec["workload"] == "speedbench-qual":
                assert spec["dataset"] in {f"speed_bench_{c}" for c in QUALITATIVE} and spec["sessions"] == 80
            else:
                from evals.speed import decode
                decode.rows(spec["test"], 1, spec["temperature"], tag=0)
                assert spec["workload"] == "prefill" or spec["requests"] >= spec["concurrency"]


QUALITATIVE = ("coding", "humanities", "math", "multilingual", "qa", "rag", "reasoning", "roleplay", "stem", "summarization",
               "writing")


def test_engines_suite_is_its_two_halves_and_covers_every_category():
    assert [n for n, _ in bench.cells("engines")] == [n for s in ("engines-low", "engines-high") for n, _ in bench.cells(s)]
    names = [n for n, _ in bench.cells("engines")]
    assert {f"speedbench-qual/{c}/c{k}" for c in QUALITATIVE for k in (1, 4, 32)} <= set(names)
    assert {"prefill/128000", "decode/prose-t0.7/c128", "longctx/t0/c32"} <= set(names)
    spec = dict(bench.cells("engines"))["decode/chat-t0/c1"]
    assert spec["requests"] == 8 and dict(bench.cells("engines"))["decode/chat-t0/c64"]["requests"] == 128


def test_configs_read_only_variables_the_launcher_sets():
    common = {"BENCH_URL", "BENCH_MODEL", "BENCH_TOKENIZER", "BENCH_ARTIFACTS", "BENCH_CONCURRENCY"}
    sets = {"speedbench.yaml": common | {"BENCH_DATASET", "BENCH_REQUESTS", "BENCH_SEED"},
            "speedbench_qual.yaml": common | {"BENCH_DATASET", "BENCH_SESSIONS", "BENCH_SEED"},
            "trace.yaml": common | {"BENCH_INPUT", "BENCH_SESSIONS", "BENCH_RAW"}}  # BENCH_RAW defaults to false
    assert {p.name for p in bench.CONFIGS.glob("*.yaml")} == set(sets)
    for name, expected in sets.items():
        used = set(re.findall(r"\$\{(BENCH_[A-Z_]+)", (bench.CONFIGS / name).read_text()))
        assert used == expected, name


def test_engines_name_only_paths():
    for name, engine in bench.SUITE["engines"].items():
        assert set(engine) <= {"flush", "info", "about"}, name
        assert all(engine[k].startswith("/") for k in ("flush", "info") if k in engine)


def test_cell_seeds_are_stable_and_salted():
    assert bench.cell_seed("x") == bench.cell_seed("x") != bench.cell_seed("x", "salt")


def write_exports(artifacts: Path, sessions: dict[str, list[tuple[int, int]]]):
    artifacts.mkdir(parents=True)
    summary = {"request_count": {"avg": 4.0}, "output_token_throughput": {"avg": 100.0},
               "output_token_throughput_per_user": {"p50": 40.0}, "time_to_first_token": {"p50": 250.0, "p99": 900.0},
               "inter_token_latency": {"p50": 20.0}, "input_sequence_length": {"avg": 1000.0},
               "output_sequence_length": {"avg": 50.0}, "usage_prompt_cache_read_tokens": {"avg": 600.0},
               "benchmark_duration": {"avg": 3.0}, "error_summary": []}
    (artifacts / "profile_export_aiperf.json").write_text(json.dumps(summary))
    lines = [json.dumps({"metadata": {"conversation_id": sid, "request_start_ns": a, "request_end_ns": b},
                         "metrics": {"osl_mismatch_diff_pct": {"value": -10.0 if sid == "s2" else 0.0}}})
             for sid, calls in sessions.items() for a, b in calls]
    (artifacts / "profile_export.jsonl").write_text("\n".join(lines) + "\n")


def test_summary_reads_aiperf_exports(tmp_path):
    write_exports(tmp_path / "a", {"s1": [(0, 1_000_000_000), (2_000_000_000, 5_000_000_000)], "s2": [(0, 2_000_000_000)]})
    s = bench.summarize(tmp_path / "a")
    assert s["output_tok_s"] == 100 and s["ttft_s_p50"] == 0.25 and s["cache_read_share"] == 0.6
    assert s["short_outputs"] == 1 and "session_times" not in s


def test_decode_rate_runs_from_first_token_and_prefill_reports_the_second_request(tmp_path):
    art = tmp_path / "p"
    art.mkdir()
    (art / "profile_export_aiperf.json").write_text(json.dumps({"request_count": {"avg": 2.0}}))
    recs = [({"request_start_ns": 0, "request_end_ns": 4_000_000_000},
             {"time_to_first_token": 2000.0, "output_sequence_length": 1, "usage_prompt_tokens": 1000}),
            ({"request_start_ns": 5_000_000_000, "request_end_ns": 6_000_000_000},
             {"time_to_first_token": 500.0, "output_sequence_length": 1, "usage_prompt_tokens": 1010,
              "usage_prompt_cache_read_tokens": 0})]
    (art / "profile_export.jsonl").write_text("".join(
        json.dumps({"metadata": m, "metrics": {k: {"value": v} for k, v in x.items()}}) + "\n" for m, x in recs))
    s = bench.summarize(art, prefill=True)
    assert s["decode_tok_s"] == 2 / 4.0  # 2 tokens, first token at 2 s, last at 6 s
    assert s["prefill_prompt_tokens"] == 1010 and s["prefill_ttft_s"] == 0.5 and s["prefill_tok_s"] == 2020
    assert s["prefill_cached_tokens"] == 0 and "prefill_tok_s" not in bench.summarize(art)
    assert "prefill 2020 tok/s" in bench.row("prefill/8000", s)


FAKE_AIPERF = '''import json, os, pathlib, sys
art = pathlib.Path(os.environ["BENCH_ARTIFACTS"])
art.mkdir(parents=True)
(art.parent / "env.json").write_text(json.dumps({k: v for k, v in os.environ.items() if k.startswith("BENCH_")}))
(art.parent / "argv.json").write_text(json.dumps(sys.argv[1:]))
(art / "profile_export_aiperf.json").write_text(json.dumps({"request_count": {"avg": 1.0}, "output_token_throughput": {"avg": 9.0}}))
(art / "profile_export.jsonl").write_text(json.dumps({"metadata": {"conversation_id": "x", "request_start_ns": 0, "request_end_ns": 1}}) + "\\n")
if "BENCH_DATASET" in os.environ:  # AIPerf's public loader caches the resolved set under its working directory
    d = os.environ["BENCH_DATASET"]
    name = "_".join(d.split("_")[2:4]) if d.startswith("speed_bench_throughput_") else "qualitative"
    p = pathlib.Path(".cache/aiperf/datasets/speed-bench") / f"{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\\n")
'''


def test_run_flushes_before_every_cell_and_resumes(tmp_path, monkeypatch, snapshot):
    from evals import mock_server as mock_chat_server
    from http.server import ThreadingHTTPServer
    flushes = []
    real = mock_chat_server.Handler.do_POST

    def do_post(self):
        if self.path == "/flush_cache":
            flushes.append(1)
        return real(self)
    monkeypatch.setattr(mock_chat_server.Handler, "do_POST", do_post)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mock_chat_server.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake = tmp_path / "aiperf"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_AIPERF)
    fake.chmod(0o755)
    monkeypatch.setitem(bench.SUITE["suites"], "tiny", [
        {"workload": "speedbench", "isl": ["1k"], "entropy": ["low_entropy"], "concurrency": [1, 2], "requests": 4},
        {"workload": "speedbench-qual", "categories": ["coding"], "concurrency": [4], "sessions": 80},
        {"workload": "decode", "tests": ["code"], "temperatures": [0.7], "concurrency": [3], "requests_per_stream": 2,
         "min_requests": 2},
        {"workload": "prefill", "lengths": [4000]}])
    monkeypatch.setattr(bench.datasets, "ROOT", tmp_path / "datasets")
    monkeypatch.setattr(bench, "WORK", tmp_path / "datasets" / "aiperf")
    from evals import checks
    monkeypatch.setattr(checks, "run", lambda url, model: {"model": model, "passed": False, "checks": [
        {"name": "thinking_off", "status": "pass", "detail": "ok"}, {"name": "ignore_eos", "status": "fail", "detail": "60"}]})
    monkeypatch.setattr(sys, "argv", ["bench", "run", "--engine", "hibana", "--engine-version", "test",
                                      "--url", f"http://127.0.0.1:{server.server_address[1]}", "--suite", "tiny",
                                      "--aiperf", str(fake), "--out", str(tmp_path / "run")])
    run_argv = list(sys.argv)
    try:
        with pytest.raises(SystemExit, match="not resolved yet .*qualitative.jsonl.*throughput_1k.jsonl"):
            bench.main()  # SPEED-Bench sets are resolved first, with no engine serving
        monkeypatch.setattr(sys, "argv", ["bench", "prepare", "--suite", "tiny", "--aiperf", str(fake)])
        assert bench.main() == 0
        assert (bench.WORK / ".cache/aiperf/datasets/speed-bench/throughput_1k.jsonl").is_file()
        monkeypatch.setattr(sys, "argv", run_argv)
        assert bench.main() == 0
        assert len(flushes) == 5
        env = json.loads((tmp_path / "run/speedbench/1k-low_entropy/c2/env.json").read_text())
        assert env["BENCH_DATASET"] == "speed_bench_throughput_1k_low_entropy" and env["BENCH_CONCURRENCY"] == "2"
        assert env["BENCH_REQUESTS"] == "4" and env["BENCH_MODEL"] == "mock-model"
        qual = tmp_path / "run/speedbench-qual/coding/c4"
        env = json.loads((qual / "env.json").read_text())
        assert env["BENCH_DATASET"] == "speed_bench_coding" and env["BENCH_SESSIONS"] == "80" and "BENCH_REQUESTS" not in env
        assert json.loads((qual / "argv.json").read_text())[2].endswith("speedbench_qual.yaml")
        cell = tmp_path / "run/decode/code-t0.7/c3"
        rows = [json.loads(line) for line in (cell / "input.jsonl").read_text().splitlines()]
        assert json.loads((cell / "env.json").read_text())["BENCH_SESSIONS"] == "6" == str(len(rows))
        assert rows[0]["output_length"] == 1024 and rows[0]["extra"]["top_k"] == 20
        assert json.loads((cell / "argv.json").read_text())[2].endswith("trace.yaml")
        assert len((tmp_path / "run/prefill/4000/input.jsonl").read_text().splitlines()) == 2
        run = json.loads((tmp_path / "run/run.json").read_text())
        assert run["engine"] == "hibana" and run["server"]["engine"] == "mock"
        assert run["checks"] == {"thinking_off": "pass", "ignore_eos": "fail"}
        assert json.loads((tmp_path / "run/checks.json").read_text())["passed"] is False
        assert bench.main() == 0 and len(flushes) == 5  # finished cells are skipped
        def gone(url, engine):
            raise OSError("connection refused")
        monkeypatch.setattr(bench, "flush", gone)
        monkeypatch.setattr(sys, "argv", [*run_argv[:-1], str(tmp_path / "run2")])
        assert bench.main() == 2  # a failed flush stops the suite instead of running warm cells
    finally:
        server.shutdown()
        server.server_close()  # refuse connections, as a stopped engine does
    with pytest.raises(SystemExit, match="no server answers"):
        bench.main()
