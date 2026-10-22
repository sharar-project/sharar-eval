import argparse
import json
import random
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from evals import agent_replay as ar
from evals import efficiency

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command",
                                           "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def replay_dir(tmp_path, turns=(3, 2)):
    """A replay directory in agentperf-local's layout: a manifest and one trace per task."""
    d = tmp_path / "replay"
    (d / "traces").mkdir(parents=True)
    tasks = []
    for k, n in enumerate(turns):
        rows = [{"conversation_idx": i, "messages": [{"role": "system", "content": f"task {k}"},
                                                     *[{"role": "user", "content": f"turn {j}"} for j in range(i + 1)]],
                 "tools": TOOLS, "target_output_tokens": 5 + i, "recorded_prompt_tokens": 100 * (i + 1),
                 "simulated_tool_delay_ms_after": 0} for i in range(n)]
        (d / "traces" / f"t{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        tasks.append({"task_id": f"t{k}", "trace": f"traces/t{k}.jsonl"})
    (d / "manifest-arm64.json").write_text(json.dumps({"name": "test", "tasks": tasks}))
    return d


@pytest.fixture
def server(tmp_path, monkeypatch):
    from evals import mock_server
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(mock_server.Handler, "log_path", str(log))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", log
    srv.shutdown()


def test_level_runs_copies_past_the_task_count_each_under_its_own_namespace(tmp_path, server):
    url, log = server
    tasks = ar.load_tasks(replay_dir(tmp_path))
    out = tmp_path / "agents.jsonl"
    a = argparse.Namespace(url=url, model="mock-model", label="t", tool_mode="none")
    with out.open("a") as f:
        lv = ar.level(a, tasks, 3, f, random.Random(0))  # 3 agents over 2 tasks: 2 copies, 4 tasks, 10 turns
    assert lv["tasks"] == 4 and lv["turns"] == 10 and lv["failed_turns"] == 0
    sent = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(sent) == 10
    assert {(r["max_tokens"], r["ignore_eos"], r["temperature"], r["top_p"], r["top_k"]) for r in sent if r["max_tokens"] == 5} \
        == {(5, True, 0.7, 0.8, 20)}
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} and r["tools"] == 1 for r in sent)
    heads = {r["first"] for r in sent}  # first 40 characters of the first message: the copy's namespace
    assert len(heads) == 2
    turns = [json.loads(line) for line in out.read_text().splitlines()]
    assert {(t["task"], t["copy"]) for t in turns} == {("t0", 0), ("t1", 0), ("t0", 1), ("t1", 1)}
    s = ar.summarize(turns + [lv])[0]
    assert s["tasks"] == 4 and s["turns"] == 10 and s["failed_turns"] == 0 and s["output_tokens"] == 2 * (5 + 6 + 7 + 5 + 6)


def test_check_and_summarize_apply_agentperfs_acceptance_rule():
    row = {"target_output_tokens": 10, "tools": TOOLS}
    ok = {"done": True, "completion_tokens": 10, "finish": "tool_calls", "tool_calls": [{"name": "bash", "arguments": "{"}]}
    assert ar.check(ok, row) == ["tool: arguments not JSON"]
    assert ar.check({**ok, "completion_tokens": 7, "tool_calls": [{"name": "rm", "arguments": "[]"}]}, row) == [
        "exact: 7 of 10 tokens, finish tool_calls", "tool: unknown name 'rm'", "tool: arguments not an object"]
    assert ar.check({"error": "URLError: refused"}, row) == ["http: URLError: refused"]
    base = {"kind": "agent_turn", "label": "L", "agents": 2, "done": True, "target": 10, "prompt_tokens": 100,
            "cached_tokens": 50, "tool_calls": [], "finish": "length"}
    turns = [{**base, "task": "a", "copy": 0, "t0": 0.0, "first": 0.5, "last": 1.5, "end": 1.6, "completion_tokens": 10},
             {**base, "task": "a", "copy": 0, "t0": 3.0, "first": 3.5, "last": 4.5, "end": 4.6, "completion_tokens": 10},
             {**base, "task": "b", "copy": 0, "t0": 0.0, "first": 0.2, "last": 1.0, "end": 1.0, "completion_tokens": 9}]
    s = ar.summarize(turns + [{"kind": "agent_level", "label": "L", "agents": 2, "wall_s": 5.0}])[0]
    # tool-call validity is not judged, a short output fails; task time runs from first sent to last ended
    assert s["failed_turns"] == 1 and s["tasks"] == 2 and s["task_p50_s"] == pytest.approx(2.8)
    assert s["aggregate_tok_s"] == pytest.approx(29 / 5) and s["cache_share"] == 0.5
    assert s["decode_tok_s_pooled"] == pytest.approx((9 + 9 + 8) / (1.0 + 1.0 + 0.8), abs=0.05)


def test_agent_replay_writes_engine_bench_records_and_resumes(tmp_path, server, monkeypatch, capsys):
    url, _ = server
    replay = replay_dir(tmp_path)
    out = tmp_path / "run"
    argv = ["efficiency", "agent-replay", "--engine", "openai", "--engine-version", "test", "--url", url, "--out", str(out),
            "--agents", "1,2", "--replay", str(replay), "--no-checks", "--label", "mock"]
    monkeypatch.setattr(sys, "argv", argv)
    assert efficiency.main() == 0
    records = efficiency.read_records(out / "agents.jsonl")
    assert sorted(r["agents"] for r in records if r["kind"] == "agent_level") == [1, 2]
    assert sum(r["kind"] == "agent_turn" for r in records) == 10
    rows = (out / "agents.csv").read_text().splitlines()
    assert rows[0].startswith("label,agents,tasks,turns,wall_s,aggregate_tok_s,tasks_per_hour") and len(rows) == 3
    # a level cut short (turns, no level record) is dropped and rerun; finished levels are kept
    kept = [r for r in records if not (r["agents"] == 2 and r["kind"] == "agent_level")]
    (out / "agents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    capsys.readouterr()
    assert efficiency.main() == 0
    printed = capsys.readouterr().out
    assert "(done)" in printed.splitlines()[2] and "(done)" not in printed.splitlines()[3]
    records = efficiency.read_records(out / "agents.jsonl")
    assert sum(r["kind"] == "agent_turn" for r in records) == 10
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["other"])
    with pytest.raises(SystemExit, match="other options"):
        efficiency.main()


def test_every_suite_option_is_a_flag_and_the_suite_sets_every_flag():
    real = argparse.ArgumentParser.parse_args
    for part, options in ((p, o) for s in efficiency.SUITES["suites"].values() for p, o in s.items()):
        assert part in efficiency.PARTS
        argv = [part, "--engine", "openai", "--engine-version", "x", "--url", "http://x", "--out", "o"]
        for key, value in options.items():
            flag = "--" + key.replace("_", "-")
            argv += [flag] if value is True else [flag, ",".join(map(str, value)) if isinstance(value, list) else str(value)]
        captured = {}

        def grab(self, args=None, namespace=None):
            captured["ns"] = real(self, argv, namespace)
            raise SystemExit(0)
        argparse.ArgumentParser.parse_args = grab
        try:
            with pytest.raises(SystemExit):
                efficiency.main()
        finally:
            argparse.ArgumentParser.parse_args = real
        ns = vars(captured["ns"])
        for key, value in options.items():
            assert ns[key] == value, (part, key)
        suite_keys = {"engine", "engine_version", "url", "model", "snapshot", "aiperf", "no_checks", "out",
                      *efficiency.DEFAULTS[part], *options}
        assert set(ns) - {"mode"} <= suite_keys, (part, set(ns) - suite_keys)
        for key, value in efficiency.DEFAULTS[part].items():
            assert ns[key] == value, (part, key)
