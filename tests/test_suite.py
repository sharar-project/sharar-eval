import json
import zlib

import pytest

from evals import suite


def write_run(path, label, tasks):
    path.mkdir()
    (path / "run.json").write_text(json.dumps({"label": label, "engine": "img", "tasks": {t: {"max_tokens": 10} for t in tasks}}))
    for task, items in tasks.items():
        (path / task).mkdir()
        (path / task / "s00000.json").write_text(json.dumps({"task": task, "start": 0, "items": items}))


def test_show_and_compare_read_the_same_shards(tmp_path):
    a = [{"index": i, "correct": i % 2 == 0, "completion_tokens": 10 if i == 0 else 4} for i in range(4)]
    b = [{**x, "correct": True} for x in a]
    write_run(tmp_path / "a", "exact", {"gsm8k": a})
    write_run(tmp_path / "b", "fast", {"gsm8k": b})
    shown = suite.summary(tmp_path / "a")["tasks"]["gsm8k"]
    assert (shown["n"], shown["score"], shown["capped"], shown["tokens_per_item"]) == (4, 50.0, 1, 5.5)
    card = suite.scorecard(tmp_path / "a", tmp_path / "b")
    row = card["tasks"]["gsm8k"]
    assert (row["wins"], row["losses"], row["delta"]) == (2, 0, 50.0)
    assert card["pooled"]["n"] == 4


def test_evals_tree_hashes_the_shipped_files(tmp_path, monkeypatch):
    """The same code gives the same hash wherever it lives; any shipped file changing changes it, and files the
    package does not ship (caches, notes) do not."""
    import shutil

    from evals import suite
    copy = tmp_path / "evals"
    shutil.copytree(suite.PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(suite, "PACKAGE", copy)
    first = suite.evals_tree()
    assert first.startswith("sha256:")
    (copy / "__pycache__").mkdir(exist_ok=True)
    (copy / "__pycache__" / "x.pyc").write_bytes(b"cache")
    (copy / "notes.txt").write_text("not shipped")
    assert suite.evals_tree() == first
    (copy / "suite.py").write_text((copy / "suite.py").read_text() + "\n# changed\n")
    assert suite.evals_tree() != first


def test_max_tokens_replaces_each_cap_and_a_run_dir_keeps_one(tmp_path, monkeypatch):
    import argparse

    import pytest
    monkeypatch.setattr(suite, "server_info", lambda url: {"target": "t"})
    monkeypatch.setattr(suite.datasets, "check", lambda names: {})
    assert (suite.task_max_tokens("gpqa"), suite.task_max_tokens("gpqa", 32768), suite.task_max_tokens("needles", 32768)) == (8192, 32768, None)

    def args(cap):
        return argparse.Namespace(label="fast", url="u", model="sharar", plain=False, accept="server", concurrency=4,
                                  engine="e", suite=None, tasks="gpqa", limit=0, max_tokens=cap)
    suite.record(tmp_path, args(32768), ["gpqa"])
    assert json.loads((tmp_path / "run.json").read_text())["tasks"]["gpqa"] == {"max_tokens": 32768}
    suite.record(tmp_path, args(32768), ["gpqa"])  # resuming with the same cap
    with pytest.raises(SystemExit, match="same --max-tokens"):
        suite.record(tmp_path, args(0), ["gpqa"])


@pytest.fixture
def request_mode():
    """accuracy_eval's module-wide request settings, restored after the test."""
    from evals import accuracy_eval
    saved = dict(accuracy_eval.SAMPLING), dict(accuracy_eval.THINKING)
    yield accuracy_eval
    for d, v in zip((accuracy_eval.SAMPLING, accuracy_eval.THINKING), saved):
        d.clear()
        d.update(v)


def test_thinking_requests_and_replies(request_mode, monkeypatch):
    import argparse
    ae = request_mode
    sent, replies = [], []
    monkeypatch.setattr(ae, "post", lambda url, body, timeout: (sent.append(body), replies.pop(0))[1])

    def reply(content, reasoning=None, finish="stop"):
        msg = {"content": content, **({"reasoning_content": reasoning} if reasoning else {})}
        return {"choices": [{"message": msg, "finish_reason": finish}], "usage": {"completion_tokens": 9, "reasoning_tokens": 7}}

    replies.append(reply("Answer: B"))
    assert ae.ask("u", "q", 100, None) == ("Answer: B", {"completion_tokens": 9, "reasoning_tokens": 7})
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": False} and "seed" not in sent[-1]  # greedy, as before

    ap = argparse.ArgumentParser()
    ae.add_request_args(ap)
    ae.apply_request_args(ap, ap.parse_args(["--thinking", "--reasoning-effort", "medium", "--temperature", "1.0", "--seed", "3"]))
    assert ae.request_mode() == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "thinking": True, "reasoning_effort": "medium", "seed": 3}
    replies.append(reply("Answer: C", reasoning="x" * 500 + "so C"))
    text, usage = ae.ask("u", "q", 100, None)
    assert (text, usage["reasoning_tail"]) == ("Answer: C", ("x" * 500 + "so C")[-400:])
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert sent[-1]["seed"] == zlib.crc32(b"q", 3) != zlib.crc32(b"q")  # --seed 0 keeps the seeds sampled runs had
    replies.append(reply("thinking...</think>\n\nAnswer: D"))  # a server without a reasoning parser
    assert ae.ask("u", "q", 100, None)[0] == "Answer: D"
    replies.append(reply("thinking that never ends", finish="length"))  # and capped inside the thinking
    text, usage = ae.ask("u", "q", 100, None)
    assert (text, usage["reasoning_tail"]) == ("", "thinking that never ends")
    for bad in (["--seed", "1"], ["--reasoning-effort", "low"]):
        with pytest.raises(SystemExit):
            ae.apply_request_args(ap, ap.parse_args(bad))


def test_a_run_dir_keeps_one_request_mode(tmp_path, monkeypatch, request_mode):
    """A run directory from before the request mode was recorded resumes as greedy with thinking off; a thinking run
    cannot write into it, and needs a cap and no tasks that send their own requests."""
    import argparse
    monkeypatch.setattr(suite, "server_info", lambda url: {"target": "t"})
    monkeypatch.setattr(suite.datasets, "check", lambda names: {})
    args = argparse.Namespace(label="fast", url="u", model="sharar", plain=False, accept="server", concurrency=4,
                              engine="e", suite=None, tasks="gpqa", limit=0, max_tokens=0)
    suite.record(tmp_path, args, ["gpqa"])
    rec = json.loads((tmp_path / "run.json").read_text())
    assert rec["request"] == suite.GREEDY
    del rec["request"]
    (tmp_path / "run.json").write_text(json.dumps(rec))
    suite.record(tmp_path, args, ["gpqa"])  # an older run directory resumes
    request_mode.THINKING.update(thinking=True)
    with pytest.raises(SystemExit, match="request differ"):
        suite.record(tmp_path, args, ["gpqa"])
    for argv, why in ((["--thinking"], "needs --max-tokens"), (["--thinking", "--max-tokens", "65536", "--tasks", "gpqa,needles"], "needles")):
        with pytest.raises(SystemExit):
            suite.cmd_run(["--label", "x", "--out", str(tmp_path / "new"), *argv])
        assert not (tmp_path / "new").exists(), why


def test_show_means_repeats(tmp_path, capsys):
    for k, correct in enumerate(([True, True, False, False], [True, True, True, False])):
        write_run(tmp_path / f"s{k}", f"seed{k}", {"gpqa": [{"index": i, "correct": c, "completion_tokens": 4} for i, c in enumerate(correct)]})
    rows = suite.mean_of([suite.summary(tmp_path / "s0"), suite.summary(tmp_path / "s1")])
    assert (rows["gpqa"]["scores"], rows["gpqa"]["mean"], round(rows["gpqa"]["sd"], 2)) == ([50.0, 75.0], 62.5, 17.68)
    assert suite.cmd_show([str(tmp_path / "s0"), str(tmp_path / "s1")]) == 0
    assert "| gpqa | 4 | 50.0% | 75.0% | 62.5% | 17.7 |" in capsys.readouterr().out


def test_a_shard_is_saved_as_its_items_complete(tmp_path, monkeypatch):
    """Items run one request each; the completed ones from the shard's start are saved whenever they grow, a failure is
    retried once without stopping the others, and a stopped shard resumes after what it saved."""
    import threading
    gate, calls = threading.Event(), {}
    monkeypatch.setattr(suite, "task_items", lambda name: [{"id": f"q{i}"} for i in range(5)])

    def fake_shard(out, name, url, items, accept, concurrency, max_tokens):
        assert len(items) == 1 and concurrency == 1
        i = int(items[0]["id"][1:])
        calls[i] = calls.get(i, 0) + 1
        if i == 0:
            gate.wait(5)  # the first item finishes last
        if i == 2 and calls[i] == 1:
            raise TimeoutError("timed out")  # retried
        if i == 3:
            raise TimeoutError("always")  # fails twice: raised at the end
        return [{"index": 0, "id": items[0]["id"], "correct": True, "completion_tokens": 1}]
    monkeypatch.setattr(suite, "run_shard", fake_shard)
    saved = []
    real_save = suite.save_shard
    monkeypatch.setattr(suite, "save_shard", lambda *a: (saved.append([x["id"] for x in a[3]]), real_save(*a)))
    threading.Timer(0.3, gate.set).start()
    with pytest.raises(TimeoutError, match="always"):
        suite.run_task(tmp_path, "gsm8k", 0, "u", None, 4, 0)
    assert calls[2] == 2 and calls[3] == 2
    assert saved == [["q0", "q1", "q2"]]  # nothing before q0 completed; q3 never did
    shard = json.loads((tmp_path / "gsm8k" / "s00000.json").read_text())
    assert [x["index"] for x in shard["items"]] == [0, 1, 2]

    calls.clear()
    monkeypatch.setattr(suite, "run_shard", lambda out, name, url, items, *a: (
        calls.setdefault(items[0]["id"], 1), [{"index": 0, "id": items[0]["id"], "correct": False, "completion_tokens": 1}])[1])
    suite.run_task(tmp_path, "gsm8k", 0, "u", None, 4, 0)  # resumes at q3
    assert sorted(calls) == ["q3", "q4"]
    shard = json.loads((tmp_path / "gsm8k" / "s00000.json").read_text())
    assert [x["index"] for x in shard["items"]] == [0, 1, 2, 3, 4] and [x["id"] for x in shard["items"]] == ["q0", "q1", "q2", "q3", "q4"]
