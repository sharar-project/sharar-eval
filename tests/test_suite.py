import json

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
