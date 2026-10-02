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
