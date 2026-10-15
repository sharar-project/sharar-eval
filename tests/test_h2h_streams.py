"""h2h --streams: the closed-loop sweep's window counting and prompt order."""
from evals.h2h import interleaved, window_tokens


def test_window_tokens_spreads_a_request_over_its_span():
    r = {"t_first": 10.0, "t_last": 20.0, "completion_tokens": 100}
    assert window_tokens(r, 0, 30) == 100
    assert window_tokens(r, 15, 30) == 50
    assert window_tokens(r, 12, 14) == 20
    assert window_tokens(r, 25, 30) == 0
    assert window_tokens({"t_first": 5.0, "t_last": 5.0, "completion_tokens": 1}, 0, 10) == 1
    assert window_tokens({"t_first": None, "t_last": None, "completion_tokens": 9}, 0, 10) == 0


def test_interleaved_rotates_categories():
    prompts = [{"id": f"{c}{i}", "cat": c} for c in "abc" for i in range(3)] + [{"id": "a3", "cat": "a"}]
    order = [p["cat"] for p in interleaved(prompts)]
    assert order[:6] == ["a", "b", "c", "a", "b", "c"] and sorted(order) == sorted(p["cat"] for p in prompts)
