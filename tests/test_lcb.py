import base64
import json
import pickle
import zlib

import pytest

from evals import accuracy_eval, datasets, lcb, suite


def encode(tests: list[dict]) -> str:
    """The dataset's private-test encoding: base64 of zlib of a pickled JSON string."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def problem(pid, tests, fn_name=None, starter="", public=1):
    """A hand-made problem: the first `public` tests public (JSON), the rest private (encoded)."""
    kind = "functional" if fn_name else "stdin"
    tests = [{"input": i, "output": o, "testtype": kind} for i, o in tests]
    return {"id": pid, "group": "easy", "platform": "leetcode" if fn_name else "atcoder", "date": "2025-03-01",
            "question": f"Question {pid}.", "starter_code": starter, "fn_name": fn_name,
            "public_tests": json.dumps(tests[:public]), "private_tests": encode(tests[public:])}


# stdin: print the sum of two integers on one line, and their mean to 6 decimals on the next
SUM = problem("sum", [("1 2\n", "3\n1.500000\n"), ("10 -4\n", "6\n3.000000\n"), ("0 0\n", "0\n0.000000\n")])
SUM_OK = "a, b = map(int, input().split())\nprint(a + b)\nprint(f'{(a + b) / 2:.6f}')"
# functional: the indices of the two numbers that add up to target
TWO = problem("two", [('[2, 7, 11, 15]\n9', "[0, 1]"), ('[3, 2, 4]\n6', "[1, 2]"), ('[3, 3]\n6', "[0, 1]")], fn_name="twoSum",
              starter="class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        ")
TWO_OK = """class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, x in enumerate(nums):
            if target - x in seen:
                return (seen[target - x], i)  # a tuple counts as the list
            seen[x] = i
"""


def test_prompt_is_lcb_runners_chat_format():
    assert lcb.SYSTEM.startswith("You are an expert Python programmer. You will be given a question")
    assert lcb.prompt(SUM) == (
        "### Question:\nQuestion sum.\n\n"
        "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
        "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
        "inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n")
    assert lcb.prompt(TWO) == (
        "### Question:\nQuestion two.\n\n"
        "### Format: You will use the following starter code to write the solution to the problem and enclose your code within "
        f"delimiters.\n```python\n{TWO['starter_code']}\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n")


def test_extract_code_takes_the_last_fenced_block():
    reply = "First try:\n```python\nprint(1)\n```\nBetter:\n```python\nimport sys\nprint(2)\n```\nDone."
    assert lcb.extract_code(reply) == "import sys\nprint(2)"
    assert lcb.extract_code("```\nprint(3)\n```") == "print(3)"
    assert lcb.extract_code("print(4)") == "" and lcb.extract_code("```python\nprint(5)") == ""  # unclosed: no code


def test_private_tests_decode_and_refuse_objects():
    tests = [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
    assert lcb.decode_tests(json.dumps(tests)) == tests
    assert lcb.decode_tests(encode(tests)) == tests
    evil = base64.b64encode(zlib.compress(pickle.dumps(pytest.raises))).decode()  # a pickle that would import
    with pytest.raises(pickle.UnpicklingError):
        lcb.decode_tests(evil)
    row = lcb.grade({**SUM, "private_tests": evil}, SUM_OK)  # a problem the harness cannot read: wrong, not a crash
    assert not row["correct"] and row["error"].startswith("grading failed")


def test_stdin_comparison_strips_lines_and_compares_numbers_as_decimals():
    def ok(expected: str, printed: str) -> bool:
        it = problem("cmp", [("\n", expected)])
        return lcb.grade(it, f"print({printed!r}, end='')")["correct"]
    assert ok("3\n1.500000\n", "3  \n1.500000")  # trailing whitespace and newline
    assert ok("1.5\n", "1.500000000")  # another formatting of the same number
    assert ok("100 0.25\n", "1e2   .25")
    assert ok("-0\n", "0")
    assert not ok("1.5\n", "1.5000001")  # equal as decimals, no epsilon
    assert not ok("a b\n", "a  b")  # inner whitespace only matters when a field is not a number (lcb_runner's rule)
    assert not ok("1\n2\n", "1")  # line count differs
    assert not ok("YES\n", "yes")


def test_stdin_problem_end_to_end():
    right = lcb.grade(SUM, SUM_OK)
    assert (right["correct"], right["tests"], right["passed"], right["error"]) == (True, 3, 3, "")
    wrong = lcb.grade(SUM, "a, b = map(int, input().split())\nprint(a + b)\nprint((a + b) // 2)")  # fails the first test
    assert (wrong["correct"], wrong["passed"], wrong["error"]) == (False, 0, "Wrong answer at output_line_idx=1: 1 != 1.500000")
    private = lcb.grade(SUM, "a, b = map(int, input().split())\nprint(a + b)\nprint('1.500000' if a == 1 else 0)")
    assert (private["correct"], private["passed"]) == (False, 1)  # the public test passes, the first private one fails
    main = "import sys\ndef main():\n    a, b = map(int, sys.stdin.read().split())\n    print(a + b)\n    print('%.6f' % ((a + b) / 2))\n" \
           "if __name__ == '__main__':\n    main()"
    assert lcb.grade(SUM, main)["correct"]  # a __main__ guard, sys.stdin.read
    assert lcb.grade(SUM, SUM_OK + "\nexit()")["correct"]  # exit() ends the test, not the run
    assert lcb.grade(SUM, "import numpy as np\n" + SUM_OK)["correct"]  # numpy is imported before the guard, as in lcb_runner
    crash = lcb.grade(SUM, "raise ValueError('no')")
    assert (crash["correct"], crash["error"]) == (False, "Runtime Error")
    slow = lcb.grade(SUM, "while True:\n    pass", timeout=1)
    assert (slow["correct"], slow["error"]) == (False, "Time Limit Exceeded")
    assert lcb.grade(SUM, "")["error"] == "no code block"


def test_functional_problem_end_to_end():
    assert lcb.grade(TWO, TWO_OK)["correct"]
    wrong = lcb.grade(TWO, TWO_OK.replace("(seen[target - x], i)", "[i, seen[target - x]]"))
    assert (wrong["correct"], wrong["error"]) == (False, "Wrong Answer")
    assert not lcb.grade(TWO, "class Solution:\n    def other(self):\n        pass")["correct"]  # no twoSum
    assert not lcb.grade(TWO, "class Solution:\n    def twoSum(self, nums, target):\n        exit()")["correct"]


def test_grading_runs_in_parallel():
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda _: lcb.grade(SUM, "import time\ntime.sleep(0.5)\n" + SUM_OK), range(8)))
    assert all(r["correct"] for r in rows)
    assert time.perf_counter() - t0 < 8 * 0.5 * 3 / 2  # 8 problems x 3 tests x 0.5 s, sequential: 12 s


@pytest.fixture
def request_mode():
    saved = dict(accuracy_eval.SAMPLING), dict(accuracy_eval.THINKING)
    yield accuracy_eval
    for d, v in zip((accuracy_eval.SAMPLING, accuracy_eval.THINKING), saved):
        d.clear()
        d.update(v)


def test_run_sends_the_system_prompt_and_grades_the_reply_after_thinking(request_mode, monkeypatch, tmp_path):
    sent = []
    replies = {"sum": f"<think>```python\nprint(0)\n```</think>\n\nHere:\n```python\n{SUM_OK}\n```",  # no reasoning parser
               "two": f"```python\nprint(0)\n```"}
    request_mode.THINKING.update(thinking=True)

    def post(url, body, timeout):
        sent.append(body)
        pid = "sum" if "Question sum." in body["messages"][1]["content"] else "two"
        if pid == "two":  # thinking holds the right code, the reply the wrong one
            msg = {"content": replies[pid], "reasoning_content": f"```python\n{TWO_OK}\n```"}
        else:
            msg = {"content": replies[pid]}
        return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 50}}
    monkeypatch.setattr(accuracy_eval, "post", post)
    rows = lcb.run("lcb_v6", "u", [SUM, TWO], None, 32768, 2)
    assert [(r["id"], r["correct"]) for r in rows] == [("sum", True), ("two", False)]
    assert [m["role"] for m in sent[0]["messages"]] == ["system", "user"] and sent[0]["messages"][0]["content"] == lcb.SYSTEM
    assert rows[1]["reasoning_tail"].startswith("```python") and rows[0]["group"] == "easy"

    # through the suite: `--tasks lcb_v6` writes a shard of graded items
    monkeypatch.setattr(suite, "task_items", lambda name: [SUM, TWO])
    suite.run_task(tmp_path, "lcb_v6", 0, "u", None, 2, 0)
    shard = json.loads((tmp_path / "lcb_v6" / "s00000.json").read_text())
    assert [(x["index"], x["id"], x["correct"]) for x in shard["items"]] == [(0, "sum", True), (1, "two", False)]


def test_lcb_is_registered_pinned_and_in_no_suite():
    assert suite.TASKS["lcb_v6"][0] is lcb and suite.task_max_tokens("lcb_v6") == 131072 and suite.task_files("lcb_v6") == ["lcb/test6.jsonl"]
    assert not any(name == "lcb_v6" for tasks in suite.SUITES.values() for name, _ in tasks)
    assert "lcb/test6.jsonl" in datasets.pins() and "0fe84c3912ea0c4d4a78037083943e8f0c4dd505" in datasets.FILES["lcb/test6.jsonl"]


def test_the_pinned_set():
    if not (datasets.ROOT / lcb.FILE).is_file():
        pytest.skip(f"{lcb.FILE} not fetched (sharar-eval datasets fetch)")
    datasets.check([lcb.FILE])
    items = lcb.load("lcb_v6")
    assert len(items) == len({it["id"] for it in items}) == 175
    assert min(it["date"] for it in items) == "2025-01-04" and max(it["date"] for it in items) == "2025-04-06"
    assert sum(it["fn_name"] is not None for it in items) == sum(it["platform"] == "leetcode" for it in items) == 63
    first = items[0]
    assert first["id"] == "abc387_b" and len(lcb.decode_tests(first["public_tests"]) + lcb.decode_tests(first["private_tests"])) > 2
