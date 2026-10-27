"""LiveCodeBench v6 code generation (Jain et al., 2024) against a running server: every problem new in release_v6
(test6.jsonl, contest dates 2025-01-04 to 2025-04-06), the set model reports' "LiveCodeBench v6" scores are out of. 175
problems (112 AtCoder, stdin; 63 LeetCode, functional), in the dataset's order; `group` is the difficulty (80 hard, 52
medium, 43 easy).

    sharar-eval run --tasks lcb_v6 --thinking --max-tokens 131072 --label fast --out runs/fast-think

Ported from LiveCodeBench (lcb_runner at 28fef95, MIT):
- items: livecodebench/code_generation_lite@0fe84c3 (datasets.py pins it). release_v6 is test.jsonl ... test6.jsonl
  (its loading script's ALLOWED_FILES); test6.jsonl holds the problems release_v6 added, the one file read.
- prompt: lcb_runner's for chat models (LMStyle.OpenAIChat): its system message, then the question, the starter code
  (functional) or the stdin instructions, in a ```python block. Requests as in evals.accuracy_eval.
- code: lcb_runner's extract_code on the reply after the thinking: the lines between the last two lines holding
  ```, usually the last ```python block (a reply with fewer than two such lines grades as no code, which fails).
- grading: lcb_runner's run_test (evals/lcb_exec.py): the public tests, then the private ones (base64, zlib, a
  pickled JSON string), in order, stopping at the first failure, each with a 6 s alarm. stdin tests run the program
  as a function on mocked stdin and compare stdout line by line after stripping, a line that differs matching if
  its whitespace-separated fields are equal as decimals; functional tests call Solution().<func_name>(*args), the
  arguments one JSON value per input line, and compare with ==. Correct (pass@1) = every test passes. One
  subprocess per problem (`python -I`, as lcb_runner's one process per generation), killed after lcb_runner's
  overall cap of (6 + 1) s per test + 5 s; differences: a 4 GiB address-space and a CPU-time limit, and a fresh
  interpreter rather than a fork of the evaluator.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.accuracy_eval import ask
from evals.datasets import ROOT

FILE = "lcb/test6.jsonl"
TASKS = {"lcb_v6": ([FILE], 32768)}  # task: (dataset files, default max_tokens)
TIMEOUT = 6  # seconds per test (lcb_runner's --timeout default)
MEMORY = 4 * 1024 ** 3
EXEC = Path(__file__).resolve().parent / "lcb_exec.py"

# lcb_runner/prompts/code_generation.py: PromptConstants and get_generic_question_template_answer
SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct "
          "Python program that matches the specification and passes all tests.")
WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
                   "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, "
                   "runs the algorithm and writes output to STDOUT.")


def prompt(it: dict) -> str:
    text = f"### Question:\n{it['question']}\n\n"
    if it["starter_code"]:
        text += f"### Format: {WITH_STARTER}\n```python\n{it['starter_code']}\n```\n\n"
    else:
        text += f"### Format: {WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    return text + "### Answer: (use the provided format with backticks)\n\n"


def extract_code(text: str) -> str:
    """lcb_runner/utils/extraction_utils.py extract_code: between the last two lines that hold ```."""
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[fences[-2] + 1:fences[-1]]) if len(fences) >= 2 else ""


class _PlainData(pickle.Unpickler):
    """The private tests are a pickled str: refuse anything that would import (pickle runs what it imports)."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"{module}.{name}: the private tests should be plain data")


def decode_tests(raw: str) -> list[dict]:
    """A problem's test cases: JSON (the public ones), else base64 of zlib of a pickled JSON string (the private ones)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_PlainData(io.BytesIO(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))).load())


def load(task: str) -> list[dict]:
    items = []
    for line in (ROOT / FILE).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        items.append({"id": r["question_id"], "group": r["difficulty"], "platform": r["platform"], "date": r["contest_date"][:10],
                      "question": r["question_content"], "starter_code": r["starter_code"],
                      "fn_name": json.loads(r["metadata"]).get("func_name"),
                      "public_tests": r["public_test_cases"], "private_tests": r["private_test_cases"]})
    return items


def grade(it: dict, code: str, timeout: int = TIMEOUT) -> dict:
    """Run `code` on the problem's tests in a subprocess (evals/lcb_exec.py). Never raises: a problem the harness cannot
    run grades as wrong, with the reason in "error"."""
    t0 = time.perf_counter()
    try:
        cases = decode_tests(it["public_tests"]) + decode_tests(it["private_tests"])
        cap = (timeout + 1) * len(cases) + 5  # lcb_runner's check_correctness: the whole problem, past the per-test alarms
        results, meta = [-1] * len(cases), {"error_message": "global timeout"}
        if not code.strip():
            results, meta = [-4], {"error_message": "no code block"}
        else:
            sample = {"code": code, "inputs": [c["input"] for c in cases], "outputs": [c["output"] for c in cases],
                      "fn_name": it["fn_name"], "timeout": timeout, "memory": MEMORY, "cpu": cap + 1}
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
                try:
                    subprocess.run([sys.executable, "-I", str(EXEC), "sample.json", "result.json"], cwd=d, timeout=cap,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}, start_new_session=True)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    meta = {"error_message": "the test process ended without a result (memory or CPU limit, or a crash)"}
                out = Path(d) / "result.json"
                if out.is_file() and out.stat().st_size:
                    done = json.loads(out.read_text(encoding="utf-8"))
                    results, meta = done["results"], done["metadata"]
        # lcb_runner's pass@1: np.all(results > 0), the results stopping at the first failure
        correct = bool(results) and all(r > 0 for r in results)
        passed = next((k for k, r in enumerate(results) if r is not True), len(results))
        error = "" if correct else str(meta.get("error_message") or meta.get("error") or "")[:300]
        return {"correct": correct, "tests": len(cases), "passed": passed, "error": error, "grade_seconds": time.perf_counter() - t0}
    except Exception as exc:  # the harness, not the program: still one wrong item, not a failed run
        return {"correct": False, "tests": 0, "passed": 0, "error": f"grading failed: {exc!r}"[:300], "grade_seconds": time.perf_counter() - t0}


def run(task: str, url: str, items: list[dict], accept, max_tokens: int, concurrency: int) -> list[dict]:
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, prompt(it), max_tokens, accept, system=SYSTEM)
        row = grade(it, extract_code(text))
        thinking = {k: usage[k] for k in ("reasoning_tokens", "reasoning_tail") if usage.get(k)}
        return {"index": i, "id": it["id"], "group": it["group"], "platform": it["platform"], **row, "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens"),
                "capped": (usage.get("completion_tokens") or 0) >= max_tokens, "text_tail": text[-160:], **thinking}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))
