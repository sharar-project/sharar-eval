"""One LiveCodeBench program against its tests, run by evals.lcb as `python -I lcb_exec.py SAMPLE RESULT` in its own
subprocess. Ported from LiveCodeBench lcb_runner/evaluation/testing_util.py at 28fef95 (MIT): the same wrapping of a
stdin program into a function, the same mocked stdin, comparisons, per-test alarm and reliability guard, and the same
result codes (True / False per test until the first failure; -2 wrong answer, -3 time limit, -4 runtime error).
Differences: the guard also caps the address space (lcb_runner calls it without a limit) and CPU time; `help` is
cleared on the builtins module (this file runs as __main__, where __builtins__ is the module, not a dict).

SAMPLE is JSON {"inputs": [...], "outputs": [...], "fn_name": name or null, "timeout": s, "memory": bytes, "cpu": s};
RESULT gets {"results": [...], "metadata": {...}}. Standard library only (numpy is imported first when installed).
"""
import ast
import faulthandler
import json
import os
import signal
import sys
import time
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn(s, length=300):
    s = s if isinstance(s, str) else str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2:]


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Capturing(list):
    """stdout as a list (lcb_runner's, from https://stackoverflow.com/a/16571630/6416660)."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1  # closing it is a no-op
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    """A trailing `if __name__ == '__main__':` block becomes top-level code."""
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    """A stdin program as `wrapped_function()`: its imports stay at the top, everything else becomes the body."""
    try:
        import_stmts, all_other_stmts = [], []
        astree = ast.parse(code)
        for stmt in astree.body:
            (import_stmts if isinstance(stmt, (ast.Import, ast.ImportFrom)) else all_other_stmts).append(stmt)
        function_ast = ast.FunctionDef(name="wrapped_function", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                                       body=all_other_stmts, decorator_list=[], lineno=-1)
        return import_string + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(function_ast)
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        # LeetCode wraps solutions in `Solution`
        compiled_sol = tmp_sol.Solution() if "class Solution" in code else tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    val = val.strip()  # no empty list for a trailing newline
    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int):
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return
    method = get_function(compiled_sol, fn_name)
    if method is None:
        return
    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)
            if isinstance(prediction, tuple):  # don't penalize tuples: the ground truth sequences are lists
                prediction = list(prediction)
            tmp_result = prediction == gt_out
            all_results.append(tmp_result)
            if not tmp_result:
                return all_results, {"output": truncatefn(prediction), "inputs": truncatefn(gt_inp), "expected": truncatefn(gt_out),
                                     "error_code": -2, "error_message": "Wrong Answer"}
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {"error": repr(e), "error_code": -3, "error_message": "Time Limit Exceeded",
                                     "inputs": truncatefn(gt_inp), "expected": truncatefn(gt_out)}
            all_results.append(-4)
            return all_results, {"error": repr(e), "error_code": -4, "error_message": "Runtime Error",
                                 "inputs": truncatefn(gt_inp), "expected": truncatefn(gt_out)}
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return all_results, {"execution time": total_execution}


def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    code = make_function(clean_if_name(code))
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return
    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {"error": repr(e), "error_code": -3, "error_message": "Time Limit Exceeded",
                                         "inputs": truncatefn(gt_inp), "expected": truncatefn(gt_out)}
                all_results.append(-4)
                return all_results, {"error": repr(e), "error_code": -4, "error_message": "Runtime Error",
                                     "inputs": truncatefn(gt_inp), "expected": truncatefn(gt_out)}
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)
        WA_send_args = {"output": truncatefn(prediction), "inputs": truncatefn(gt_inp), "expected": truncatefn(gt_out), "error_code": -2}
        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args
        for output_line_idx, (pred_line, gt_line) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = f"Wrong answer at {output_line_idx=}: {truncatefn(pred_line)} != {truncatefn(gt_line)}"
            if pred_line == gt_line:  # exact match
                continue
            # element-wise, as decimals (np.isclose(50000000000000000, 50000000000000001) is True; Decimal compares exactly)
            success, decimal_prediction_line = convert_line_to_decimals(pred_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            success, decimal_gtout_line = convert_line_to_decimals(gt_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            if decimal_prediction_line == decimal_gtout_line:
                continue
            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)
    return all_results, {"execution time": total_execution_time}


def run_test(sample: dict, test: str, timeout: int = 6):
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard(sample.get("memory"), sample.get("cpu"))
    grade = grade_stdio if sample.get("fn_name") is None else grade_call_based
    kwargs = {} if sample.get("fn_name") is None else {"fn_name": sample["fn_name"]}
    signal.alarm(timeout)
    try:
        results, metadata = grade(code=test, all_inputs=sample["inputs"], all_outputs=sample["outputs"], timeout=timeout, **kwargs)
        return results, metadata  # (unpacked here: a program without the function returns None, a runtime error)
    except Exception as e:
        return [-4], {"error_code": -4, "error_message": f"Error during testing: {e}"}
    finally:
        signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None, maximum_cpu_seconds=None):
    """lcb_runner's: disables functions that could change the test (fork, kill, remove files, ...). Not a sandbox."""
    import resource
    if maximum_memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    if maximum_cpu_seconds is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (maximum_cpu_seconds, maximum_cpu_seconds))
    faulthandler.disable()

    import builtins
    builtins.quit = None
    builtins.help = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty", "killpg",
                 "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot", "lchflags",
                 "lchmod", "lchown", "getcwd", "chdir"):
        setattr(os, name, None)
    import shutil
    shutil.rmtree = shutil.move = shutil.chown = None
    import subprocess
    subprocess.Popen = None
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None


def main() -> None:
    sys.set_int_max_str_digits(50000)  # lcb_runner's compute_code_generation_metrics sets it for its workers
    try:  # lcb_runner's testing_util imports numpy before the guard, so programs can import it (after it, importing fails)
        import numpy  # noqa: F401
    except ImportError:
        pass
    with open(sys.argv[1], encoding="utf-8") as f:
        sample = json.load(f)
    out = open(sys.argv[2], "w", encoding="utf-8")  # opened before the guard and the program run
    try:
        results, metadata = run_test(sample, sample["code"], timeout=sample["timeout"])
    except BaseException as e:  # SystemExit from a functional solution, for one: lcb_runner's worker dies with no result
        results, metadata = [-1] * len(sample["inputs"]), {"error_code": -1, "error_message": f"no result: {e!r}"}
    fixed = []
    for r in results:  # a functional solution's numpy scalar compares to a numpy bool (lcb_runner fixes them the same way)
        if not isinstance(r, (bool, int)):
            try:
                r = bool(r)
            except Exception:
                r = False
        fixed.append(r)
    out.write(json.dumps({"results": fixed, "metadata": metadata}, default=str))
    out.close()
    os._exit(0)  # threads the program left running do not hold the process


if __name__ == "__main__":
    main()
