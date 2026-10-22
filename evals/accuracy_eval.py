"""Task-accuracy benchmarks against a running server: GSM8K and MGSM (exact match on the final
number) and HumanEval and MBPP (pass@1, tests executed).

    sharar-eval accuracy --task gsm8k --limit 300 --label exact --accept 1:0 --output DIR
    sharar-eval accuracy --task humaneval --label w3 --output DIR
    sharar-eval accuracy --task mgsm --per-language 24 --label fast --accept server --output DIR
    sharar-eval accuracy compare DIR/exact-gsm8k.json DIR/w3-gsm8k.json

MGSM (multilingual grade-school math, 11 languages x 250, the url-nlp TSVs) asks
for the reasoning in the question's language; `--per-language N` takes the first
N items of every language. MBPP is the 500-task test split (task ids 11-510) of
the original release, graded on its three `test_list` asserts.

Greedy with thinking disabled by default (`--thinking` turns it on, `--reasoning-effort` sets the
template's level, xhigh by default; `--temperature T` samples with top_p 0.95, top_k 20 and one seed
per item shared by every arm, `--seed S` another set of seeds), identical prompts for every arm;
`--accept K:M` pins the acceptance rule per request (1:0 = exact, 2:1.0 = the
relaxed default; `server` sends no acceptance fields, so the server's profile
default applies). HumanEval and MBPP
completions run in a subprocess with no network, a CPU and memory limit and a
timeout. Results hold per-item correctness so two arms compare paired (McNemar).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import zlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.datasets import ROOT

DATA = ROOT / "accuracy"


def post(url: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


SAMPLING = {"temperature": 0.0}  # --temperature T samples with top_p 0.95, top_k 20 (the checkpoint's generation config)
SERVER = {"model": "sharar", "plain": False}  # --model / --plain: another OpenAI-compatible server (no Sharar-only fields)
# --thinking / --reasoning-effort (None: the template's default, xhigh on Qwen3.8) / --seed (the per-item seeds' offset)
THINKING = {"thinking": False, "reasoning_effort": None, "seed": 0}
EFFORTS = ("xhigh", "medium", "low")


def request_mode() -> dict:
    """The sampling and thinking settings every request of a run shares (recorded in run.json and result files)."""
    return {**SAMPLING, **THINKING}


def add_request_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sample at this temperature (top_p 0.95, top_k 20, one seed per prompt shared by every arm)")
    ap.add_argument("--seed", type=int, default=0, help="with --temperature: another seed per prompt (repeat a sampled run)")
    ap.add_argument("--thinking", action="store_true", help="thinking on (the model's default; off unless given)")
    ap.add_argument("--reasoning-effort", choices=EFFORTS, help="with --thinking: the template's reasoning effort (default xhigh)")


def apply_request_args(ap: argparse.ArgumentParser, args) -> None:
    if args.seed and args.temperature <= 0:
        ap.error("--seed needs --temperature")
    if args.reasoning_effort and not args.thinking:
        ap.error("--reasoning-effort needs --thinking")
    if args.temperature > 0:
        SAMPLING.update(temperature=args.temperature, top_p=0.95, top_k=20)
    THINKING.update(thinking=args.thinking, reasoning_effort=args.reasoning_effort, seed=args.seed)


def ask(url: str, content: str, max_tokens: int, accept: tuple[int, float]) -> tuple[str, dict]:
    """The reply (thinking removed) and the usage, with the thinking's last 400 characters as usage["reasoning_tail"]."""
    kwargs = {"enable_thinking": THINKING["thinking"]}
    if THINKING["reasoning_effort"]:
        kwargs["reasoning_effort"] = THINKING["reasoning_effort"]
    body = {"model": SERVER["model"], "messages": [{"role": "user", "content": content}], **SAMPLING, "max_tokens": max_tokens,
            "chat_template_kwargs": kwargs}
    if not SERVER["plain"]:
        body["trace"] = False
        if accept is not None:  # None: the server's default acceptance (its profile)
            body.update(accept_top_k=accept[0], accept_margin=accept[1])
    if body["temperature"] > 0:
        body["seed"] = zlib.crc32(content.encode(), THINKING["seed"])  # the same seed per item in every arm; --seed 0 as before
    out = post(url, body, timeout=max(900, max_tokens / 10))  # a long cap (run --max-tokens): 10 tokens a second at the least
    choice, usage = out["choices"][0], dict(out.get("usage") or {})
    text, reasoning = choice["message"].get("content") or "", choice["message"].get("reasoning_content") or ""
    if "</think>" in text:  # a server without a reasoning parser: the thinking arrives in the content
        reasoning, text = text.rsplit("</think>", 1)
        text = text.lstrip()
    elif THINKING["thinking"] and not reasoning and choice.get("finish_reason") == "length":
        reasoning, text = text, ""  # and capped inside it: no reply
    if reasoning:
        usage["reasoning_tail"] = reasoning[-400:]
    return text, usage


# ---------------------------------------------------------------- GSM8K

GSM_SUFFIX = "\n\nSolve the problem step by step. On the last line write only: Answer: <number>"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm_pred(text: str) -> str | None:
    tail = text.rsplit("Answer:", 1)
    src = tail[1] if len(tail) == 2 else text
    nums = NUM.findall(src)
    if not nums:
        nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return a == b


def run_gsm8k(url, items, accept, max_tokens, concurrency):
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, it["question"] + GSM_SUFFIX, max_tokens, accept)
        pred, gold = gsm_pred(text), gsm_gold(it["answer"])
        return {"index": i, "correct": same_number(pred, gold), "pred": pred, "gold": gold, "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens"), "text_tail": text[-160:]}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- MGSM

MGSM_LANGS = ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")
MGSM_SUFFIX = ("\n\nSolve the problem step by step, reasoning in the language of the question. "
               "On the last line write only: Answer: <number> (digits only, no thousands separators)")


def mgsm_items(per_language: int = 0) -> list[dict]:
    items = []
    for lang in MGSM_LANGS:
        rows = [l.split("\t") for l in (DATA / "mgsm" / f"mgsm_{lang}.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = rows[:per_language] if per_language else rows
        items += [{"lang": lang, "index": i, "question": q, "answer": a.strip()} for i, (q, a) in enumerate(rows)]
    return items


def run_mgsm(url, items, accept, max_tokens, concurrency):
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, it["question"] + MGSM_SUFFIX, max_tokens, accept)
        pred = gsm_pred(text)
        return {"index": i, "lang": it["lang"], "correct": same_number(pred, it["answer"]), "pred": pred, "gold": it["answer"],
                "seconds": time.perf_counter() - t0, "completion_tokens": usage.get("completion_tokens"),
                "relaxed_tokens": usage.get("relaxed_tokens"), "text_tail": text[-160:]}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- HumanEval

HE_PREFIX = ("Complete the following Python function. Reply with the complete function (including the signature and any imports it needs) "
             "in a single ```python code block and nothing else.\n\n")
BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def he_program(prompt: str, completion: str, entry_point: str) -> str:
    blocks = BLOCK.findall(completion)
    code = max(blocks, key=len) if blocks else completion
    if f"def {entry_point}" not in code:
        code = prompt + code  # the model returned a body only
    return code


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    os.setsid()


def he_check(program: str, test: str, entry_point: str | None) -> tuple[bool, str]:
    """Run program + test in a sandboxed subprocess; HumanEval calls check(entry_point), MBPP's asserts run as they stand."""
    src = program + "\n\n" + test + (f"\n\ncheck({entry_point})\n" if entry_point else "\n")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "candidate.py"
        path.write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", "-S", str(path)], cwd=d, capture_output=True, text=True, timeout=15,
                               env={"PATH": "/usr/bin:/bin"}, preexec_fn=_limits)
            return r.returncode == 0, (r.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def run_humaneval(url, items, accept, max_tokens, concurrency):
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, HE_PREFIX + it["prompt"], max_tokens, accept)
        program = he_program(it["prompt"], text, it["entry_point"])
        ok, err = he_check(program, it["test"], it["entry_point"])
        return {"index": i, "task_id": it["task_id"], "correct": ok, "error": err if not ok else "", "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens")}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- MBPP

MBPP_PREFIX = "You are an expert Python programmer, and here is your task: "
MBPP_SUFFIX = ("\nYour code should pass these tests:\n\n{tests}\n\nReply with the complete code "
               "(the function and any imports it needs) in a single ```python code block and nothing else.")


def mbpp_items() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "mbpp.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if 11 <= r["task_id"] <= 510]  # the test split


def run_mbpp(url, items, accept, max_tokens, concurrency):
    def one(i, it):
        t0 = time.perf_counter()
        tests = "\n".join(it["test_list"])
        text, usage = ask(url, MBPP_PREFIX + it["text"] + MBPP_SUFFIX.format(tests=tests), max_tokens, accept)
        blocks = BLOCK.findall(text)
        program = max(blocks, key=len) if blocks else text
        ok, err = he_check((it.get("test_setup_code") or "") + "\n" + program, tests, None)
        return {"index": i, "task_id": it["task_id"], "correct": ok, "error": err if not ok else "", "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens")}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- tasks

TASKS = {  # task: (dataset files, default max_tokens, runner)
    "gsm8k": (["accuracy/gsm8k_test.jsonl"], 640, run_gsm8k),
    "humaneval": (["accuracy/HumanEval.jsonl"], 768, run_humaneval),
    "mbpp": (["accuracy/mbpp.jsonl"], 768, run_mbpp),
    "mgsm": ([f"accuracy/mgsm/mgsm_{lang}.tsv" for lang in MGSM_LANGS], 768, run_mgsm),
}


def load(task: str, per_language: int = 0) -> list[dict]:
    if task == "mgsm":
        return mgsm_items(per_language)
    if task == "mbpp":
        return mbpp_items()
    return [json.loads(l) for l in (ROOT / TASKS[task][0][0]).read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- compare

def mcnemar(a: list[bool], b: list[bool]) -> dict:
    n01 = sum(1 for x, y in zip(a, b) if x and not y)  # a right, b wrong
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    p = min(1.0, sum(math.comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(n01 - n / 2)) / 2 ** n) if n else 1.0
    return {"a_only": n01, "b_only": n10, "p": p}


def compare(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    base = runs[0]
    print(f"{'arm':<12} {'task':<10} {'n':>4} {'acc':>7} {'vs first: +/-':>14} {'p':>6} {'tok/item':>9} {'relaxed%':>9}")
    for r in runs:
        acc = sum(x["correct"] for x in r["items"]) / len(r["items"])
        m = mcnemar([x["correct"] for x in base["items"]], [x["correct"] for x in r["items"]]) if r is not base else {"a_only": 0, "b_only": 0, "p": 1.0}
        toks = sum(x["completion_tokens"] or 0 for x in r["items"]); rel = sum(x["relaxed_tokens"] or 0 for x in r["items"])
        print(f"{r['label']:<12} {r['task']:<10} {len(r['items']):>4} {100*acc:6.1f}% {m['b_only']:>6}/{m['a_only']:<7} {m['p']:6.2f} {toks/len(r['items']):9.0f} {100*rel/max(1,toks):8.2f}%")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--accept", default="1:0", help="K:M acceptance pinned on every request (1:0 exact); 'server': send none (the server's default)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N items (a shard of a longer run; with --limit)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-language", type=int, default=0, help="mgsm: the first N items of each language (0: all 250)")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="sharar")
    ap.add_argument("--plain", action="store_true", help="another OpenAI-compatible server: no accept/trace fields, no /get_server_info")
    add_request_args(ap)
    args = ap.parse_args()
    SERVER.update(model=args.model, plain=args.plain)
    apply_request_args(ap, args)
    if args.accept == "server":
        accept = None
    else:
        k, m = args.accept.split(":")
        accept = (int(k), float(m))
    items = load(args.task, args.per_language)
    run, max_tokens = TASKS[args.task][2], args.max_tokens or TASKS[args.task][1]
    items = items[args.start:]
    if args.limit:
        items = items[:args.limit]
    t0 = time.perf_counter()
    rows = run(args.url, items, accept, max_tokens, args.concurrency)
    wall = time.perf_counter() - t0
    acc = sum(r["correct"] for r in rows) / len(rows)
    info = {}
    if not args.plain:
        with urllib.request.urlopen(f"{args.url}/get_server_info", timeout=10) as r:
            info = json.loads(r.read())
    result = {"label": args.label, "task": args.task, "accept": accept if accept is not None else "server", "per_language": args.per_language or None, "start": args.start, "sampling": request_mode(), "n": len(rows), "accuracy": acc, "wall_seconds": wall,
              "max_tokens": max_tokens, "speculative": info.get("speculative"), "items": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}-{args.task}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"label": args.label, "task": args.task, "n": len(rows), "accuracy": round(acc, 4), "wall_s": round(wall, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
