"""Terminal-Bench 2.1 through Harbor against a running server (sharar#28): agentic tasks in a terminal,
the Terminus-2 agent driving the server's OpenAI endpoint, pass/fail per task from each task's tests.

    python -m evals.tbench oracle --out DIR      # once per box: build every task image, run the reference solutions
    python -m evals.suite run --suite agentic --label fast --out runs/fast        # every valid task, two attempts

- **Tasks.** Terminal-Bench 2.1 (89 tasks; harbor `terminal-bench/terminal-bench-2-1`, pinned by the
  hash of its file manifest). Version 2.1, not 4.0: 4.0's tasks are tuned for frontier models with
  8-hour agent timeouts, so a local model would sit at the floor.
- **The GB10 is aarch64.** The published task images are amd64 only, so every run builds each task's
  image from its Dockerfile (`--force-build`; Docker's layer cache makes repeats cheap). `oracle` builds
  them all and runs each task's reference solution. Only tasks the oracle passes here are evaluated;
  they are listed in evals/workloads/tbench_valid.json.
- **Agent.** Terminus-2 (harbor 0.23.0, in its own venv: `python3 -m venv .venv-harbor &&
  .venv-harbor/bin/pip install -r requirements-harbor.txt`), model `hosted_vllm/<--model>` at
  `<url>/v1`. Requests are greedy (temperature 0) with thinking off, as in the other evals. The
  acceptance rule is the server's own unless `--accept` pins one.
- **Items.** One item per (task, attempt), attempt-major. `--limit 89` is one attempt of every task.
  A task's run lives in <run dir>/tbench/jobs/<task>#<attempt>/, so an interrupted run resumes per task.
- **Concurrency.** Two tasks at a time at most. Each task container takes up to a few GiB, and
  Flash-Next leaves about 27 GiB free.
- **On the GB10** (2026-10-02): 78 of 89 tasks pass the oracle; the other 11 fail their reference
  solution or image build on aarch64, twice. Flash-Next `fast` took 20 min for 3 tasks at 2 at a time
  (solved 1), so the full suite (156 items) is about 18 h per arm.
- **Power.** About 78 pass/fail tasks per attempt can only detect differences of about 10 points.
  Treat this as an agentic check, not a fine gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.accuracy_eval import SERVER
from evals.datasets import ROOT

REPO = Path(__file__).resolve().parent.parent
HARBOR = REPO / ".venv-harbor" / "bin" / "harbor"
REGISTRY_NAME = "terminal-bench/terminal-bench-2-1"
DATASET = ROOT / "tbench" / "terminal-bench-2-1"
MANIFEST = "tbench/terminal-bench-2-1.sha256"
VALID = Path(__file__).resolve().parent / "workloads" / "tbench_valid.json"
ATTEMPTS, CONCURRENCY = 2, 2
TASKS = {"tbench": ([MANIFEST], None)}
WORKDIR = True  # evals.suite passes the task's run directory to run()


def manifest_text(root: Path = DATASET) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n" for p in files)


def fetch() -> None:
    """Download the dataset with harbor and write its file manifest (pinned in datasets.sha256)."""
    if not DATASET.is_dir():
        subprocess.run([str(HARBOR), "download", REGISTRY_NAME, "--export", "-o", str(DATASET.parent)], check=True)
    (ROOT / MANIFEST).write_text(manifest_text())


def harbor(args: list[str], log: Path) -> int:
    if not HARBOR.is_file():
        raise SystemExit(f"{HARBOR} not found: python3 -m venv .venv-harbor && .venv-harbor/bin/pip install -r requirements-harbor.txt")
    with log.open("a") as f:
        return subprocess.run([str(HARBOR), "run", "-p", str(DATASET), "--force-build", "-y", "-q", *args],
                              stdout=f, stderr=subprocess.STDOUT).returncode


def trial(job: Path) -> dict | None:
    """The single trial result of a harbor job directory, or None if it has not finished."""
    found = [p for p in job.glob("*/result.json")] if job.is_dir() else []
    return json.loads(found[0].read_text()) if found else None


def category(task: str) -> str:
    return tomllib.loads((DATASET / task / "task.toml").read_text())["metadata"].get("category", "")


# ---------------------------------------------------------------- oracle

def oracle(out: Path, concurrency: int) -> None:
    """Build every task image here and run the reference solutions; the tasks that pass are the eval set."""
    out.mkdir(parents=True, exist_ok=True)
    job = out / "oracle"
    if not (job / "result.json").is_file():
        harbor(["-a", "oracle", "-n", str(concurrency), "--environment-build-timeout-multiplier", "4", "-o", str(out),
                "--job-name", "oracle"], out / "oracle.log")
    results = {}
    for p in sorted(job.glob("*/result.json")):
        r = json.loads(p.read_text())
        task = r["task_name"].split("/", 1)[1]
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = (r.get("exception_info") or {}).get("exception_type")
        results[task] = "pass" if reward == 1.0 else (exc or f"reward {reward}")
    tasks = sorted(p.name for p in DATASET.iterdir() if (p / "task.toml").is_file())
    missing = [t for t in tasks if t not in results]
    valid = [t for t in tasks if results.get(t) == "pass"]
    VALID.write_text(json.dumps({
        "dataset": REGISTRY_NAME, "manifest_sha256": hashlib.sha256((ROOT / MANIFEST).read_bytes()).hexdigest(),
        "harbor": subprocess.run([str(HARBOR), "--version"], capture_output=True, text=True).stdout.strip(),
        "machine": platform.machine(), "date": dt.date.today().isoformat(), "valid": valid,
        "excluded": {t: results.get(t, "no result") for t in tasks if t not in valid}}, indent=1) + "\n")
    print(f"{len(valid)} of {len(tasks)} tasks pass the oracle here ({len(missing)} without a result); {VALID}")


# ---------------------------------------------------------------- suite interface

def load(task: str) -> list[dict]:
    if not VALID.is_file():
        raise SystemExit(f"{VALID} missing: run `python -m evals.tbench oracle --out DIR` once on this box")
    valid = json.loads(VALID.read_text())["valid"]
    return [{"id": f"{t}#{a}", "task": t, "attempt": a, "group": category(t)} for a in range(ATTEMPTS) for t in valid]


def agent_args(url: str, accept) -> list[str]:
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    if not SERVER["plain"]:
        body["trace"] = False
        if accept is not None:
            body.update(accept_top_k=accept[0], accept_margin=accept[1])
    context = 131072
    if not SERVER["plain"]:
        with urllib.request.urlopen(f"{url}/get_server_info", timeout=30) as r:
            context = json.loads(r.read()).get("context_length") or context
    info = {"max_input_tokens": context, "max_output_tokens": 16384, "input_cost_per_token": 0, "output_cost_per_token": 0}
    return ["-a", "terminus-2", "-m", f"hosted_vllm/{SERVER['model']}", "--ak", f"api_base={url}/v1", "--ak", "temperature=0",
            "--ak", f"model_info={json.dumps(info)}", "--ak", f"llm_call_kwargs={json.dumps({'extra_body': body})}"]


def run(task: str, url: str, items: list[dict], accept, max_tokens, concurrency: int, workdir: Path) -> list[dict]:
    jobs = workdir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    args = agent_args(url, accept)

    def one(i, it):
        job = jobs / it["id"]
        t0 = time.perf_counter()
        r = trial(job)
        if r is None:  # not run, or interrupted: start this task over
            shutil.rmtree(job, ignore_errors=True)
            harbor([*args, "-i", it["task"], "-n", "1", "-o", str(jobs), "--job-name", it["id"]], workdir / "harbor.log")
            r = trial(job) or {}
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        usage = r.get("agent_result") or {}
        return {"index": i, "id": it["id"], "group": it["group"], "correct": reward == 1.0, "reward": reward,
                "exception": (r.get("exception_info") or {}).get("exception_type"), "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("n_output_tokens"), "prompt_tokens": usage.get("n_input_tokens")}

    with ThreadPoolExecutor(max_workers=min(concurrency, CONCURRENCY)) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["oracle", "fetch"])
    ap.add_argument("--out", type=Path, help="oracle: job directory")
    ap.add_argument("--concurrency", type=int, default=4, help="oracle: tasks at a time (no model server needed)")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch()
    else:
        if not args.out:
            ap.error("oracle needs --out")
        oracle(args.out, args.concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
