"""Terminal-Bench 2.1 through Harbor against a running server (sharar#28): agentic tasks in a terminal,
an agent driving the server, pass/fail per task from each task's tests.

    sharar-eval tbench oracle --out DIR      # once per box: build every task image, run the reference solutions
    sharar-eval run --suite agentic --label fast --out runs/fast        # every valid task, two attempts
    sharar-eval run --tasks tbench-claude-code,tbench-opencode --limit 78 --label fast --out runs/fast

- **Tasks.** Terminal-Bench 2.1 (89 tasks; harbor `terminal-bench/terminal-bench-2-1`, pinned by the
  hash of its file manifest). Version 2.1, not 4.0: 4.0's tasks are tuned for frontier models with
  8-hour agent timeouts, so a local model would sit at the floor.
- **The GB10 is aarch64.** The published task images are amd64 only, so every run builds each task's
  image from its Dockerfile (`--force-build`; Docker's layer cache makes repeats cheap). `oracle` builds
  them all and runs each task's reference solution. Only tasks the oracle passes here are evaluated;
  they are listed in evals/workloads/tbench_valid.json.
- **Agents.** Harbor 0.23.0 runs them, installed on its own: `uv tool install harbor==0.23.0`.
  - `tbench`: Terminus-2, model `hosted_vllm/<--model>` at `<url>/v1`. It asks for commands in plain-text JSON,
    so the server's tool-call parser is never used. Requests are greedy (temperature 0) with thinking off, as
    in the other evals. The acceptance rule is the server's own unless `--accept` pins one.
  - `tbench-claude-code` and `tbench-opencode`: the agents people use, with native tool calls. Claude Code
    talks Anthropic Messages (`<url>/v1/messages`) and OpenCode talks OpenAI chat with tools (`<url>/v1`),
    set up as `sharar-ctl launch claude` and `sharar-ctl launch opencode` set them up. Harbor installs the
    pinned versions (AGENT_VERSIONS) in each task container. Sampling and thinking are the agent's own,
    and the acceptance rule is the server's (these agents cannot send `--accept`). Claude Code needs a
    model name containing "claude": against Sharar it is claude-sharar-qwen3.8, and with `--plain` it is `--model`.
  - Each native-agent task's traffic goes through a recording proxy (evals/agent_traffic.py) on Docker's
    bridge, into <run dir>/<task>/traffic/<item>/. Its tool-call check is in each item (`traffic`): leaks,
    bad JSON, unknown tools, missing parameters, stop reasons and HTTP errors, which should all be 0.
    `sharar-eval agent-traffic check <run dir>/<task>/traffic --show 5` prints examples.
- **Items.** One item per (task, attempt), attempt-major. `--limit 89` is one attempt of every task.
  A task's run lives in <run dir>/<tbench task>/jobs/<task>#<attempt>/, so an interrupted run resumes per task.
- **Concurrency.** Two tasks at a time at most. Each task container takes up to a few GiB, and
  Flash-Next leaves about 27 GiB free.
- **On the GB10** (2026-10-02): 78 of 89 tasks pass the oracle; the other 11 fail their reference
  solution or image build on aarch64, twice. Flash-Next `fast` took 20 min for 3 tasks at 2 at a time
  (solved 1), so the full suite (156 items) is about 18 h per arm.
- **Native agents on the GB10** (2026-10-07, 22 tasks, one attempt, `fast`): Claude Code solved 14 on
  the 27B and 12 on Flash-Next, OpenCode 13 and 12. 22 of the 37 misses were timeouts. Across 1,618 requests
  there were 2 leaks: Flash-Next ended its turn inside thinking with a tool call in it (sharar#120). The 2 HTTP
  errors per Claude Code run were chess-best-move's board image, which a text-only server rejects (sharar#121).
  Evidence: sharar-results reviews/2026-10-07-tbench-agents.
- **Power.** About 78 pass/fail tasks per attempt can only detect differences of about 10 points.
  Treat this as an agentic check, not a fine gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals import agent_traffic
from evals.accuracy_eval import SERVER
from evals.datasets import ROOT

REPO = Path(__file__).resolve().parent.parent
HARBOR_VERSION = "0.23.0"


def _harbor() -> Path:
    """$SHARAR_HARBOR, else harbor on PATH (uv tool install), else an older .venv-harbor here or in this checkout."""
    if configured := os.environ.get("SHARAR_HARBOR"):
        return Path(configured).expanduser()
    if found := shutil.which("harbor"):
        return Path(found)
    for venv in (Path.cwd() / ".venv-harbor", REPO / ".venv-harbor"):
        if (venv / "bin" / "harbor").is_file():
            return venv / "bin" / "harbor"
    return Path("harbor")


HARBOR = _harbor()
REGISTRY_NAME = "terminal-bench/terminal-bench-2-1"
DATASET = ROOT / "tbench" / "terminal-bench-2-1"
MANIFEST = "tbench/terminal-bench-2-1.sha256"
VALID = Path(__file__).resolve().parent / "workloads" / "tbench_valid.json"
ATTEMPTS, CONCURRENCY = 2, 2
AGENTS = {"tbench": "terminus-2", "tbench-claude-code": "claude-code", "tbench-opencode": "opencode"}
TASKS = {t: ([MANIFEST], None) for t in AGENTS}
AGENT_VERSIONS = {"claude-code": "2.1.292", "opencode": "1.18.35"}  # the versions measured on 2026-10-07
CLAUDE_MODEL = "claude-sharar-qwen3.8"  # Claude Code only accepts model names containing "claude"
OUTPUT_TOKENS = 32768  # OpenCode's output limit, as sharar-ctl launch opencode sets it
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
        raise SystemExit(f"harbor not found: uv tool install harbor=={HARBOR_VERSION} (or set SHARAR_HARBOR)")
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
        raise SystemExit(f"{VALID} missing: run `sharar-eval tbench oracle --out DIR` once on this box")
    valid = json.loads(VALID.read_text())["valid"]
    return [{"id": f"{t}#{a}", "task": t, "attempt": a, "group": category(t)} for a in range(ATTEMPTS) for t in valid]


def context_length(url: str) -> int:
    if SERVER["plain"]:
        return 131072
    with urllib.request.urlopen(f"{url}/get_server_info", timeout=30) as r:
        return json.loads(r.read()).get("context_length") or 131072


def bridge_address() -> str:
    """The host's address on Docker's default bridge: task containers reach the recording proxies there."""
    out = subprocess.run(["docker", "network", "inspect", "bridge", "-f", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"],
                         capture_output=True, text=True).stdout.strip()
    return out or "172.17.0.1"


def agent_args(url: str, accept) -> list[str]:
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    if not SERVER["plain"]:
        body["trace"] = False
        if accept is not None:
            body.update(accept_top_k=accept[0], accept_margin=accept[1])
    info = {"max_input_tokens": context_length(url), "max_output_tokens": 16384, "input_cost_per_token": 0, "output_cost_per_token": 0}
    return ["-a", "terminus-2", "-m", f"hosted_vllm/{SERVER['model']}", "--ak", f"api_base={url}/v1", "--ak", "temperature=0",
            "--ak", f"model_info={json.dumps(info)}", "--ak", f"llm_call_kwargs={json.dumps({'extra_body': body})}"]


def native_agent_args(agent: str, url: str, context: int) -> list[str]:
    """Claude Code or OpenCode against `url` (a recording proxy), configured as sharar-ctl launch configures them."""
    version = ["--ak", f"version={AGENT_VERSIONS[agent]}"]
    if agent == "claude-code":
        model = SERVER["model"] if SERVER["plain"] else CLAUDE_MODEL
        return ["-a", "claude-code", "-m", model, *version, "--ae", f"ANTHROPIC_BASE_URL={url}", "--ae", "ANTHROPIC_API_KEY=sharar",
                "--ae", f"CLAUDE_CODE_MAX_CONTEXT_TOKENS={context}", "--ae", "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1",
                "--ae", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]
    model = SERVER["model"]
    provider = {"npm": "@ai-sdk/openai-compatible", "name": "Sharar", "options": {"baseURL": f"{url}/v1", "apiKey": "sharar"},
                "models": {model: {"name": model, "limit": {"context": context, "output": OUTPUT_TOKENS}}}}
    return ["-a", "opencode", "-m", f"sharar/{model}", *version, "--ak", "opencode_config=" + json.dumps({"provider": {"sharar": provider}})]


def run(task: str, url: str, items: list[dict], accept, max_tokens, concurrency: int, workdir: Path) -> list[dict]:
    jobs = workdir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    agent = AGENTS[task]
    if agent == "terminus-2":
        args = agent_args(url, accept)
    elif accept is not None:
        raise SystemExit(f"{task}: {agent} cannot send a per-request acceptance rule; run it with --accept server")
    else:
        bridge, context = bridge_address(), context_length(url)

    def one(i, it):
        job = jobs / it["id"]
        traffic = workdir / "traffic" / it["id"]
        t0 = time.perf_counter()
        r = trial(job)
        if r is None:  # not run, or interrupted: start this task over
            shutil.rmtree(job, ignore_errors=True)
            run_args = ["-i", it["task"], "-n", "1", "-o", str(jobs), "--job-name", it["id"]]
            if agent == "terminus-2":
                harbor([*args, *run_args], workdir / "harbor.log")
            else:
                shutil.rmtree(traffic, ignore_errors=True)
                proxy = agent_traffic.Recorder(bridge, 0, url, traffic)
                threading.Thread(target=proxy.serve_forever, daemon=True).start()
                try:
                    proxy_url = f"http://{bridge}:{proxy.server_address[1]}"
                    harbor([*native_agent_args(agent, proxy_url, context), *run_args], workdir / "harbor.log")
                finally:
                    proxy.shutdown()
                    proxy.server_close()
            r = trial(job) or {}
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        usage = r.get("agent_result") or {}
        row = {"index": i, "id": it["id"], "group": it["group"], "correct": reward == 1.0, "reward": reward,
               "exception": (r.get("exception_info") or {}).get("exception_type"), "seconds": time.perf_counter() - t0,
               "completion_tokens": usage.get("n_output_tokens"), "prompt_tokens": usage.get("n_input_tokens")}
        if agent != "terminus-2":
            row.update(agent_version=(r.get("agent_info") or {}).get("version"), traffic=agent_traffic.check(traffic)["counts"])
        return row

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
