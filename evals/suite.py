"""One command to evaluate a server: a named suite of tasks into one run directory, recorded with
the versions that produced it, and one paired scorecard between two run directories.

    python -m evals.suite run --url http://127.0.0.1:30004 --label fast --suite quick --out runs/fast
    python -m evals.suite compare runs/exact runs/fast
    python -m evals.suite run --label fast --tasks gpqa,mmlupro --limit 50 --out runs/fast   # any tasks, first N items

Suites (Flash-Next on the GB10, 4 streams):
    quick   GSM8K first 300, HumanEval 164, MGSM 24 per language (264), needles at 16K and 64K   ~1 h
    full    GSM8K 1319, HumanEval 164, MBPP 500, MGSM 2750                                      ~4 h
    hard    MMLU-Pro 1,400, GPQA Diamond 198, IFEval 541, WMT24++ 1,000                          4-6 h
    all     full + hard + needles

`--tasks a,b` runs those tasks instead of a suite (names: gsm8k, humaneval, mbpp, mgsm, mgsm-pl24,
mmlupro, gpqa, ifeval, wmt, needles) and `--limit N` takes the first N items of each; both work with
`--suite` too. A limited run is a prefix of the full one, so a later longer run only adds the rest.

A run directory holds one server configuration. Each task runs in shards of 100 items
(<task>/s00000.json, ...), so an interrupted run resumes where it stopped and `full` reuses
the GSM8K shards `quick` wrote. run.json records the label, the server's /get_server_info,
the engine version given with --engine, this repository's commit, the sha256 of every dataset
file read and each task's token cap; a later `run` into the same directory must match it.

`compare` pairs the two runs item by item on every task both have (on the items both have):
binary tasks with McNemar and a normal interval of the paired difference, WMT24++ with a paired
bootstrap of chrF++, needles as counts. Pooled is all binary tasks together.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from evals import accuracy_eval, datasets, hard_eval, ruler2
from evals.accuracy_eval import SERVER, mcnemar

REPO = Path(__file__).resolve().parent.parent
SHARD = 100
# task name: (module, module task, loader options)
TASKS = {
    "gsm8k": (accuracy_eval, "gsm8k", {}),
    "humaneval": (accuracy_eval, "humaneval", {}),
    "mbpp": (accuracy_eval, "mbpp", {}),
    "mgsm": (accuracy_eval, "mgsm", {}),
    "mgsm-pl24": (accuracy_eval, "mgsm", {"per_language": 24}),
    "mmlupro": (hard_eval, "mmlupro", {}),
    "gpqa": (hard_eval, "gpqa", {}),
    "ifeval": (hard_eval, "ifeval", {}),
    "wmt": (hard_eval, "wmt", {}),
    **{t: (ruler2, t, {}) for t in ruler2.TASKS},  # ruler2-32k, ruler2-64k, ruler2-128k
    "needles": (None, "needles", {"lengths": [16384, 65536], "depths": [0.9, 0.5, 0.1], "output_tokens": 64}),
}
FULL = [("gsm8k", 0), ("humaneval", 0), ("mbpp", 0), ("mgsm", 0)]
HARD = [("mmlupro", 0), ("gpqa", 0), ("ifeval", 0), ("wmt", 0)]
SUITES = {  # suite: [(task, first N items; 0 = all)]
    "quick": [("gsm8k", 300), ("humaneval", 0), ("mgsm-pl24", 0), ("needles", 0)],
    "full": FULL,
    "hard": HARD,
    "long": [("ruler2-32k", 300), ("ruler2-128k", 120)],
    "all": FULL + HARD + [("ruler2-32k", 300), ("ruler2-128k", 120), ("needles", 0)],
}
# /get_server_info fields that identify a configuration (not its counters)
SERVER_KEYS = ("engine", "target", "model_path", "revision", "w3_target_identity", "context_length", "max_running_requests",
               "speculative", "attention_backend", "gdn_backend", "kv_cache_dtype", "mamba_ssm_dtype", "config")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def task_files(name: str) -> list[str]:
    mod, task, _ = TASKS[name]
    if mod is None:
        from evals import long_context
        return long_context.FILES
    return mod.TASKS[task][0]


def task_items(name: str) -> list[dict]:
    mod, task, opts = TASKS[name]
    return mod.load(task, **opts)


def task_max_tokens(name: str) -> int | None:
    mod, task, _ = TASKS[name]
    return mod.TASKS[task][1] if mod else None


def run_shard(name: str, url: str, items: list[dict], accept, concurrency: int) -> list[dict]:
    mod, task, _ = TASKS[name]
    if mod is accuracy_eval:
        return mod.TASKS[task][2](url, items, accept, task_max_tokens(name), concurrency)
    return mod.run(task, url, items, accept, task_max_tokens(name), concurrency)


# ---------------------------------------------------------------- run

def server_info(url: str) -> dict:
    if SERVER["plain"]:
        return {}
    with urllib.request.urlopen(f"{url}/get_server_info", timeout=30) as r:
        info = json.loads(r.read())
    return {k: info.get(k) for k in SERVER_KEYS}


def record(out: Path, args, names: list[str]) -> dict:
    """Create or check run.json: the same server configuration, request settings and evals tree as before."""
    path = out / "run.json"
    current = {"label": args.label, "url": args.url, "model": args.model, "plain": args.plain, "accept": args.accept,
               "concurrency": args.concurrency, "server": server_info(args.url), "evals_tree": git("rev-parse", "HEAD:evals")}
    rec = json.loads(path.read_text()) if path.exists() else {**current, "engine": args.engine, "datasets": {}, "tasks": {}, "runs": []}
    differs = [k for k in current if k not in ("url", "concurrency") and rec.get(k) != current[k]]
    if differs:
        raise SystemExit(f"{out}: {', '.join(differs)} differ from the run recorded there; use another --out")
    if args.engine and rec.get("engine") and args.engine != rec["engine"]:
        raise SystemExit(f"{out}: recorded --engine {rec['engine']!r}, given {args.engine!r}")
    rec["engine"] = rec.get("engine") or args.engine
    for name in names:
        rec["datasets"].update(datasets.check(task_files(name)))
        rec["tasks"][name] = {"max_tokens": task_max_tokens(name), **TASKS[name][2]}
    rec["runs"].append({"suite": args.suite, "tasks": args.tasks, "limit": args.limit or None, "started": dt.datetime.now().isoformat(timespec="seconds"),
                        "evals_commit": git("rev-parse", "HEAD"), "evals_dirty": bool(git("status", "--porcelain", "--", "evals", "datasets.sha256"))})
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def run_needles(out: Path, url: str) -> None:
    from evals import long_context
    done = out / "needles" / "needles.jsonl"
    if done.exists():
        return
    log("needles")
    tmp = done.with_suffix(".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)
    long_context.run(url, tmp, **TASKS["needles"][2])
    tmp.rename(done)


def run_task(out: Path, name: str, limit: int, url: str, accept, concurrency: int) -> None:
    items = task_items(name)
    total = min(limit, len(items)) if limit else len(items)
    for start in range(0, total, SHARD):
        path = out / name / f"s{start:05d}.json"
        have = json.loads(path.read_text())["items"] if path.exists() else []  # a shorter --limit may have filled part of it
        first = start + len(have)
        want = items[first:min(start + SHARD, total)]
        if not want:
            continue
        t0 = time.monotonic()
        rows = run_shard(name, url, want, accept, concurrency)
        for r in rows:
            r["index"] += first
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".partial").write_text(json.dumps({"task": name, "start": start, "items": have + rows}, ensure_ascii=False))
        path.with_suffix(".partial").rename(path)
        key = "score" if "score" in rows[0] else "correct"
        mean = sum(float(r[key]) for r in rows) / len(rows)
        log(f"{name} {first}-{first + len(rows) - 1} of {len(items)}: {mean * (1 if key == 'score' else 100):.1f}"
            f"{'' if key == 'score' else '%'} in {time.monotonic() - t0:.0f}s")


def cmd_run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="evals.suite run")
    ap.add_argument("--url", default="http://127.0.0.1:30004")
    ap.add_argument("--label", required=True)
    ap.add_argument("--suite", choices=sorted(SUITES), help="a named suite (default quick, unless --tasks is given)")
    ap.add_argument("--tasks", help=f"comma-separated tasks instead of a suite: {','.join(TASKS)}")
    ap.add_argument("--limit", type=int, default=0, help="the first N items of each task (the needles ignore it)")
    ap.add_argument("--out", type=Path, required=True, help="run directory for this server configuration")
    ap.add_argument("--engine", default=None, help="engine version, recorded verbatim (image tag and commit, or a pinned digest)")
    ap.add_argument("--accept", default="server", help="K:M acceptance pinned on every request (1:0 exact); 'server' (default): the server's profile")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--skip", action="append", default=[], choices=sorted(TASKS), help="leave this task out (repeatable)")
    ap.add_argument("--model", default="sharar")
    ap.add_argument("--plain", action="store_true", help="another OpenAI-compatible server: no Sharar-only request fields, no /get_server_info")
    args = ap.parse_args(argv)
    args.url = args.url.rstrip("/")
    SERVER.update(model=args.model, plain=args.plain)
    accept = None if args.accept == "server" else (int(args.accept.split(":")[0]), float(args.accept.split(":")[1]))
    skip = set(args.skip) | ({"needles"} if args.plain else set())  # the needles send Sharar request fields
    if args.tasks:
        names = args.tasks.split(",")
        if set(names) - set(TASKS):
            ap.error(f"unknown tasks: {sorted(set(names) - set(TASKS))}")
        base = [(n, k) for n, k in SUITES[args.suite] if n in names] if args.suite else [(n, 0) for n in names]
    else:
        args.suite = args.suite or "quick"
        base = SUITES[args.suite]
    plan = [(n, min(k, args.limit) if k and args.limit else k or args.limit) for n, k in base if n not in skip]
    if not plan:
        ap.error("nothing to run")
    record(args.out, args, [n for n, _ in plan])
    t0 = time.monotonic()
    for name, limit in plan:
        if name == "needles":
            run_needles(args.out, args.url)
        else:
            run_task(args.out, name, limit, args.url, accept, args.concurrency)
    rec = json.loads((args.out / "run.json").read_text())
    rec["runs"][-1]["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (args.out / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    log(f"{args.suite or args.tasks} done in {(time.monotonic() - t0) / 60:.0f} min: {args.out}")
    return 0


# ---------------------------------------------------------------- compare

def load_items(run: Path, name: str) -> list[dict]:
    """The contiguous completed items of a task, in order."""
    items = []
    for path in sorted((run / name).glob("s*.json")):
        shard = json.loads(path.read_text())
        if shard["start"] != len(items):
            break
        items += shard["items"]
    return items


def paired_acc(base: list[bool], cand: list[bool]) -> dict:
    n = len(base)
    m = mcnemar(base, cand)
    wins, losses = m["b_only"], m["a_only"]
    diff = (wins - losses) / n
    se = math.sqrt(max(0.0, (wins + losses) / n - diff ** 2) / n)
    return {"n": n, "base": sum(base) / n, "cand": sum(cand) / n, "delta": 100 * diff,
            "ci95": [100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)], "wins": wins, "losses": losses, "p": m["p"]}


def group(item: dict) -> str:
    return item.get("group") or item.get("lang") or ""


def side_stats(items: list[dict], max_tokens: int | None) -> dict:
    toks = [x.get("completion_tokens") or 0 for x in items]
    return {"tokens_per_item": sum(toks) / len(items), "capped": sum(t >= max_tokens for t in toks) if max_tokens else None}


def needle_stats(run: Path) -> dict | None:
    f = run / "needles" / "needles.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    hits = [r for r in rows if r.get("kind") == "needle"]
    return {"correct": sum(r["correct"] for r in hits), "n": len(hits),
            "decode_tps": {f"{r['length']}/{r['mode']}": r["decode_tps"] for r in rows if r.get("kind") == "decode"}}


def scorecard(base: Path, cand: Path, tasks: list[str] | None = None) -> dict:
    rb, rc = (json.loads((d / "run.json").read_text()) for d in (base, cand))
    card = {"base": {"dir": str(base), "label": rb["label"], "engine": rb.get("engine")},
            "cand": {"dir": str(cand), "label": rc["label"], "engine": rc.get("engine")}, "tasks": {}}
    pb, pc = [], []
    for name in tasks or TASKS:
        if name == "needles":
            nb, nc = needle_stats(base), needle_stats(cand)
            if nb and nc:
                card["tasks"][name] = {"kind": "needles", "base": nb, "cand": nc}
            continue
        xb, xc = load_items(base, name), load_items(cand, name)
        n = min(len(xb), len(xc))
        if not n:
            continue
        xb, xc = xb[:n], xc[:n]
        assert all(a.get("id", a["index"]) == b.get("id", b["index"]) for a, b in zip(xb, xc)), name
        cap = (rb["tasks"].get(name) or {}).get("max_tokens")
        if "score" in xb[0]:
            s = hard_eval.paired_scores([x["score"] for x in xb], [x["score"] for x in xc])
            row = {"kind": "score", "n": n, "base": sum(x["score"] for x in xb) / n, "cand": sum(x["score"] for x in xc) / n, **s}
        else:
            row = {"kind": "binary", **paired_acc([x["correct"] for x in xb], [x["correct"] for x in xc])}
            pb += [x["correct"] for x in xb]
            pc += [x["correct"] for x in xc]
        groups = sorted({group(x) for x in xb} - {""})
        if len(groups) > 1:
            key = "score" if row["kind"] == "score" else "correct"
            row["groups"] = {g: [sum(float(x[key]) for x in xs if group(x) == g) / sum(group(x) == g for x in xs) for xs in (xb, xc)]
                             for g in groups}
        card["tasks"][name] = {**row, "base_side": side_stats(xb, cap), "cand_side": side_stats(xc, cap)}
    if pb:
        card["pooled"] = paired_acc(pb, pc)
    return card


def show(card: dict, detail: bool) -> None:
    b, c = card["base"], card["cand"]
    print(f"base {b['label']} ({b['engine'] or 'engine not recorded'})  vs  candidate {c['label']} ({c['engine'] or 'engine not recorded'})\n")
    print(f"| task | n | {b['label']} | {c['label']} | delta [95%] | better/worse | p | tokens/item | capped |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in card["tasks"].items():
        if r["kind"] == "needles":
            print(f"| needles | {r['base']['n']} | {r['base']['correct']}/{r['base']['n']} | {r['cand']['correct']}/{r['cand']['n']} | | | | | |")
            continue
        sb, sc = r["base_side"], r["cand_side"]
        cap = f"{sb['capped']} / {sc['capped']}" if sb["capped"] is not None else ""
        if r["kind"] == "binary":
            vals = f"{100 * r['base']:.1f}% | {100 * r['cand']:.1f}% | {r['delta']:+.1f} pp [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]"
        else:
            vals = f"{r['base']:.2f} | {r['cand']:.2f} | {r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
        print(f"| {name} | {r['n']} | {vals} | {r['wins']}/{r['losses']} | {r['p']:.3f} | {sb['tokens_per_item']:.0f} / {sc['tokens_per_item']:.0f} | {cap} |")
    if "pooled" in card:
        p = card["pooled"]
        print(f"| pooled (binary) | {p['n']} | {100 * p['base']:.1f}% | {100 * p['cand']:.1f}% | {p['delta']:+.1f} pp "
              f"[{p['ci95'][0]:+.1f}, {p['ci95'][1]:+.1f}] | {p['wins']}/{p['losses']} | {p['p']:.3f} | | |")
    if detail:
        for name, r in card["tasks"].items():
            if r.get("groups"):
                scale = 1 if r["kind"] == "score" else 100
                print(f"\n{name} by group ({b['label']}, delta):")
                for g, (vb, vc) in r["groups"].items():
                    print(f"  {g:<28} {scale * vb:7.2f}  {scale * (vc - vb):+6.2f}")


def cmd_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="evals.suite compare")
    ap.add_argument("base", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--json", type=Path, help="also write the scorecard as JSON")
    ap.add_argument("--detail", action="store_true", help="per category / language")
    ap.add_argument("--tasks", help="comma-separated tasks to compare (default: every task both runs have)")
    args = ap.parse_args(argv)
    tasks = args.tasks.split(",") if args.tasks else None
    if tasks and set(tasks) - set(TASKS):
        ap.error(f"unknown tasks: {sorted(set(tasks) - set(TASKS))}")
    card = scorecard(args.base, args.cand, tasks)
    show(card, args.detail)
    if args.json:
        args.json.write_text(json.dumps(card, indent=1) + "\n")
    return 0


def main() -> int:
    cmds = {"run": cmd_run, "compare": cmd_compare}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return 2
    return cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
