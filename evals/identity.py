"""Exact greedy output identity between two servers (the refactor gate, sharar#24).

Runs the h2h decode prompts (thinking off and on) greedily with exact acceptance
(`accept_top_k: 1`, so speculation cannot change the continuation) and records
the full text per prompt; `compare` reports identical prompts and the first
differing character.

    python -m evals.identity run --url http://127.0.0.1:30004 --out a.json
    python -m evals.identity compare a.json b.json
"""
import argparse
import json
import sys
from pathlib import Path

from evals.h2h import SPEC, body, stream


def run(args):
    spec = json.loads(SPEC.read_text())
    ns = argparse.Namespace(model=args.model, seed=None, extra={"accept_top_k": 1, "trace": False})
    rows = {}
    for thinking in (False, True):
        for p in spec["decode"]:
            b = body(ns, spec, p["prompt"], "greedy", thinking, max_tokens=args.max_tokens)
            r = stream(args.url, b, args.api_key)
            rows[f'{p["id"]}/{"think" if thinking else "plain"}'] = {"text": r["text"], "tokens": r["completion_tokens"]}
            print(p["id"], thinking, r["completion_tokens"], flush=True)
    Path(args.out).write_text(json.dumps({"url": args.url, "max_tokens": args.max_tokens, "rows": rows}, indent=1, ensure_ascii=False))


def compare(args):
    a, b = (json.loads(Path(p).read_text())["rows"] for p in (args.a, args.b))
    same, diff = 0, []
    for key in sorted(set(a) | set(b)):
        ta, tb = a.get(key, {}).get("text"), b.get(key, {}).get("text")
        if ta == tb:
            same += 1
        else:
            first = next((i for i, (x, y) in enumerate(zip(ta or "", tb or "")) if x != y), min(len(ta or ""), len(tb or "")))
            diff.append((key, first))
    print(f"identical {same}/{same + len(diff)}")
    for key, first in diff:
        print(f"  {key}: first difference at character {first}")
    return 0 if not diff else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="sharar")
    r.add_argument("--api-key", default=None)
    r.add_argument("--max-tokens", type=int, default=384)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    sys.exit(run(args) if args.cmd == "run" else compare(args))


if __name__ == "__main__":
    main()
