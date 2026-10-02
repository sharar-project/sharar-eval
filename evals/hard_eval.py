"""Harder, less saturated task evaluations against a running server (sharar#28): MMLU-Pro (knowledge,
10 choices), GPQA Diamond (graduate science), IFEval (instruction following) and WMT24++ (English into
ten languages, chrF++). GSM8K and HumanEval sit at 93-95% for these models, so a small loss needs
thousands of items; these tasks sit lower, test knowledge, chat and multilingual generation, where
lossy weight changes move the distribution most, and give more discordant pairs per item.

    python -m evals.hard_eval --task mmlupro --label fast --output DIR
    python -m evals.hard_eval compare DIR/exact-mmlupro.json DIR/fast-mmlupro.json

Requests and sampling follow `evals.accuracy_eval` (greedy, thinking disabled, identical prompts in
every arm, `--accept` as there; the default here is `server`, the arm's profile decides). MMLU-Pro
takes the first `--per-category` items (default 100, 1,400 in all) of each of its 14 categories in
question-id order; GPQA shuffles each question's options with a seed from its text, the same in every
arm; IFEval is graded strict per prompt with the reference checker (evals/ifeval, google-research
instruction_following_eval, Apache-2.0); WMT24++ takes the first `--per-language` good-source segments
(default 100) of each language and scores sentence chrF++ (sacrebleu). Binary tasks compare paired with
McNemar, WMT24++ with a paired bootstrap of the mean chrF++ and a sign test.

Datasets: `python -m evals.datasets fetch` (GPQA is gated: HF_TOKEN).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.accuracy_eval import SAMPLING, SERVER, ask, mcnemar
from evals.datasets import ROOT, download as fetch_url

DATA = ROOT / "hard"
os.environ.setdefault("NLTK_DATA", str(DATA.parent / "nltk"))

SOURCES = {  # dataset repository, pinned revision, file(s)
    "mmlupro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", ["data/test-00000-of-00001.parquet"]),
    "gpqa": ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", ["gpqa_diamond.csv"]),  # gated: HF_TOKEN
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", ["ifeval_input_data.jsonl"]),
    "wmt": ("google/wmt24pp", "fd7405c06494bc66a57b25f55d217a72f96e60dc", None),  # one file per language, below
}
WMT_LANGS = {"de_DE": "German", "es_MX": "Spanish", "fr_FR": "French", "ru_RU": "Russian", "ja_JP": "Japanese",
             "zh_CN": "Simplified Chinese", "th_TH": "Thai", "bn_IN": "Bengali", "te_IN": "Telugu", "sw_KE": "Swahili"}
LETTERS = "ABCDEFGHIJ"
ANSWER = re.compile(r"(?:Answer|answer is)\s*[:：]?\s*\(?([A-J])\)?")


def download(repo: str, revision: str, path: str) -> bytes:
    return fetch_url(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}")


def fetch() -> None:
    """Write each task's items as JSONL under DATA and print their sha256 (datasets.sha256 pins them)."""
    import csv
    import pyarrow.parquet as pq
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    repo, rev, files = SOURCES["mmlupro"]
    out["mmlupro.jsonl"] = pq.read_table(io.BytesIO(download(repo, rev, files[0]))).to_pylist()
    repo, rev, files = SOURCES["gpqa"]
    out["gpqa_diamond.jsonl"] = list(csv.DictReader(io.StringIO(download(repo, rev, files[0]).decode("utf-8"))))
    repo, rev, files = SOURCES["ifeval"]
    out["ifeval.jsonl"] = [json.loads(l) for l in download(repo, rev, files[0]).decode().splitlines() if l.strip()]
    repo, rev, _ = SOURCES["wmt"]
    out["wmt24pp.jsonl"] = [json.loads(l) for lang in WMT_LANGS for l in download(repo, rev, f"en-{lang}.jsonl").decode().splitlines() if l.strip()]
    for name, rows in out.items():
        path = DATA / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  hard/{name}  ({len(rows)} rows)")


def rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").split("\n") if l.strip()]  # not splitlines: texts hold U+2028


# ---------------------------------------------------------------- items

def mmlupro_items(per_category: int) -> list[dict]:
    by_cat: dict[str, list] = {}
    for r in sorted(rows("mmlupro.jsonl"), key=lambda r: r["question_id"]):
        by_cat.setdefault(r["category"], []).append(r)
    items = []
    for cat in sorted(by_cat):
        for r in by_cat[cat][:per_category]:
            items.append({"id": r["question_id"], "group": cat, "question": r["question"], "options": list(r["options"]),
                          "gold": r["answer"]})
    return items


def gpqa_items() -> list[dict]:
    items = []
    for r in rows("gpqa_diamond.jsonl"):
        q = r["Question"].strip()
        options = [r["Correct Answer"].strip()] + [r[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
        order = list(range(4))
        random.Random(zlib.crc32(q.encode())).shuffle(order)  # the same order in every arm
        items.append({"id": r["Record ID"], "group": r.get("High-level domain") or "", "question": q,
                      "options": [options[i] for i in order], "gold": LETTERS[order.index(0)]})
    return items


def ifeval_items() -> list[dict]:
    return [{"id": r["key"], "group": "", "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
             "kwargs": r["kwargs"]} for r in rows("ifeval.jsonl")]


def wmt_items(per_language: int) -> list[dict]:
    items, taken = [], {}
    for r in rows("wmt24pp.jsonl"):
        lang = r["lp"].split("-", 1)[1]
        if r["is_bad_source"] or r["domain"] == "canary" or taken.get(lang, 0) >= per_language:
            continue
        taken[lang] = taken.get(lang, 0) + 1
        items.append({"id": f"{lang}:{r['segment_id']}", "group": lang, "source": r["source"], "target": r["target"]})
    return items


# ---------------------------------------------------------------- prompts and grading

def choice_prompt(it: dict) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(it["options"]))
    return (f"{it['question']}\n\n{opts}\n\nThink step by step, then finish with a line containing only: "
            f"Answer: <letter>")


def choice_pred(text: str) -> str | None:
    found = ANSWER.findall(text)
    return found[-1] if found else None


def ifeval_grade(it: dict, response: str) -> dict:
    """Strict and loose prompt-level results with the reference checker (its evaluation_lib, inlined)."""
    from evals.ifeval import instructions_registry as registry

    def follows(text: str, iid: str, kwargs: dict) -> bool:
        inst = registry.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**{k: v for k, v in kwargs.items() if v is not None})
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=it["prompt"])
        return bool(text.strip()) and inst.check_following(text)

    lines = response.split("\n")
    variants = [response, response.replace("*", ""), "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(),
                "\n".join(lines[1:-1]).strip()]
    variants += [v.replace("*", "") for v in variants[2:]]
    strict, loose = [], []
    for iid, kwargs in zip(it["instruction_id_list"], it["kwargs"]):
        strict.append(follows(response, iid, kwargs))
        loose.append(any(follows(v, iid, kwargs) for v in variants))
    return {"correct": all(strict), "loose": all(loose), "instructions": len(strict), "instructions_followed": sum(strict)}


def run(task: str, url: str, items: list[dict], accept, max_tokens: int, concurrency: int) -> list[dict]:
    import sacrebleu

    def one(i, it):
        t0 = time.perf_counter()
        if task in ("mmlupro", "gpqa"):
            text, usage = ask(url, choice_prompt(it), max_tokens, accept)
            pred = choice_pred(text)
            row = {"correct": pred == it["gold"], "pred": pred, "gold": it["gold"], "text_tail": text[-160:]}
        elif task == "ifeval":
            text, usage = ask(url, it["prompt"], max_tokens, accept)
            row = ifeval_grade(it, text)
        else:
            lang = WMT_LANGS[it["group"]]
            text, usage = ask(url, f"Translate the following English text into {lang}. Reply with the translation only.\n\n{it['source']}",
                              max_tokens, accept)
            hyp = text.strip()
            row = {"score": sacrebleu.sentence_chrf(hyp, [it["target"]], word_order=2).score, "hyp": hyp[:400]}
        return {"index": i, "id": it["id"], "group": it["group"], **row, "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens"),
                "capped": (usage.get("completion_tokens") or 0) >= max_tokens}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- tasks

# Generous caps: thinking is off but the model still reasons at length, and a capped answer scores as wrong,
# which favours whichever arm writes shorter answers (GPQA at 2,048: half the exact arm's answers were cut off).
TASKS = {  # task: (dataset files, default max_tokens)
    "mmlupro": (["hard/mmlupro.jsonl"], 4096),
    "gpqa": (["hard/gpqa_diamond.jsonl"], 8192),
    "ifeval": (["hard/ifeval.jsonl", "nltk/tokenizers/punkt.zip", "nltk/tokenizers/punkt_tab.zip"], 4096),
    "wmt": (["hard/wmt24pp.jsonl"], 1024),
}


def load(task: str, per_category: int = 100, per_language: int = 100) -> list[dict]:
    return {"mmlupro": lambda: mmlupro_items(per_category), "gpqa": gpqa_items, "ifeval": ifeval_items,
            "wmt": lambda: wmt_items(per_language)}[task]()


# ---------------------------------------------------------------- compare

def paired_scores(a: list[float], b: list[float], resamples: int = 2000) -> dict:
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(0)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(resamples))
    wins, losses = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
    sign = mcnemar([True] * losses + [False] * wins, [False] * losses + [True] * wins)
    return {"delta": sum(d) / len(d), "ci95": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
            "wins": wins, "losses": losses, "p": sign["p"]}


def compare(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    base = runs[0]
    by_id = {x["id"]: x for x in base["items"]}
    print(f"{'arm':<14} {'task':<8} {'n':>5} {'score':>7} {'vs first':>22} {'p':>6} {'tok/item':>9} {'relaxed%':>9} {'capped':>7}")
    for r in runs:
        pairs = [(by_id[x["id"]], x) for x in r["items"] if x["id"] in by_id]
        toks = sum(x["completion_tokens"] or 0 for x in r["items"]); rel = sum(x["relaxed_tokens"] or 0 for x in r["items"])
        if "score" in r["items"][0]:
            score = sum(x["score"] for x in r["items"]) / len(r["items"])
            s = paired_scores([a["score"] for a, _ in pairs], [b["score"] for _, b in pairs])
            vs = f"{s['delta']:+.2f} [{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}]" if r is not base else ""
            p = s["p"] if r is not base else 1.0
            shown = f"{score:6.2f} "
        else:
            score = sum(x["correct"] for x in r["items"]) / len(r["items"])
            m = mcnemar([a["correct"] for a, _ in pairs], [b["correct"] for _, b in pairs])
            vs = f"{100 * (m['b_only'] - m['a_only']) / len(pairs):+.1f} pp ({m['b_only']}/{m['a_only']})" if r is not base else ""
            p = m["p"] if r is not base else 1.0
            shown = f"{100 * score:6.1f}%"
        print(f"{r['label']:<14} {r['task']:<8} {len(r['items']):>5} {shown} {vs:>22} {p:6.3f} {toks / len(r['items']):9.0f} {100 * rel / max(1, toks):8.2f}% {sum(bool(x.get('capped')) for x in r['items']):>7}")
    groups = sorted({x["group"] for x in base["items"] if x["group"]})
    if len(groups) > 1:
        key = "score" if "score" in base["items"][0] else "correct"
        print(f"\nby {'language' if base['task'] == 'wmt' else 'category'} (mean {key}; first arm, then each other arm's delta):")
        for g in groups:
            vals = []
            for r in runs:
                xs = [float(x[key]) for x in r["items"] if x["group"] == g]
                vals.append(sum(xs) / len(xs) * (1 if key == "score" else 100))
            print(f"  {g:<24} {vals[0]:7.2f}  " + "  ".join(f"{v - vals[0]:+6.2f}" for v in vals[1:]))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30004")
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--accept", default="server", help="K:M acceptance pinned on every request (1:0 exact); 'server' (default): the arm's profile")
    ap.add_argument("--per-category", type=int, default=100, help="mmlupro: the first N items of each of the 14 categories")
    ap.add_argument("--per-language", type=int, default=100, help="wmt: the first N good-source segments of each language")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="sharar")
    ap.add_argument("--plain", action="store_true", help="another OpenAI-compatible server: no accept/trace fields, no /get_server_info")
    args = ap.parse_args()
    SERVER.update(model=args.model, plain=args.plain)
    accept = None if args.accept == "server" else (int(args.accept.split(":")[0]), float(args.accept.split(":")[1]))
    items = load(args.task, args.per_category, args.per_language)
    if args.limit:
        items = items[:args.limit]
    max_tokens = args.max_tokens or TASKS[args.task][1]
    t0 = time.perf_counter()
    result_rows = run(args.task, args.url, items, accept, max_tokens, args.concurrency)
    wall = time.perf_counter() - t0
    info = {}
    if not args.plain:
        with urllib.request.urlopen(f"{args.url}/get_server_info", timeout=10) as r:
            info = json.loads(r.read())
    key = "score" if args.task == "wmt" else "correct"
    score = sum(float(r[key]) for r in result_rows) / len(result_rows)
    result = {"label": args.label, "task": args.task, "accept": accept if accept is not None else "server", "sampling": dict(SAMPLING),
              "n": len(result_rows), "score": score, "wall_seconds": wall, "max_tokens": max_tokens,
              "source": {"repo": SOURCES[args.task][0], "revision": SOURCES[args.task][1]},
              "server": {k: info.get(k) for k in ("target", "revision", "w3_target_identity", "w3_warning", "speculative")},
              "profile": (info.get("config") or {}).get("profile"), "items": result_rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}-{args.task}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"label": args.label, "task": args.task, "n": len(result_rows), "score": round(score, 4), "wall_s": round(wall, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
