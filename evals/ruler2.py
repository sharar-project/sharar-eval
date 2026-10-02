"""RULER v2 long-context tasks against a running server (sharar#28): three families, four levels each.

    python -m evals.ruler2 prepare --length 32768          # generate the pinned sets (python -m evals.datasets fetch does all)
    python -m evals.suite run --suite long --label fast --out runs/fast   # 32K and 128K, 10 per task (~3 h)
    python -m evals.suite run --tasks ruler2-64k --limit 240 --label fast --out runs/fast

RULER v2 (Hsieh et al., NVIDIA, 2025) grades each family from basic retrieval up to reasoning over what
was retrieved, so a long-context loss shows as "cannot find it" or as "found it, cannot use it":
    mk_niah   multi-key needles: basic = a 10-digit value for one word key among look-alike needles;
              easy = copy the MMLU question with a given id; medium = copy it, then answer it (5-shot);
              hard = answer it directly (5-shot)
    mv_niah   multi-value needles: basic = all 4 values of one key; easy = copy all 4 questions sharing
              an id; medium = copy all 4, then the k-th; hard = copy only the k-th
    qa        HotpotQA documents: basic = which document holds this text; easy = which documents answer
              this question; medium = copy the relevant documents, then answer; hard = answer directly

Ported from NVIDIA-NeMo/Skills nemo_skills/dataset/ruler2 and evaluation/evaluator/ruler.py at bcf059a
(Apache-2.0): the same prompts, generators, seed (42 per task), length fitting and scoring. Differences:
the sources are pinned parquet files (cais/mmlu@c30699e, hotpotqa/hotpot_qa@1908d6a) instead of
`datasets.load_dataset`; the QA haystack is sorted after deduplication (upstream iterates a Python set,
whose order changes between processes, so its sets are not reproducible); lengths count Qwen3.8 tokens
(the pinned tokenizer) without the chat template; requests are greedy with thinking off and at most
4,096 new tokens (upstream: 16,384, thinking per model).

A length's set is 100 samples of each task, interleaved (item i is task i % 12), so the first N items
cover every task evenly. Flash-Next `fast` (2026-10-02, 4 streams): 99.4 at 32K (48 items, ~19 s each) and
86.5 at 128K (24 items, ~75 s each; the misses are mv_niah_hard and mk_niah_medium), so 128K is where
configurations separate and 32K says whether a loss is about length at all. Scores are 0-100 per item: string match or 1 - word error rate for the
retrieval and QA tasks, the boxed letter for the two MMLU-answering tasks; the task score is the mean.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.accuracy_eval import ask
from evals.datasets import ROOT

DATA = ROOT / "ruler2"
LENGTHS = {"32k": 32768, "64k": 65536, "128k": 131072}
SAMPLES = 100
MAX_TOKENS = 4096
SOURCES = ["ruler2/src/mmlu_test.parquet", "ruler2/src/mmlu_auxiliary_train.parquet",
           "ruler2/src/hotpot_train-00000-of-00002.parquet", "ruler2/src/hotpot_train-00001-of-00002.parquet",
           "ruler2/src/hotpot_validation-00000-of-00001.parquet"]
TASKS = {f"ruler2-{k}": ([f"ruler2/ruler2-{v}.jsonl"], MAX_TOKENS) for k, v in LENGTHS.items()}

# task: (generator, options, match); the options are upstream's prepare.py arguments
NIAH = {"num_needle_k": 1, "num_needle_q": 1, "type_needle_k": "words", "type_needle_v": "numbers", "num_digits_k": 7, "num_digits_v": 10}
SUBTASKS = {
    "mk_niah_basic": ("niah", {**NIAH, "num_needle_v": 1}, "all"),
    "mk_niah_easy": ("mmlu", {"fewshot": 0, "num_order": 0, "task_type": "retrieve", "algo_type": "single"}, "all"),
    "mk_niah_medium": ("mmlu", {"fewshot": 5, "num_order": 0, "task_type": "solve", "algo_type": "2steps"}, "choice"),
    "mk_niah_hard": ("mmlu", {"fewshot": 5, "num_order": 0, "task_type": "solve", "algo_type": "single"}, "choice"),
    "mv_niah_basic": ("niah", {**NIAH, "num_needle_v": 4}, "all"),
    "mv_niah_easy": ("mmlu", {"fewshot": 0, "num_order": 4, "task_type": "niah", "algo_type": "single"}, "all"),
    "mv_niah_medium": ("mmlu", {"fewshot": 0, "num_order": 4, "task_type": "retrieve", "algo_type": "2steps"}, "2steps"),
    "mv_niah_hard": ("mmlu", {"fewshot": 0, "num_order": 4, "task_type": "retrieve", "algo_type": "single"}, "all"),
    "qa_basic": ("qa", {"task_type": "retrieve", "query_type": "doc", "algo_type": "single"}, "part"),
    "qa_easy": ("qa", {"task_type": "retrieve", "query_type": "question", "algo_type": "single"}, "part"),
    "qa_medium": ("qa", {"task_type": "solve", "query_type": "id", "algo_type": "2steps"}, "part"),
    "qa_hard": ("qa", {"task_type": "solve", "query_type": "id", "algo_type": "single"}, "part"),
}


# ---------------------------------------------------------------- generation (ported)

def random_number(rng: random.Random, digits: int = 7) -> str:
    return str(rng.randint(10 ** (digits - 1), 10 ** digits - 1))


def fit(make, tok, length: int, incremental: int, items) -> list[dict]:
    """Upstream's length fitting: binary-search the haystack size on item 0, then draw the items (`items()`) and step
    the size down per item until it fits. The random draws happen in upstream's order."""
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    per = count(make(0, incremental)[0]) / incremental
    lo, hi, best = incremental, max(int(length / per * 3), incremental * 2), None
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(make(0, mid)[0]) <= length:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    size, out = best or incremental, []
    for item in items():
        used = size
        while True:
            text, row = make(item, used)
            n = count(text)
            if n <= length:
                break
            if used <= incremental:
                raise ValueError(f"cannot fit {length} tokens")
            used -= incremental
        out.append({**row, "length": n})
    return out


def gen_niah(opt: dict, tok, length: int, rng: random.Random) -> list[dict]:
    from wonderwords import random_word
    nouns = random_word._get_words_from_text_file("nounlist.txt")
    adjs = random_word._get_words_from_text_file("adjectivelist.txt")
    words = sorted({f"{a}-{n}" for a in adjs for n in nouns})
    single = ("A special magic {t} is hidden within the following text. Make sure to memorize it. I will quiz you about the {t} afterwards.\n"
              "{context}\nWhat is the special magic {t} for {query} mentioned in the provided text? The special magic {t} for {query} mentioned in the provided text is")
    multiple = ("Some special magic {t} are hidden within the following text. Make sure to memorize them. I will quiz you about the {t} afterwards.\n"
                "{context}\nWhat are all the special magic {t} for {query} mentioned in the provided text? The special magic {t} for {query} mentioned in the provided text are")
    needle = "One of the special magic {t} for {key} is: {value}."
    nk, nv, nq = max(opt["num_needle_k"], opt["num_needle_q"]), opt["num_needle_v"], opt["num_needle_q"]

    def rand(kind, digits):
        if kind == "numbers":
            return random_number(rng, digits)
        if kind == "words":
            return rng.choice(words)
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))

    def make(index, num_haystack):
        keys, values, needles = [], [], []
        for _ in range(nk):
            keys.append(rand(opt["type_needle_k"], opt["num_digits_k"]))
            value = []
            for _ in range(nv):
                value.append(rand(opt["type_needle_v"], opt["num_digits_v"]))
                needles.append(needle.format(t=opt["type_needle_v"], key=keys[-1], value=value[-1]))
            values.append(value)
        rng.shuffle(needles)
        if nv == 1:
            sentences = [needle.format(t=opt["type_needle_v"], key=rand(opt["type_needle_k"], opt["num_digits_k"]),
                                       value=rand(opt["type_needle_v"], opt["num_digits_v"])) for _ in range(num_haystack)]
        else:
            hv = [rand(opt["type_needle_v"], opt["num_digits_v"]) for _ in range(num_haystack)]
            hk = ([rand(opt["type_needle_k"], opt["num_digits_k"]) for _ in range(math.ceil(num_haystack / nv))] * nv)[:num_haystack]
            sentences = [needle.format(t=opt["type_needle_v"], key=hk[i], value=hv[i]) for i in range(num_haystack)]
            rng.shuffle(sentences)
        for i, n in zip(sorted(rng.sample(range(num_haystack), len(needles)), reverse=True), needles):
            sentences.insert(i, n)
        picked = rng.sample(range(nk), nq)
        queries, answers = [keys[i] for i in picked], [a for i in picked for a in values[i]]
        query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]
        template, t = (single, opt["type_needle_v"][:-1]) if nq * nv == 1 else (multiple, opt["type_needle_v"])
        text = template.format(t=t, context="\n".join(sentences), query=query)
        return text, {"index": index, "question": text, "expected": answers}

    incremental = max(5, nv * nk) if length >= 4096 else 5
    return fit(make, tok, length, incremental, lambda: range(SAMPLES))


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def mmlu_rows(name: str) -> list[dict]:
    import pyarrow.parquet as pq
    out = []
    for d in pq.read_table(ROOT / f"ruler2/src/mmlu_{name}.parquet").to_pylist():
        c = d["choices"]
        letter = "ABCD"[d["answer"]]
        out.append({"Question": re.sub(r"\s+", " ", d["question"] + f"\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}"),
                    "Solution": f" \\boxed{{{letter}}}", "Answer": letter})
    return out


def gen_mmlu(opt: dict, tok, length: int, rng: random.Random) -> list[dict]:
    task, algo, fewshot, num_order = opt["task_type"], opt["algo_type"], opt["fewshot"], opt["num_order"]
    if task == "retrieve":
        context_prompt = "Below are some questions. I will ask you to copy one of them. Please copy and paste the question you find.\n\n{needles}"
        problem_prompt = {"single": "Please copy the {order}Question {i} from the context.",
                          "2steps": "Please first copy all the Question {i} from the context and then copy the {order}Question {i} at the end."}[algo]
        example_prompt = problem_prompt + "\nQuestion {i}: {question}"
    elif task == "niah":
        context_prompt = "Below are some questions. I will ask you to copy some of them. Please copy and paste the questions you find.\n\n{needles}"
        problem_prompt = "Please find and copy all the Question {i} from the context."
    else:
        context_prompt = ("Below are some questions. I will ask you to solve one of them. Please solve the question you find and make sure to put "
                          "the answer (and only answer) inside \\boxed\\{{\\}}.\n\n{needles}")
        problem_prompt = {"single": "Please solve the Question {i} from the context with an answer from A, B, C, D.",
                          "2steps": "Please copy the Question {i} from the context and then solve it with an answer from A, B, C, D."}[algo]
        example_prompt = problem_prompt + ("\nSolution:{solution}" if algo == "single" else "\nQuestion {i}: {question}\nSolution:{solution}")
    needle, haystack = mmlu_rows("test"), mmlu_rows("auxiliary_train")

    def make(index, num_qs):
        repeats = (num_qs + len(haystack) - 1) // len(haystack) if num_qs > len(haystack) else 1
        ctx = [dict(item) for item in rng.sample([item for item in haystack for _ in range(repeats)], num_qs)]
        if num_order > 0:
            numbers = [random_number(rng) for _ in range(math.ceil((num_qs + 1) / num_order))] * num_order
        else:
            numbers = [random_number(rng) for _ in range(num_qs + 1)]
        rng.shuffle(numbers)
        numbers = numbers[:num_qs + 1]
        for i, q in enumerate(ctx):
            q["random_index"] = numbers[i]
        rng.shuffle(ctx)
        examples = rng.sample(ctx, fewshot)
        true = {**needle[index], "random_index": numbers[-1]}
        ctx.insert(rng.randint(0, len(ctx)), true)
        counts = defaultdict(int)
        for q in ctx:
            counts[q["random_index"]] += 1
            q["order"] = ordinal(counts[q["random_index"]]) + " (1 indexed) " if num_order > 0 else ""
        needles = "\n\n".join(f"Question {q['random_index']}: {q['Question']}" for q in ctx)
        problem = (problem_prompt.format(i=true["random_index"]) if task == "niah"
                   else problem_prompt.format(i=true["random_index"], order=true["order"]))
        example = ""
        if fewshot > 0:
            if task == "retrieve":
                example = "\n\n".join(example_prompt.format(i=q["random_index"], question=q["Question"], order=q["order"]) for q in examples)
            elif algo == "single":
                example = "\n\n".join(example_prompt.format(i=q["random_index"], solution=q["Solution"]) for q in examples)
            else:
                example = "\n\n".join(example_prompt.format(i=q["random_index"], question=q["Question"], solution=q["Solution"]) for q in examples)
            example = f"Here are some examples to help you understand the task:\n\n{example}\n\nHere is the actual task you need to solve:\n\n"
        text = f"{context_prompt.format(needles=needles)}\n\n{example}{problem}"
        if task == "retrieve":
            expected = [true["Question"]]
        elif task == "niah":
            expected = [c["Question"] for c in ctx if c["random_index"] == true["random_index"]]
        else:
            expected = true["Answer"]
        return text, {"index": index, "question": text, "expected": expected}

    return fit(make, tok, length, max(10, fewshot), lambda: rng.sample(list(range(len(needle))), min(len(needle), SAMPLES)))


def hotpot() -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq
    doc = lambda t, s: f"{t}\n{''.join(s)}"
    docs = set()
    for name in ("train-00000-of-00002", "train-00001-of-00002"):
        for d in pq.read_table(ROOT / f"ruler2/src/hotpot_{name}.parquet", columns=["context"]).to_pylist():
            docs.update(doc(t, s) for t, s in zip(d["context"]["title"], d["context"]["sentences"]))
    haystack = [{"text": d} for d in sorted(docs)]  # sorted: upstream's set order is not reproducible
    needle = []
    for d in pq.read_table(ROOT / "ruler2/src/hotpot_validation-00000-of-00001.parquet").to_pylist():
        sup = d["supporting_facts"]["title"]
        pairs = list(zip(d["context"]["title"], d["context"]["sentences"]))
        needle.append({"question": d["question"], "answer": [d["answer"]],
                       "context": [{"text": doc(t, s)} for t, s in pairs if t in sup],
                       "distractor": [{"text": doc(t, s)} for t, s in pairs if t not in sup]})
    return haystack, [n for n in needle if n["answer"]]


def gen_qa(opt: dict, tok, length: int, rng: random.Random, sources=None) -> list[dict]:
    task, query, algo = opt["task_type"], opt["query_type"], opt["algo_type"]
    if task == "retrieve" and query == "doc":
        context_prompt = ("Below are some documents. I will give you a text at the end. Please find the document index of the text. "
                          "Only give me the index without any document contents.\n\n{needles}")
        problem_prompt = "Text: {text}\nMost relevant document index:"
    elif task == "retrieve":
        context_prompt = ("Below are some documents. I will give you a question at the end. Please find the index of the most relevant document "
                          "that can help answer the question. Only give me the index without any document contents.\n\n{needles}")
        problem_prompt = "Question: {question}\nIndex of the most relevant document that can help answer the question:"
    else:
        context_prompt = "Below are some documents. I will ask you to answer a question based on the documents. Please answer the question.\n\n{needles}"
        problem_prompt = {"single": "Please answer the following question based on the documents.\n\nQuestion: {question}",
                          "2steps": "Please first find and copy paste the documents relevant to the following question and then answer it based on the documents you find.\n\nQuestion: {question}"}[algo]
    haystack, needle = sources or hotpot()

    def make(index, num_docs):
        cur = dict(needle[index])
        cur["context"] = [{**c, "random_index": random_number(rng)} for c in cur["context"]]
        cur["distractor"] = [{**c, "random_index": random_number(rng)} for c in cur["distractor"]]
        remaining = len(haystack) - len({c["text"] for c in cur["context"] + cur["distractor"]})
        repeats = (num_docs + remaining - 1) // remaining if num_docs > remaining else 1
        ctx = rng.sample([i for i in range(len(haystack)) for _ in range(repeats)], num_docs)
        ctx = [{**haystack[i], "random_index": random_number(rng)} for i in ctx] + cur["context"]
        if num_docs > 0:
            ctx += cur["distractor"]
        rng.shuffle(ctx)
        needles = "\n\n".join(f"Document {c['random_index']}:\n{c['text']}" for c in ctx)
        if task == "retrieve" and query == "doc":
            problem, expected = problem_prompt.format(text=cur["context"][0]["text"]), [cur["context"][0]["random_index"]]
        elif task == "retrieve":
            problem, expected = problem_prompt.format(question=cur["question"]), [c["random_index"] for c in cur["context"]]
        else:
            problem, expected = problem_prompt.format(question=cur["question"]), cur["answer"]
        text = f"{context_prompt.format(needles=needles)}\n\n{problem}"
        return text, {"index": index, "question": text, "expected": expected}

    return fit(make, tok, length, 5, lambda: range(SAMPLES))


def prepare(length: int) -> Path:
    """Generate the 12 tasks at one length (seed 42 per task, as upstream) into one interleaved JSONL file."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "long_context" / "tokenizer.json"))
    gens = {"niah": gen_niah, "mmlu": gen_mmlu, "qa": gen_qa}
    qa_sources = None
    per_task = {}
    for name, (gen, opt, match) in SUBTASKS.items():
        t0 = time.monotonic()
        if gen == "qa":
            qa_sources = qa_sources or hotpot()
            rows = gen_qa(opt, tok, length, random.Random(42), qa_sources)
        else:
            rows = gens[gen](opt, tok, length, random.Random(42))
        per_task[name] = [{"id": f"{name}:{r['index']}", "group": name, "match": match, **r} for r in rows]
        print(f"{name}: {len(rows)} samples, {min(r['length'] for r in rows)}-{max(r['length'] for r in rows)} tokens, "
              f"{time.monotonic() - t0:.0f}s", flush=True)
    path = DATA / f"ruler2-{length}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [per_task[name][i] for i in range(SAMPLES) for name in SUBTASKS]  # interleaved: any prefix covers every task
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------- scoring (ported)

def wer(hyp: str, ref: str) -> float:
    import editdistance
    r = ref.split()
    return editdistance.eval(hyp.split(), r) / len(r) if r else math.inf


def soft(pred: str, ref: str) -> float:
    pred, ref = pred.lower(), ref.lower()
    return max(1.0 if ref in pred else 0.0, 1 - wer(pred, ref))


def choice(text: str) -> str | None:
    """NeMo-Skills' multiple-choice extraction: the last \\boxed{} (a letter in it), else the last "Answer: X"."""
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        inner = boxed[-1].strip()
        if len(inner) == 1:
            return inner.upper()
        found = re.findall(r"\b[A-Z]\b(?!.*\b[A-Z]\b)", inner, re.DOTALL)
        if found:
            return found[-1].upper()
    found = re.findall(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])", text)
    return found[-1].upper() if found else None


def score(match: str, text: str, expected) -> float:
    if match == "choice":
        return 100.0 if choice(text) == expected else 0.0
    if match == "2steps":
        text = text.split("\n\n")[-1]
    if match == "part":
        text = re.sub(r"Document \d+:(?:.*\n)+?\n", "", text)
        return 100.0 * max(soft(text, r) for r in expected)
    return 100.0 * sum(soft(text, r) for r in expected) / len(expected)


# ---------------------------------------------------------------- suite interface

def load(task: str) -> list[dict]:
    path = ROOT / TASKS[task][0][0]
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def run(task: str, url: str, items: list[dict], accept, max_tokens: int, concurrency: int) -> list[dict]:
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, it["question"], max_tokens, accept)
        return {"index": i, "id": it["id"], "group": it["group"], "score": score(it["match"], text, it["expected"]),
                "prompt_tokens": usage.get("prompt_tokens"), "seconds": time.perf_counter() - t0, "text_tail": text[-200:],
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens"),
                "capped": (usage.get("completion_tokens") or 0) >= max_tokens}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("--length", type=int, action="append", help=f"tokens (default: {', '.join(map(str, LENGTHS.values()))})")
    args = ap.parse_args()
    for length in args.length or LENGTHS.values():
        print(f"{prepare(length)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
