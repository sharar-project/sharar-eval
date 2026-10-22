"""Fixed-prompt request sets as AIPerf mooncake_trace rows: the decode, long-context and prefill tests of
engine-bench's h2h.py (whose prompts, caps and document generator these are, verbatim), for the `decode`,
`longctx` and `prefill` workloads of `sharar-eval bench`.

- decode (chat, prose, code, math): 8 hand-written prompts per test, output capped at 256 / 1,024 / 1,024 / 768
  tokens with natural stops; request k sends prompt k mod 8.
- longctx: a fresh synthetic document of 12,600 tokens per request (h2h.py's size), then a question; output capped
  at 256.
- prefill: two fresh synthetic documents of the cell's length (4,000-128,000 tokens), one output token each; the
  first warms the server's prefill shapes, the second is the one to report.
Documents are cut to exactly their length in the pinned tokenizer's tokens (evals/speed/model.py; h2h.py's were about
0.79 of the generator's target). The request tag, the question or instruction and the chat template add about 30.
Every request opens with "[request N] ", N unique to the build (the time it was built), so no two requests share
more than a dozen prefix tokens, within a cell or across runs, whatever the server caches. Thinking off; greedy, or
at temperature 0.7 Qwen3's non-thinking preset (top-p 0.95, top-k 20, min-p 0, as h2h.py sent but for min-p).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

WORDS = ("time year people way day man thing woman life child world school state family student group country problem "
         "hand part place case week company system program question work government number night point home water room "
         "mother area money story fact month lot right study book eye job word business issue side kind head house "
         "service friend father power hour game line end member law car city community name river mountain signal "
         "engine garden theory market winter method bridge letter window voice paper field").split()

CHAT = ["What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
        "Give me three tips for sleeping better.",
        "What's the difference between a virus and a bacterium?",
        "How do I politely decline a meeting invitation from my manager?",
        "Explain what an index fund is in simple terms.",
        "What are good houseplants for a room with little light?",
        "Why do cats purr?",
        "How long should I boil an egg for a jammy yolk?"]
PROSE = ["Write a detailed essay on the history of bridge engineering, from Roman arches to modern cable-stayed bridges.",
         "Write a long short story about a lighthouse keeper who discovers a message in a bottle.",
         "Write an in-depth article explaining how the immune system fights infection, for a general audience.",
         "Write a detailed travel guide to spending a week in Kyoto, day by day.",
         "Write a long essay on the causes and consequences of the Industrial Revolution.",
         "Write a detailed explanation of how modern CPUs execute instructions, from fetch to retire.",
         "Write a long reflective essay on what makes a city livable.",
         "Write a detailed history of the printing press and its effects on European society."]
CODE = ["Write a complete Python module implementing an LRU cache class with get, put, delete and resize, full docstrings, type hints, and a pytest test suite.",
        "Write a C++17 thread-safe bounded blocking queue with push, pop, try_pop and close, plus a small demo main().",
        "Write a Rust command-line tool that counts word frequencies in a file and prints the top N, with error handling and unit tests.",
        "Write a TypeScript implementation of a debounce and a throttle function with generics, JSDoc and Jest tests.",
        "Write a Python asyncio web crawler with a concurrency limit, robots.txt respect and depth limit.",
        "Write a Go HTTP server with a JSON REST API for a todo list (CRUD), in-memory storage with a mutex, and tests.",
        "Write a Python implementation of Dijkstra's and A* shortest path on a grid, with a comparison benchmark.",
        "Write a SQL schema for a library system (books, members, loans) plus ten example queries with explanations."]
MATH = ["A train leaves at 9:40 at 72 km/h; another leaves the same station at 10:15 at 90 km/h on the same track. When and where does the second catch the first? Solve step by step.",
        "Find all real solutions of x^4 - 5x^2 + 4 = 0, showing each step.",
        "A rectangle's perimeter is 46 cm and its diagonal is 17 cm. Find its area. Show your reasoning step by step.",
        "How many ways can 8 people sit around a round table if two particular people refuse to sit next to each other? Explain step by step.",
        "Compute the sum of all positive divisors of 360, step by step.",
        "A tank fills in 6 hours with pipe A and 9 hours with pipe B; pipe C drains it in 12 hours. With all three open, how long to fill? Step by step.",
        "Prove that the square root of 2 is irrational, then explain each step.",
        "What is the probability of getting at least two sixes in five rolls of a fair die? Show the computation step by step."]
WORKLOADS = {  # name: (prompts, max_tokens)
    "chat": (CHAT, 256), "prose": (PROSE, 1024), "code": (CODE, 1024), "math": (MATH, 768), "longctx": (None, 256)}
PREFILL_LENGTHS = (4000, 8000, 16000, 32000, 64000, 128000)
LONGCTX_TOKENS = 12600  # h2h.py's long-context document, as the Qwen3.8 tokenizer counts it
QUESTION = "\n\nIn three sentences, what are the most frequent themes in the text above?"
SUMMARIZE = "\n\nSummarize the text above in one sentence."


def prose_doc(tokens, seed):
    rng = random.Random(seed); out = []
    while len(out) < tokens * 0.72:
        s = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]; out += s[:-1] + [s[-1] + "."]
    return " ".join(out)


_tokenizers = {}


def sized_doc(tokens, seed, snapshot=None):
    """A synthetic document of exactly `tokens` tokens in the pinned tokenizer, cut where a token ends. `snapshot`: the
    Hugging Face cache holding the pinned snapshot (evals/speed/model.py's default when None)."""
    if snapshot not in _tokenizers:
        from evals.speed import model
        _tokenizers[snapshot] = model.SnapshotTokenizer(model.snapshot_dir(snapshot))
    tok = _tokenizers[snapshot]
    ids = tok.encode(prose_doc(int(tokens * 1.4), seed))  # the generator gives about 0.79 tokens per target
    n = tokens
    while len(tok.encode(text := tok.decode(ids[:n]))) > tokens:  # a cut can merge into one token less
        n -= 1
    return text


def sampling(temperature):
    extra = {"temperature": float(temperature), "chat_template_kwargs": {"enable_thinking": False}}
    return {**extra, "top_p": 0.95, "top_k": 20, "min_p": 0.0} if temperature else extra


def rows(test, requests, temperature=0.0, tag=None, snapshot=None):
    """mooncake_trace rows for one cell: `requests` single-call sessions of a decode test or longctx, or the two
    requests of a prefill length (test "prefill-<tokens>")."""
    tag = int(time.time() * 1000) % 10**9 * 100 if tag is None else tag
    out = []
    if test.startswith("prefill-"):
        tokens = int(test.split("-", 1)[1])
        for p in range(2):
            out.append((f"[request {tag + p}] " + sized_doc(tokens, tag * 7919 + p, snapshot) + SUMMARIZE, 1, 0.0))
    elif test == "longctx":
        for k in range(requests):
            out.append((f"[request {tag + k}] " + sized_doc(LONGCTX_TOKENS, tag * 1000 + k, snapshot) + QUESTION, WORKLOADS[test][1], temperature))
    else:
        prompts, cap = WORKLOADS[test]
        out = [(f"[request {tag + k}] " + prompts[k % len(prompts)], cap, temperature) for k in range(requests)]
    return [{"session_id": f"{test}-{tag + k}", "messages": [{"role": "user", "content": text}], "output_length": cap,
             "extra": sampling(t)} for k, (text, cap, t) in enumerate(out)]


def write(path: Path, test, requests, temperature=0.0, tag=None, snapshot=None) -> int:
    """The cell's rows to path (JSONL); returns how many."""
    out = rows(test, requests, temperature, tag, snapshot)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in out))
    return len(out)
