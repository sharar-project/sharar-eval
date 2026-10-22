"""Coding-agent replay at 1..N concurrent agents, after Artificial Analysis's agentperf-local (aa-agentperf-local,
Apache-2.0), whose runner replays the same traces at concurrency one only. Ported unchanged in method from
engine-bench's suite/agentload.py and suite/agentsum.py (f805635), so results stay comparable with engine-bench's;
`sharar-eval efficiency agent-replay` runs it.

Kept from agentperf-local (its replay `agentperf-default-v1`: 8 recorded agent tasks, 3 SWE-bench repository fixes
and 5 OpenClaw assistant tasks, 168 model turns, contexts up to ~56k tokens; pinned in evals/datasets.py):
- every turn sends its recorded conversation so far (system prompt, task, earlier assistant turns with their tool
  calls, tool results) with the task's `tools` schemas, streamed;
- the `exact` output policy: max_tokens = the turn's recorded output length and ignore_eos, so every engine does the
  same work whatever its model would have written; a turn that generates fewer tokens fails. agentperf-local also
  fails a turn whose finish reason is not "length"; such turns did the full work, so here they are counted apart
  ("finish_not_length", e.g. a server reporting "tool_calls" for a full-length reply that holds a call);
- cache isolation: a namespace of 32 random digits heads the first message, so turns of one conversation share their
  prefix as a real agent's do and nothing is reused from another run;
- the standard sampling preset (temperature 0.7, top-p 0.8, top-k 20, min-p 0);
- tool time: none by default (tool_mode "recorded" sleeps the recorded tool durations after each turn).
Added for many agents: N agents run at once, each taking the next task when its own ends (closed loop); the 8 tasks
run once at N <= 8, and copies run at N > 8 (each copy under its own namespace, so copies share nothing).
Deviation: thinking is turned off (chat_template_kwargs enable_thinking false), so all engines run the same template
whatever their default.

Records (one JSON object a line, as engine-bench's agents*.jsonl): "agent_turn" per turn (raw stream timings, usage,
tool calls, faults) and "agent_level" per level. `summarize` gives the per-level figures engine-bench publishes
(agentsum.py), with agentperf-local's acceptance rule for the exact policy: a turn fails only on an HTTP or stream
error or a short output. Tool calls are not judged (generation past the model's own stop, which exact asks for, cuts
and corrupts them by design); unparsed arguments and finish reasons other than "length" are counted for information.
"""
from __future__ import annotations

import collections
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

NS_DIGITS = 32


def namespace(rng):
    digits = " ".join(str(rng.randrange(10)) for _ in range(NS_DIGITS))
    return f"{digits}\nPerformance replay cache namespace. Ignore the digits above.\n\n"


def isolate(messages, ns):
    msgs = json.loads(json.dumps(messages))
    c = msgs[0].get("content")
    if isinstance(c, list):
        msgs[0]["content"] = [{"type": "text", "text": ns}] + c
    else:
        msgs[0]["content"] = ns + (c or "")
    return msgs


def pct(v, q):
    if not v:
        return None
    v = sorted(v); k = (len(v) - 1) * q; f = math.floor(k)
    return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)


def load_tasks(replay: Path, manifest: str = "manifest-arm64.json") -> list[dict]:
    """agentperf-local's replay directory: the manifest's tasks in order, each with its trace rows."""
    m = json.loads((replay / manifest).read_text())
    return [{"task_id": t["task_id"], "rows": [json.loads(line) for line in (replay / t["trace"]).read_text().splitlines()
                                               if line.strip()]} for t in m["tasks"]]


def turn_request(url, model, row, ns, timeout=3600):
    """One replayed turn, streamed; raw timings and what came back, parsed after the stream closes."""
    body = {"model": model, "messages": isolate(row["messages"], ns), "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": row["target_output_tokens"], "ignore_eos": True,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if row.get("tools"):
        body["tools"] = row["tools"]
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); lines = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                lines.append((time.time(), line))
    except Exception as e:  # noqa: BLE001
        return {"t0": t0, "end": time.time(), "error": f"{type(e).__name__}: {e}"[:300]}
    end = time.time()
    first = last = None; usage = {}; finish = None; text = []; calls = {}; done = False; chunks = 0
    for ts, line in lines:
        line = line.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        if line.endswith("[DONE]"):
            done = True; continue
        try:
            c = json.loads(line[5:])
        except json.JSONDecodeError:
            return {"t0": t0, "end": end, "error": "stream chunk is not JSON"}
        usage = c.get("usage") or usage
        ch = (c.get("choices") or [{}])[0]
        finish = ch.get("finish_reason") or finish
        d = ch.get("delta") or {}
        visible = (d.get("content") or "") + (d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:
            slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""; slot["arguments"] += fn.get("arguments") or ""
            visible = visible or "tool"
        if visible:
            first = first or ts; last = ts; chunks += 1; text.append(d.get("content") or "")
    return {"t0": t0, "end": end, "first": first, "last": last, "finish": finish, "done": done, "chunks": chunks,
            "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "tool_calls": list(calls.values()), "text_chars": len("".join(text))}


def check(r, row):
    """The turn's faults: what agentperf-local's transport and exact-length checks would refuse, plus tool calls."""
    if "error" in r:
        return ["http: " + r["error"]]
    out = []
    if not r["done"]:
        out.append("stream: no [DONE]")
    if (r["completion_tokens"] or 0) < row["target_output_tokens"]:
        out.append(f"exact: {r['completion_tokens']} of {row['target_output_tokens']} tokens, finish {r['finish']}")
    names = {t["function"]["name"] for t in row.get("tools") or []}
    for c in r["tool_calls"]:
        if c["name"] not in names:
            out.append(f"tool: unknown name {c['name']!r}")
        try:
            if not isinstance(json.loads(c["arguments"] or "{}"), dict):
                out.append("tool: arguments not an object")
        except json.JSONDecodeError:
            out.append("tool: arguments not JSON")
    return out


def level(a, tasks, n, log, rng):
    """One level: n agents at once over the tasks (copies past 8). `a` carries url, model, label and tool_mode; every
    turn is appended to `log` as it ends, and the level's record is returned."""
    copies = max(1, math.ceil(n / len(tasks)))
    queue = [(c, t) for c in range(copies) for t in tasks]
    ns = {c: namespace(rng) for c in range(copies)}  # one per copy: tasks of a copy share like agentperf's run does
    lock = threading.Lock(); episodes = []

    def agent(slot):
        while True:
            with lock:
                if not queue:
                    return
                copy, task = queue.pop(0)
            t_start = time.time(); turns = []
            for row in task["rows"]:
                r = turn_request(a.url, a.model, row, ns[copy])
                r["faults"] = check(r, row)
                r.update(task=task["task_id"], copy=copy, agent=slot, turn=row["conversation_idx"],
                         target=row["target_output_tokens"], recorded_prompt=row["recorded_prompt_tokens"])
                turns.append(r)
                with lock:
                    log.write(json.dumps({"kind": "agent_turn", "label": a.label, "agents": n, **r}) + "\n"); log.flush()
                if a.tool_mode == "recorded" and row.get("simulated_tool_delay_ms_after"):
                    time.sleep(row["simulated_tool_delay_ms_after"] / 1000)
            with lock:
                episodes.append({"task": task["task_id"], "copy": copy, "seconds": time.time() - t_start, "turns": turns})

    t0 = time.time()
    th = [threading.Thread(target=agent, args=(i,)) for i in range(n)]
    [x.start() for x in th]; [x.join() for x in th]
    wall = time.time() - t0
    turns = [t for e in episodes for t in e["turns"]]
    good = [t for t in turns if not t["faults"]]
    ok = [t for t in turns if "error" not in t and t.get("first")]
    win = [(t["completion_tokens"], t["last"] - t["first"]) for t in ok
           if not t["tool_calls"] and (t["completion_tokens"] or 0) >= 2 and t["last"] > t["first"]]
    out_tokens = sum(t["completion_tokens"] or 0 for t in ok)
    prompt = sum(t["prompt_tokens"] or 0 for t in ok); cached = [t["cached_tokens"] for t in ok if t["cached_tokens"] is not None]
    ttft = [1000 * (t["first"] - t["t0"]) for t in ok]; e2e = [1000 * (t["end"] - t["t0"]) for t in ok]
    return {"kind": "agent_level", "label": a.label, "agents": n, "tasks": len(episodes), "turns": len(turns),
            "wall_s": wall, "aggregate_tok_s": out_tokens / wall if wall else None,
            "output_tokens_per_second": sum(k - 1 for k, _ in win) / sum(s for _, s in win) if win else None,
            "e2e_output_tokens_per_second": out_tokens / sum(t["end"] - t["t0"] for t in ok) if ok else None,
            "ttft_p50_ms": pct(ttft, 0.5), "ttft_p95_ms": pct(ttft, 0.95),
            "turn_p50_ms": pct(e2e, 0.5), "turn_p95_ms": pct(e2e, 0.95),
            "task_p50_s": pct([e["seconds"] for e in episodes], 0.5), "task_p90_s": pct([e["seconds"] for e in episodes], 0.9),
            "tasks_per_hour": 3600 * len(episodes) / wall if wall else None,
            "cache_share": sum(cached) / prompt if cached and prompt else None,
            "output_tokens": out_tokens, "prompt_tokens": prompt, "failed_turns": len(turns) - len(good),
            "faults": sorted({f.split(":")[0] + ":" + f.split(":")[1][:40] for t in turns for f in t["faults"]})[:8],
            "tool_call_turns": sum(bool(t.get("tool_calls")) for t in ok), "tool_mode": a.tool_mode,
            "finish_not_length": sum(t.get("finish") != "length" for t in ok),
            "finishes": dict(sorted(collections.Counter(str(t.get("finish")) for t in ok).items()))}


def summarize(records: list[dict]) -> list[dict]:
    """Per (label, agents) level, from the turn records and the level's wall time (engine-bench's agentsum.py): task
    time is first turn sent to last turn ended, so tool time after a task's last turn is not counted."""
    walls = {(r["label"], r["agents"]): r for r in records if r["kind"] == "agent_level"}
    turns = collections.defaultdict(list)
    for r in records:
        if r["kind"] == "agent_turn":
            turns[(r["label"], r["agents"])].append(r)
    out = []
    for key, ts in sorted(turns.items()):
        lv = walls.get(key)
        if not lv:
            continue
        ok = [t for t in ts if "error" not in t and t.get("first")]
        failed = [t for t in ts if "error" in t or not t.get("done") or (t.get("completion_tokens") or 0) < t["target"]]
        bad_args = 0
        for t in ok:
            for c in t.get("tool_calls") or []:
                try:
                    json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    bad_args += 1; break
        win = [(t["completion_tokens"], t["last"] - t["first"]) for t in ok
               if not t["tool_calls"] and (t["completion_tokens"] or 0) >= 2 and t["last"] > t["first"]]
        tasks = collections.defaultdict(list)
        for t in ts:
            tasks[(t["task"], t["copy"])].append(t)
        task_s = [max(x["end"] for x in v) - min(x["t0"] for x in v) for v in tasks.values()]
        prompt = sum(t["prompt_tokens"] or 0 for t in ok); cached = [t["cached_tokens"] for t in ok if t["cached_tokens"] is not None]
        ttft = [1000 * (t["first"] - t["t0"]) for t in ok]; e2e = [1000 * (t["end"] - t["t0"]) for t in ok]
        out_tokens = sum(t["completion_tokens"] or 0 for t in ok)
        r1 = lambda v, d=1: None if v is None else round(v, d)  # noqa: E731
        out.append({"label": key[0], "agents": key[1], "tasks": len(tasks), "turns": len(ts), "wall_s": round(lv["wall_s"], 1),
                    "aggregate_tok_s": round(out_tokens / lv["wall_s"], 2), "tasks_per_hour": round(3600 * len(tasks) / lv["wall_s"], 1),
                    "task_p50_s": r1(pct(task_s, 0.5)), "task_p90_s": r1(pct(task_s, 0.9)),
                    "ttft_p50_ms": r1(pct(ttft, 0.5), None), "ttft_p95_ms": r1(pct(ttft, 0.95), None),
                    "turn_p50_ms": r1(pct(e2e, 0.5), None), "turn_p95_ms": r1(pct(e2e, 0.95), None),
                    "decode_tok_s_pooled": round(sum(k - 1 for k, _ in win) / sum(s for _, s in win), 1) if win else None,
                    "cache_share": round(sum(cached) / prompt, 3) if cached and prompt else None,
                    "output_tokens": out_tokens, "failed_turns": len(failed),
                    "info_tool_args_unparsed": bad_args, "info_finish_not_length": sum(t.get("finish") != "length" for t in ok)})
    return out
