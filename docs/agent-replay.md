# Agent replay: how we measure engines on recorded coding-agent work

`sharar-eval efficiency agent-replay` replays Artificial Analysis's recorded agent tasks against any running
OpenAI-compatible server, at 1 to N agents at once, and reports tasks per hour, time to first token and how much of
each prompt the server served from its prefix cache. This page is the method and how to reproduce our numbers; how
each engine is started is [`engines/`](../engines/README.md), and every flag is in
`uv run sharar-eval efficiency agent-replay --help`. The measured numbers, per-run records and figure scripts are on
branch `campaign/cross-engine-1027` (via PR #27), folder `campaigns/2026-10-29-agent-replay/` (`README.md`, `RUNS.md`).

## Quick start

```sh
engines/common/run.sh sharar qwen3.8-flash-next cross-engine-agents   # start, check, replay at 1/8/16/32 agents, stop
less ~/engine-baselines/sharar/qwen3.8-flash-next/<UTC time>-<host>/report.txt
# or against a server that is already running, same requests:
uv run sharar-eval efficiency agent-replay --engine sharar --engine-version "main 7387a45 --throughput" \
  --url http://127.0.0.1:30004 --agents 1,8,16,32 --thinking off --tool-choice unset --out runs/agents/sharar-fn
```

Both run the **thinking-off setting** the cross-engine figures use. The flags' defaults are the **AA setting**
instead (below), so a run without `--thinking off --tool-choice unset` measures something else.

## What is replayed

`agentperf-default-v1` from aa-agentperf-local v0.3.4 (commit `1fe8056`), pinned with SHA-256 in
`evals/datasets.py` / `datasets.sha256` and fetched on first use: 8 recorded tasks, 168 model turns.

| Tasks | Turns | Prompt per turn | Shape |
|---|---|---|---|
| 3 SWE-bench repository fixes (django 15851, django 14500, sphinx 10614) | 28-40 each | 1.5k to 19k tokens | every request ends with a user message "Tool-call turn k of 100. Remaining turns: N." that the next turn deletes |
| 5 OpenClaw assistant tasks | 5-21 each | 6k to 56k tokens | append-only |

Each turn is the recorded conversation up to that point (system prompt, task, earlier assistant turns with their tool
calls, tool results) and the task's tool schemas. The engine's own reply is timed and discarded: the next turn sends
the recorded reply, so every engine receives the same prompts token for token. `--replay DIR` replays another
directory in the same layout (`manifest-arm64.json` and `traces/`), such as the no-note copy below; its hashes are
then not recorded (`run.json` `replay_sha256` is null), so record them yourself.

## One request

Each turn is one `POST /v1/chat/completions`:

| Field | Value |
|---|---|
| `messages` | the recorded conversation; its first message is headed by the copy's namespace: 32 random digits separated by spaces, a newline, `Performance replay cache namespace. Ignore the digits above.` and a blank line (a text part of its own if the content is a list) |
| `tools` | the task's tool schemas, when the turn has any |
| `max_tokens`, `ignore_eos` | the turn's recorded output length, and `true`: every engine generates exactly that many tokens, so all do the same work; a shorter reply counts as a failed turn |
| sampling | `temperature` 0.7, `top_p` 0.8, `top_k` 20, `min_p` 0 |
| `stream`, `stream_options` | `true`, `{"include_usage": true}`: prompt, cached and completion tokens come back in the last chunk |
| `chat_template_kwargs` (`--thinking`) | `server` (default): not sent, the template's default applies; `off` / `on`: `{"enable_thinking": false / true}` |
| `tool_choice` (`--tool-choice`) | `none` (default): `"none"`, sent with the tools only, so no tool-call parsing and a call streams as text; `unset`: not sent, the server's tool-call parser stays on |

Two named settings; label every result with one of them and its exact request fields:

- **AA setting** (the flags' defaults): thinking as the server serves it, `tool_choice: "none"`. This is
  agentperf-local's default plus `--tool-choice none`, as NVIDIA's DGX Spark AA-AgentPerf result ran.
- **Thinking-off setting** (`--thinking off --tool-choice unset`; suite `cross-engine-agents`):
  `chat_template_kwargs {"enable_thinking": false}`, tools sent without `tool_choice`. The cross-engine agent figures
  use this one, so every engine runs the same template whatever its default. (The published runs used sharar-eval
  `fdef4b6`, whose only setting this was; its request body is the same.)

Each turn record keeps `reasoning_chars`: non-zero means the engine thought on that turn.

## Agents, copies and levels

At N agents (`--agents`, default `1,4,8,16`), N client threads start together. Each takes the next task from a queue,
sends its turns one at a time (the next as soon as the previous reply's stream closes, or after the recorded tool time
with `--tool-mode recorded`), then takes another task: a closed loop with at most one request per agent in flight. The
queue holds ceil(N / 8) copies of the 8 tasks: one at N <= 8 (1 agent plays all 8 in turn), two at
16, four at 32 (one task per agent when N is a multiple of 8). Each copy has its own namespace, so copies share
nothing; the tasks of one copy share theirs (as in agentperf-local, tasks with the same system prompt can reuse it).
The levels run one after another on the same server, each with fresh namespaces (seeded by `--seed`, default the
clock); no cache is flushed between them.

- `--tool-mode recorded` sleeps each turn's recorded tool time (`simulated_tool_delay_ms_after`) after the turn; the
  agent waits with it. Default `none`: no gap.
- `--sessions S`: S conversations (the first S of ceil(S / 8) copies) are open at once and the N agents take turns
  from them: an agent sends the next turn of the conversation that has waited longest, then puts it at the back of the
  line. The level ends when all S are done. Open context then outgrows what the engine keeps in GPU memory, which is
  what a prefix-cache study needs.
- `--inflight K` (with `--duration`, default 900 s, and `--no-prime`): a steady state instead of a closed loop over a
  fixed task list. Each `--agents` value is then the number of open conversations, with at most K requests in flight;
  `--sessions` is ignored. Every open conversation has its own namespace; conversation i plays task i mod 8 from a
  random turn, first primed with the turn before it (1 output token, not measured, recorded as `agent_prime`) unless
  `--no-prime`. Each in-flight slot sends the next turn of the ready conversation that has waited longest (ready: its
  last turn ended, plus its recorded tool time with `--tool-mode recorded`); a conversation that ends is replaced by a
  new one at turn 0 of the next task. Turns start for `--duration` seconds per level; those in flight at the end finish
  and count. The agent count and the server's load are then separate knobs.

## What is reported per level

The table (`OUT/agents.csv`, printed as levels end; `evals/agent_replay.py summarize`). "Answered" turns are those
without an HTTP error that streamed at least one chunk with text, reasoning or a tool call.

| Column | Definition |
|---|---|
| wall s | from starting the level's agents to the last one finishing; at N >= 8 the slowest task sets it |
| tasks | tasks run (closed loop, `--sessions`); with `--inflight`, conversations that reached their last turn, including ones that started mid-task |
| tasks/h | tasks / wall s x 3600 |
| task p50 / p90 | per task, from its first turn sent to its last turn ended (tool time after the last turn not counted); blank with `--inflight` |
| agg tok/s | completion tokens of answered turns / wall s |
| decode | pooled over answered turns with no streamed tool call and at least 2 tokens: sum of (tokens - 1) / sum of first-to-last chunk time |
| TTFT p50 / p95 | from sending the request to its first chunk with text, reasoning or a tool call, over all answered turns |
| cached | `usage.prompt_tokens_details.cached_tokens` summed / prompt tokens summed, over answered turns; `-` if the server reports none |
| failed | turns with an HTTP error, no `[DONE]`, or fewer completion tokens than the target. Tool calls are not judged (generating past the model's stop cuts them by design) |

The CSV also has turn time p50 / p95 (request sent to stream closed), output tokens, and for information
`info_tool_args_unparsed` and `info_finish_not_length` (a full-length reply whose finish reason is not `length`, e.g.
`tool_calls`; agentperf-local would fail it, here it did the full work and passes).

## Records, resume, report

- `OUT/agents.jsonl`: one `agent_turn` record per turn (raw send, first-chunk, last-chunk and end times, usage, tool
  calls, `reasoning_chars`, faults; with `--inflight` also the open context when sent), one `agent_level` per level,
  and `agent_prime` records with `--inflight`.
- `OUT/agents.csv`: the table. `OUT/efficiency.json`: the options. `OUT/run.json`: engine, `--engine-version`, model,
  server description, replay hashes, seed, code hash (rewritten by each invocation). `OUT/checks.json`: the
  pre-flight `sharar-eval check` (a failure is printed and the replay runs anyway, so read it).
- A rerun into the same `OUT` with the same options (including the `--agents` list) skips finished levels and reruns a
  level cut short (its turns without a level record are dropped); other options are refused. If the server stops
  answering after a level, the rest are not run (exit 2).
- `sharar-eval efficiency report OUT` rewrites `agents.csv` from `agents.jsonl` and prints the table; for a `run.sh`
  run, `report OUT/suite`.

## Reproducing the published figures

**AA traces as published, 1 / 8 / 16 / 32 agents, thinking-off setting, no tool time**: one `run.sh` per engine and
model, each on its own Spark, with the setting the campaign used:

| Model | Hibana (then Sharar; `main 7387a45`, `--throughput`) | vLLM (eugr recipe) | TensorFold |
|---|---|---|---|
| Qwen3.8-27B | `qwen3.8-27b` | `qwen3.8-27b-d4` | `CONTEXT=agents`, `qwen3.8-27b-maxfit` |
| Qwen3.8-Flash-Next | `qwen3.8-flash-next` | `qwen3.8-flash-next-d2` | `TF_PARALLEL=16 TF_CONTEXT=65536 TF_RESERVE_GIB=9 TF_ENV=TF_FLASHNEXT_HEAD_KV=fp8`, `qwen3.8-flash-next-p32` |
| Nemotron-3.5-Bolt | `nemotron-3.5-bolt-w3` | `nemotron-3.5-bolt-m086` | `CONTEXT=agents`, `nemotron-3.5-bolt-maxfit` |
| Nemotron-3.5-Super | `nemotron-3.5-super` | `nemotron-3.5-super-d3` | `CONTEXT=agents`, `nemotron-3.5-super-maxfit` |

```sh
engines/common/run.sh vllm qwen3.8-27b-d4 cross-engine-agents
CONTEXT=agents engines/common/run.sh tensorfold qwen3.8-27b-maxfit cross-engine-agents
```

llama.cpp runs one server per agent count, sized to hold every agent's history (`engines/llamacpp/agents.sh
SETTING`, the same suite options; `REPLAY=` for the no-note copy): settings `qwen3.8-27b-dflash2`, `qwen3.8-flash-next-p`,
`nemotron-3.5-bolt-gguf`, `nemotron-3.5-super-gguf`. Flash-Next and Super run 32 agents on 16 slots (the rest wait).

What `run.sh` does for this suite: it reserves the Spark, starts the engine (`serve.sh up`), sends its sampling probe,
runs `efficiency suite --suite cross-engine-agents` (thinking off, `tool_choice` unset, `--tool-mode none`, the pinned
replay; `checks.json` first) and `report`, stops the engine. Its warm-decode knobs do nothing here: `PRIME_PASSES`,
`MIN_HIT` and `FLUSH_RESTART` apply to warm decode only, and `CONTEXT` reaches the suite as `--context` (unused by the
agent part; it is recorded in `suite.json`) but is what sizes a TensorFold `-maxfit` server (`agents`: 16 streams of
65,536 tokens). Two differences from the campaign's own runner (`queues/*/tf-agents2.sh` on the campaign branch):
`run.sh` does not save the engine's cache counters, and its `sampling-probe.json` uses warm decode's sampling
(thinking on at `xhigh`, `engines/common/defaults.env`), not the replay's, so it says nothing about this run's
thinking: read `reasoning_chars` in `agents.jsonl` (0 on every turn in the thinking-off setting).

**No-note replay, 1 / 8 / 16 agents**: the same traces with every SWE-bench "Tool-call turn k of 100" user message
removed (108 messages, one per SWE-bench turn), so every task only appends; targets, tools, tool delays and the
OpenClaw traces are unchanged. It is not shipped; build it from the pinned copy (fetched by the first agent-replay
run, under `$SHARAR_RESULTS/datasets/replay/`, default `~/.local/share/sharar`):

```sh
python3 -I - <<'PY'
import json, os, pathlib, shutil
root = pathlib.Path(os.environ.get("SHARAR_RESULTS") or pathlib.Path.home() / ".local/share/sharar")
dst = root / "replays/agentperf-default-v1-nonote"
shutil.copytree(root / "datasets/replay/agentperf-default-v1", dst)
for f in (dst / "traces").glob("*.jsonl"):
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    for r in rows:
        r["messages"] = [m for m in r["messages"]
                         if not (m["role"] == "user" and str(m.get("content")).startswith("Tool-call turn"))]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
PY
sha256sum ${SHARAR_RESULTS:-~/.local/share/sharar}/replays/agentperf-default-v1-nonote/traces/{django,sphinx}*
# 823fa50c3690be8f611cb25379e545afd27679ffdced761b91cf0894a0826a6d  django__django-14500.jsonl
# 735592d539f17905b0612382275ee3bb0817006719d8f3fbb01b46e98bb1ef23  django__django-15851.jsonl
# f7d28a6b1db9dbb9253db89f5126599c49484f4d8a5ace5663f1611c1d58d16c  sphinx-doc__sphinx-10614.jsonl
```

The suite cannot take `--replay`, so claim the Spark and start the engine yourself (`engines/ENGINE/serve.sh up
SETTING DIR`, with the same environment as above; it serves at the `URL` of its `versions.env` or setting) and run

```sh
uv run sharar-eval efficiency agent-replay --engine vllm --engine-version "<versions.env label>; setting <SETTING>" \
  --url $URL --agents 1,8,16 --thinking off --tool-choice unset --tool-mode none \
  --replay ~/.local/share/sharar/replays/agentperf-default-v1-nonote --out DIR/agents
engines/vllm/serve.sh down DIR
```

The campaign's no-note vLLM Super run used `nemotron-3.5-super-d3-m084` (memory 0.84) instead of `-d3`.

**TTFT and prefix-cache studies** on the same replay (Qwen3.8-Flash-Next): in the thinking-off setting
`--agents 32 --sessions 192` (192 conversations open, 32 agents), and in the AA setting (default `--thinking` and
`--tool-choice`) `--inflight 16 --duration 900 --tool-mode recorded --agents 16,32,64,128,192` (open agents; the 192
level was not finished in the published run).

## Before comparing engines

- **The server reports cached tokens.** vLLM needs `--enable-prompt-tokens-details` (`engines/vllm/serve.sh` passes
  it); without it the cached column is `-`.
- **Hibana 7387a45 holds a Qwen tool-call reply back to its end** (27B, Flash-Next; seen in the thinking-off setting), so its
  TTFT over all turns includes those whole replies. Compare TTFT on streamed text turns (more than 3 chunks, over
  50 ms from first to last chunk) as well. Those replies carry tool calls, so `decode` leaves them out.
- **How an engine resumes a conversation decides its cached share**, and the SWE-bench message that each next turn
  deletes is where engines differ: an engine that can only resume at the exact end of the previous request, or at a few
  saved points, re-reads most of every SWE-bench prompt. The no-note replay separates this from engine speed.
- **Stream and slot caps.** An engine with fewer streams than agents queues the rest: TensorFold serves 16 streams at
  size `agents`, so its 32-agent level measures queueing.
- **The box matters.** TensorFold sizes its prompt cache from the memory free at start, so it gets a different cache
  on a different Spark (Flash-Next: 8.8 GiB on spark3, 5.3 GiB on spark4). Run one engine per Spark, never next to a
  download (memory and page cache), and not on spark-911c (`AVOID_HOSTS`).
- **Label every result with its setting** (thinking-off or AA, with the request fields; tool time; replay copy) and the
  engine's setting name.
