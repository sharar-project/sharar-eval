# Agent replay: how we measure engines on recorded coding-agent work

`sharar-eval efficiency agent-replay` replays Artificial Analysis's recorded agent tasks against any running
OpenAI-compatible server, at 1 to N agents at once, and reports tasks per hour, time to first token and how much of
each prompt the server served from its prefix cache. This page is the method; the engine side (how each engine is
started) is [`engines/`](../engines/README.md), and measured numbers live with their campaign (branch
`campaign/cross-engine-1027`, `campaigns/2026-10-29-agent-replay/`).

## What is replayed

`agentperf-default-v1` from aa-agentperf-local v0.3.4 (commit `1fe8056`), pinned with SHA-256 in
`evals/datasets.py` / `datasets.sha256` and fetched on first use: 8 recorded tasks, 168 model turns.

| Tasks | Turns | Prompt per turn | Shape |
|---|---|---|---|
| 3 SWE-bench repository fixes (django 15851, django 14500, sphinx 10614) | 28-40 each | 1.5k to 19k tokens | ends every request with a "Tool-call turn k of 100" message that the next turn deletes |
| 5 OpenClaw assistant tasks | 5-21 each | 6k to 56k tokens | append-only |

Each turn is the recorded conversation up to that point (system prompt, task, earlier assistant turns with their tool
calls, tool results) and the task's tool schemas. The engine's own reply is timed and discarded: the next turn sends
the recorded reply, so every engine receives the same prompts token for token. `--replay DIR` replays another copy of
the directory (for example one with the SWE-bench turn messages removed, which makes every task append-only).

## One request

| Field | Value |
|---|---|
| `messages`, `tools` | the recorded conversation and the task's tool schemas |
| `max_tokens`, `ignore_eos` | the turn's recorded output length, and `true`: every engine generates exactly that many tokens, so all do the same work; a shorter reply counts as a failed turn |
| sampling | temperature 0.7, top-p 0.8, top-k 20, min-p 0 |
| `stream`, `stream_options` | streamed, with `include_usage` (prompt, cached and completion tokens come back in the last chunk) |
| cache isolation | 32 random digits and a line of text head the first message of every conversation copy, so a run never reuses another run's cache |
| thinking (`--thinking`) | `server` (default): nothing sent, the template's default applies; `off` / `on`: `chat_template_kwargs.enable_thinking` |
| `tool_choice` (`--tool-choice`) | `none` (default): sent with the tools, so no tool-call parsing and a call streams as text; `unset`: not sent, the server's parser stays on |

Two named settings, used everywhere results are labelled:

- **AA setting** (the defaults): thinking as the server serves it, `tool_choice: "none"`. This is agentperf-local's
  default plus `--tool-choice none`, as NVIDIA's DGX Spark AA-AgentPerf result ran.
- **Thinking-off setting** (`--thinking off --tool-choice unset`, suite `cross-engine-agents`): thinking forced off,
  tools sent without `tool_choice`. The cross-engine agent figures use this one, so every engine runs the same template
  whatever its default.

## Agents, copies and levels

At N agents, N client threads start together. Each takes the next task from a queue, sends its turns one at a time
(the next one as soon as the previous reply's stream closes), then takes another task: a closed loop with one request
per agent in flight. At N <= 8 the 8 tasks run once (1 agent plays all 8 in turn); above 8 the queue holds
ceil(N / 8) copies (16 agents: 2 copies, 32: 4), one task per agent. Each copy has its own namespace, so copies share
nothing; tasks in the same copy share theirs (as in agentperf-local, tasks with the same system prompt can reuse it).
`--agents 1,8,16,32` runs one level per value, one after another on the same server, each with fresh namespaces.

- `--tool-mode recorded` sleeps each turn's recorded tool time before the next turn (default `none`: no gap).
- `--sessions S`: S conversations are open at once and the N agents take turns from them, each sending the next turn of
  the conversation that has waited longest. Open context then grows past what the engine keeps in GPU memory, which is
  what a prefix-cache study needs.
- `--inflight K --duration 900`: a steady state instead of a closed loop over a fixed task list. Each `--agents` value
  is then the number of open conversations, with at most K requests in flight, for `--duration` seconds per level; each
  open conversation starts at a random turn (primed with its previous turn unless `--no-prime`) and a finished one is
  replaced by a new one. The agent count and the server's load are then separate knobs.

## What is reported per level

| Column | Definition |
|---|---|
| wall s | from the level's first request to its last reply; at N >= 8 the slowest task sets it |
| tasks/h | tasks finished / wall time x 3600 |
| task p50 / p90 | per task, from its first turn sent to its last turn ended |
| agg tok/s | all generated tokens / wall time |
| decode | pooled per-request decode tok/s after the first token, on replies without a tool call |
| TTFT p50 / p95 | from the request to its first streamed chunk, over all turns |
| cached | `usage.prompt_tokens_details.cached_tokens` summed / prompt tokens summed |
| failed | turns with an HTTP or stream error or fewer tokens than the target |

Records: `OUT/agents.jsonl` (one record per turn with raw timestamps, usage, tool calls and reasoning characters, one
per level), `OUT/agents.csv` (the table), `OUT/run.json` (engine, `--engine-version`, server description, replay
hashes, seed), `OUT/checks.json` (the pre-flight `sharar-eval check`). A rerun into the same `OUT` skips finished
levels. `sharar-eval efficiency report OUT` prints the table again.

## Running it

```sh
# any engine already serving at --url
uv run sharar-eval efficiency agent-replay --engine sharar --engine-version "main 7387a45 --throughput" \
  --url http://127.0.0.1:30004 --agents 1,8,16,32 --thinking off --tool-choice unset --out runs/agents/sharar-fn
# or through the engine runner (start, check, replay, stop on a reserved Spark)
engines/common/run.sh sharar qwen3.8-flash-next cross-engine-agents
```

Before comparing engines, check:

- **The server reports cached tokens.** vLLM needs `--enable-prompt-tokens-details` (`engines/vllm/serve.sh` passes
  it); without it the cached column is empty.
- **Sharar 7387a45 holds a Qwen tool-call reply back to its end**, so its TTFT over all turns includes those whole
  replies. Compare TTFT on streamed text turns (more than 3 chunks, over 50 ms from first to last chunk) as well.
- **How an engine resumes a conversation decides its cached share**, and the SWE-bench message that each next turn
  deletes is where engines differ: an engine that can only resume at the exact end of the previous request, or at a few
  saved points, re-reads most of every SWE-bench prompt. Replaying with that message removed separates this from
  engine speed.
- **Stream and slot caps.** An engine with fewer streams than agents queues the rest (TensorFold serves 16); its level
  then measures queueing.
- **The box matters.** Engines that size caches from free memory get different caches on different Sparks. Run one
  engine per Spark, never next to a download (page cache), and not on the Spark that runs the desktop and Claude Code.
- **Label every result with its setting** (thinking-off or AA, tool time, replay copy) and the engine's setting name.
