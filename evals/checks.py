"""Pre-flight checks of an OpenAI-compatible chat server: the request features the benchmarks rely on, which an
engine may accept and silently ignore. Run them before benchmarking a server; evals.bench calls run() before its cells.

    sharar-eval check --url http://127.0.0.1:30004 [--model M] [--api-key K] [--out checks.json]
    python -m evals.checks --url http://127.0.0.1:30004

Every request is greedy (temperature 0; sampling_params excepted) with thinking off (chat_template_kwargs
{"enable_thinking": false}). Each check is PASS, FAIL or UNCHECKED (could not be verified, never counted as a pass):
  thinking_off     no reasoning_content or reasoning field, and no <think> or </think> in the content
  stream_usage     streamed with stream_options.include_usage, a chunk carries usage with integer token counts
  max_tokens       a long essay with max_tokens 16 stops at 16 completion tokens or fewer, finish_reason length
  ignore_eos       "Say hi." with ignore_eos and max_tokens 64 gives exactly 64 completion tokens (cells that force exact lengths)
  sampling_params  temperature 0.7, top_p 0.95, top_k 20 and min_p 0 are accepted (honouring them is not observable)
  tool_call        the reply calls add_tags with doc_id the integer 42 and tags a JSON array of strings holding
                   urgent, finance and q3, finish_reason tool_calls
  tool_stream      the same request streamed: the argument deltas joined per index give the same typed arguments
  tool_result      after the call and a role "tool" result, the reply is text quoting the code only the result held
                   (after a written call when tool_call failed, so this still runs)
Exit 0 every check passed, 1 any failed or is unchecked, 2 /v1/models could not be read (no check ran).
The tool checks follow tools/toolcheck.py of MiaAI-Lab's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold (MIT).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 600  # seconds per request: a busy or cold server can be slow, and a slow answer is not a failed check
TOOLS = [{"type": "function", "function": {
    "name": "add_tags", "description": "Attach tags to a document.",
    "parameters": {"type": "object", "required": ["doc_id", "tags"], "properties": {
        "doc_id": {"type": "integer", "description": "The document id."},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "The tags to attach."}}}}}]
ASK = [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool, "
                                   "then tell me the confirmation code the tool returns."}]
TAGS = {"urgent", "finance", "q3"}
CODE = "TAGGED-58213-KESTREL"


class CannotRun(Exception):
    """The server could not be reached, or /v1/models answered an error or no model: no check ran."""


class BadReply(Exception):
    """A reply that does not have the API's shape; the message is one line."""


def short(text, n=160):
    """text on one line, cut to n characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


def reason(exc):
    """One line for what went wrong in a request."""
    if isinstance(exc, urllib.error.HTTPError):
        text = exc.read().decode("utf-8", "replace")
        try:  # OpenAI's {"error": {"message"}}, or vLLM's and SGLang's {"object": "error", "message"}
            body = json.loads(text)
            err = body.get("error", body) if isinstance(body, dict) else body
            text = err.get("message", text) if isinstance(err, dict) else str(err)
        except ValueError:
            pass
        return f"HTTP {exc.code}: {short(text)}"
    if isinstance(exc, urllib.error.URLError):
        return f"cannot reach the server ({short(exc.reason)})"
    if isinstance(exc, TimeoutError):
        return f"no reply within {TIMEOUT} s"
    if isinstance(exc, json.JSONDecodeError):
        return f"the reply is not JSON ({exc.msg})"
    if isinstance(exc, BadReply):
        return str(exc)
    return short(f"{type(exc).__name__}: {exc}")


def tokens(usage, key="completion_tokens"):
    """usage[key] when it is an integer (not a bool), else None."""
    v = usage.get(key) if isinstance(usage, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def read_whole(reply):
    """A non-streamed completion as {content, reasoning, tool_calls [{id, name, arguments}], finish, usage}."""
    try:
        choice = reply["choices"][0]
        msg = choice["message"]
        calls = [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
                  "arguments": (c.get("function") or {}).get("arguments")} for c in msg.get("tool_calls") or []]
    except (KeyError, IndexError, TypeError, AttributeError):
        raise BadReply(f"no choices[0].message in the reply: {short(json.dumps(reply), 120)}") from None
    return {"content": msg.get("content") or "", "tool_calls": calls, "finish": choice.get("finish_reason"),
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "usage": reply.get("usage")}


def read_stream(lines):
    """A streamed completion (SSE data: lines) in read_whole's shape: deltas joined, tool calls joined per index, and
    the last usage a chunk carried, as the benchmarks read it."""
    content, reasoning, calls, finish, usage, events = [], [], {}, None, None, 0
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise BadReply(f"the stream reported an error: {short(json.dumps(chunk['error']), 120)}")
        events += 1
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for c in delta.get("tool_calls") or []:
                slot = calls.setdefault(c.get("index", 0), {"id": None, "name": "", "arguments": ""})
                fn = c.get("function") or {}
                slot["id"] = c.get("id") or slot["id"]
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if not events:
        raise BadReply("the stream carried no events")
    return {"content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)], "finish": finish,
            "reasoning": "".join(reasoning), "usage": usage}


class Server:
    """The server under check, and what one check leaves for a later one (tool_call's call, for tool_result)."""

    def __init__(self, url, api_key=None):
        self.url, self.api_key, self.model, self.call = url.rstrip("/"), api_key, None, None

    def open(self, path, body=None):
        """The response to a GET of path, or to a POST of body as JSON."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(urllib.request.Request(self.url + path, data, headers), timeout=TIMEOUT)

    def chat(self, messages, max_tokens, stream=False, **extra):
        """One chat request, greedy and thinking off unless extra overrides it, as read_whole's dict."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, **extra}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        with self.open("/v1/chat/completions", body) as r:
            return read_stream(r) if stream else read_whole(json.load(r))


def thinking_off(s):
    r = s.chat([{"role": "user", "content": "What is 17 * 23?"}], 200)
    if r["reasoning"]:
        return "fail", f"reasoning came back with thinking off: {short(r['reasoning'], 80)!r}"
    if "<think>" in r["content"] or "</think>" in r["content"]:  # the opening tag may sit in the prompt
        return "fail", f"a think tag in the content: {short(r['content'], 80)!r}"
    if not r["content"].strip():
        return "fail", f"an empty reply (finish_reason {r['finish']!r})"
    return "pass", f"no reasoning; content {short(r['content'], 60)!r}"


def stream_usage(s):
    r = s.chat([{"role": "user", "content": "Count from 1 to 10."}], 64, stream=True)
    if r["usage"] is None:
        return "fail", "no chunk carried usage: stream_options.include_usage ignored"
    p, n = tokens(r["usage"], "prompt_tokens"), tokens(r["usage"])
    if p is None or n is None:
        return "fail", f"usage lacks integer prompt_tokens and completion_tokens: {short(json.dumps(r['usage']), 100)}"
    return "pass", f"usage in the stream: prompt_tokens {p}, completion_tokens {n}"


def max_tokens(s):
    r = s.chat([{"role": "user", "content": "Write a long essay on the history of the printing press."}], 16)
    n = tokens(r["usage"])
    if n is not None and n > 16:
        return "fail", f"completion_tokens {n} with max_tokens 16 (finish_reason {r['finish']!r})"
    if r["finish"] != "length":
        return "fail", f"finish_reason {r['finish']!r}, not 'length' (completion_tokens {n})"
    if n is None:
        return "unchecked", "finish_reason length, but the reply has no integer usage.completion_tokens"
    return "pass", f"completion_tokens {n} of max_tokens 16, finish_reason length"


def ignore_eos(s):
    r = s.chat([{"role": "user", "content": "Say hi."}], 64, ignore_eos=True)
    n = tokens(r["usage"])
    if n is None:
        return "unchecked", f"the reply has no integer usage.completion_tokens (finish_reason {r['finish']!r})"
    if n != 64:
        hint = ": stopped at end of sequence, ignore_eos not honoured" if n < 64 else ""
        return "fail", f"completion_tokens {n}, not 64 (finish_reason {r['finish']!r}){hint}"
    return "pass", "completion_tokens 64 of max_tokens 64"


def sampling_params(s):
    r = s.chat([{"role": "user", "content": "Name a colour."}], 32, temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)
    if not r["content"].strip() and not r["tool_calls"]:
        return "fail", f"HTTP 200 but an empty reply (finish_reason {r['finish']!r})"
    return "pass", "accepted: temperature 0.7, top_p 0.95, top_k 20, min_p 0.0 (honouring them is not observable)"


def call_problem(r):
    """Why a reply is not the add_tags call the tool checks want, or None when it is."""
    if not r["tool_calls"]:
        return f"no tool call (finish_reason {r['finish']!r}, content {short(r['content'], 80)!r})"
    c = r["tool_calls"][0]
    if c["name"] != "add_tags":
        return f"called {c['name']!r}, not add_tags"
    if not isinstance(c["arguments"], str):  # OpenAI clients json.loads a string
        return f"arguments are a {type(c['arguments']).__name__}, not a JSON string"
    try:
        args = json.loads(c["arguments"])
    except ValueError:
        return f"arguments are not JSON: {short(c['arguments'], 80)!r}"
    if not isinstance(args, dict):
        return f"arguments are not a JSON object: {short(c['arguments'], 80)}"
    doc, tags = args.get("doc_id"), args.get("tags")
    if isinstance(doc, bool) or not isinstance(doc, int) or doc != 42:
        return f"doc_id is {doc!r} ({type(doc).__name__}), not the integer 42"
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return f"tags is {short(json.dumps(tags), 80)} ({type(tags).__name__}), not a JSON array of strings"
    if missing := TAGS - {t.strip().lower() for t in tags}:
        return f"tags {tags} miss {sorted(missing)}"
    if r["finish"] != "tool_calls":
        return f"typed arguments, but finish_reason {r['finish']!r}, not 'tool_calls'"
    return None


def tool_call(s):
    r = s.chat(ASK, 512, tools=TOOLS)
    if problem := call_problem(r):
        return "fail", problem
    s.call = r["content"], r["tool_calls"][0]
    return "pass", f"add_tags({short(r['tool_calls'][0]['arguments'], 80)}), finish_reason tool_calls"


def tool_stream(s):
    r = s.chat(ASK, 512, stream=True, tools=TOOLS)
    if problem := call_problem(r):
        return "fail", f"streamed: {problem}"
    return "pass", f"the joined deltas give add_tags({short(r['tool_calls'][0]['arguments'], 80)}), finish tool_calls"


def tool_result(s):
    used = "the server's own call" if s.call else "a written call (tool_call failed)"
    content, call = s.call or ("", {"id": "call_0", "arguments": json.dumps({"doc_id": 42, "tags": sorted(TAGS)})})
    call_id = call["id"] or "call_0"
    messages = ASK + [
        {"role": "assistant", "content": content, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "add_tags", "arguments": call["arguments"]}}]},
        {"role": "tool", "tool_call_id": call_id, "name": "add_tags",
         "content": json.dumps({"ok": True, "doc_id": 42, "tags_added": 3, "confirmation": CODE})}]
    r = s.chat(messages, 256, tools=TOOLS)
    if r["tool_calls"]:
        c = r["tool_calls"][0]
        again = f"{c['name']}({short(c['arguments'], 60)})"
        return "fail", f"after {used}, answered the tool result with another call {again} instead of text"
    if CODE.lower() not in r["content"].lower():
        return "fail", f"after {used}, the reply lacks {CODE}: {short(r['content'], 100)!r}"
    return "pass", f"after {used} and its result, the reply quotes {CODE}"


# In run order (tool_result after tool_call, whose call it reuses). Each takes the Server and returns (status, detail);
# the module docstring says what each wants, and run() makes an exception a check raises its fail.
CHECKS = (thinking_off, stream_usage, max_tokens, ignore_eos, sampling_params, tool_call, tool_stream, tool_result)


def run(url, model=None, api_key=None) -> dict:
    """Every check against the server at url (its base URL, no /v1), in order.

    Returns {"model", "checks": [{"name", "status": "pass" | "fail" | "unchecked", "detail"}], "passed"}, passed
    only when every check passed. /v1/models is read first even with model given, and raises CannotRun when it fails:
    a server that cannot be reached is told apart from checks that fail. A check's own error is its fail."""
    s = Server(url, api_key)
    try:
        with s.open("/v1/models") as r:
            listing = json.load(r)
    except Exception as exc:  # noqa: BLE001  any failure here means no check can run
        raise CannotRun(f"{s.url}/v1/models: {reason(exc)}") from None
    try:
        s.model = model or listing["data"][0]["id"]
    except (KeyError, IndexError, TypeError):
        raise CannotRun(f"{s.url}/v1/models lists no model: {short(json.dumps(listing), 120)}") from None
    results = []
    for check in CHECKS:
        try:
            status, detail = check(s)
        except Exception as exc:  # noqa: BLE001  one check's error is its fail, never the others'
            status, detail = "fail", reason(exc)
        results.append({"name": check.__name__, "status": status, "detail": detail})
    return {"model": s.model, "checks": results, "passed": all(c["status"] == "pass" for c in results)}


def line(check):
    """A check as the command prints it: PASS name: detail."""
    return f"{check['status'].upper()} {check['name']}: {check['detail']}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the server's base URL, e.g. http://127.0.0.1:30004")
    parser.add_argument("--model", help="served model name (default: the first of /v1/models)")
    parser.add_argument("--api-key", help="sent as a bearer token")
    parser.add_argument("--out", type=Path, help="also write the result here as JSON")
    args = parser.parse_args(argv)
    try:
        result = run(args.url, args.model, args.api_key)
    except CannotRun as exc:
        print(f"checks could not run: {exc}", file=sys.stderr)
        return 2
    for c in result["checks"]:
        print(line(c))
    count = {k: sum(c["status"] == k for c in result["checks"]) for k in ("pass", "fail", "unchecked")}
    print(f"{count['pass']}/{len(result['checks'])} checks passed, {count['fail']} failed, "
          f"{count['unchecked']} unchecked (model {result['model']} at {args.url})", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"url": args.url, **result}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
