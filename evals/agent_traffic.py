"""Record an agent's traffic to a server and check the tool calls in it (sharar#28: tool-call parsing under
real agents). evals.tbench runs one recording proxy per agent task and adds the check's counts to each item.

    sharar-eval agent-traffic proxy --bind 172.17.0.1 --port 31100 --upstream http://127.0.0.1:30004 --out DIR
    sharar-eval agent-traffic check DIR [DIR ...] [--show 5]        # DIR is searched recursively

- **Proxy.** Every request goes through unchanged and every response streams back unchanged. Each POST is
  kept as <seq>.req.json.gz (the body) and <seq>.resp.gz (the raw response bytes, SSE included), with one
  line per request in index.jsonl (path, status, time to first byte).
- **Check.** It reads Anthropic Messages and OpenAI chat responses, streamed or not, and counts:
  - leak: tool-call markup (<tool_call>, <function=, <parameter=) in text or thinking instead of a tool call
  - bad_json: tool arguments that do not parse as a JSON object
  - unknown: a tool name the request did not offer
  - missing: a required parameter (the tool's JSON schema) absent
  - stop: the stop or finish reason disagrees with the content (tool_use without a call, or the reverse)
  - http: a non-200 status, an error event or a response that does not parse
  A request still in flight when the agent was stopped (its timeout) counts as unfinished, not as a fault.
  These are the server's faults. The tool results the agent marked as errors (is_error, or a tool message
  starting with an error) are counted apart, grouped by their first line: mostly the model's own commands
  failing, which is not a parsing problem.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import http.client
import itertools
import json
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ANOMALIES = ("leak", "bad_json", "unknown", "missing", "stop", "http")
MARKUP = re.compile(r"<tool_call>|</tool_call>|<function=|<parameter=|</function>")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection", "content-length"}


# ---------------------------------------------------------------- proxy

class Recorder(ThreadingHTTPServer):
    """A pass-through proxy to `upstream` that records every POST into `out`. serve_forever() in a thread."""

    daemon_threads = True

    def __init__(self, bind: str, port: int, upstream: str, out: Path):
        u = urllib.parse.urlsplit(upstream)
        self.upstream = (u.hostname, u.port or 80)
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.seq = itertools.count(len(list(out.glob("*.req.json.gz"))))
        self.lock = threading.Lock()
        super().__init__((bind, port), _Handler)

    def index(self, rec: dict) -> None:
        with self.lock, open(self.out / "index.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Recorder

    def log_message(self, *a):
        pass

    def _forward(self):
        t0 = time.time()
        out = self.server.out
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        n = next(self.server.seq) if self.command == "POST" else None
        if n is not None:
            (out / f"{n:06d}.req.json.gz").write_bytes(gzip.compress(body))
        conn = http.client.HTTPConnection(*self.server.upstream, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
        headers["Content-Length"] = str(len(body))
        rec = {"seq": n, "t0": t0, "method": self.command, "path": self.path, "client": self.client_address[0]}
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001
            self.send_error(502, str(e))
            rec.update(error=str(e), t_end=time.time())
            self.server.index(rec)
            return
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        raw, first = [], None
        try:
            while chunk := resp.read1(65536):
                first = first or time.time()
                raw.append(chunk)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as e:
            rec["client_gone"] = str(e)
        if n is not None:
            (out / f"{n:06d}.resp.gz").write_bytes(gzip.compress(b"".join(raw)))
        rec.update(status=resp.status, ttfb=(first - t0) if first else None, t_end=time.time())
        self.server.index(rec)

    do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = _forward


# ---------------------------------------------------------------- responses

def sse_events(raw: bytes):
    for line in raw.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            data = line[5:].strip()
            if data and data != b"[DONE]":
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    yield {"_unparsed": data[:200].decode(errors="replace")}


def messages_response(raw: bytes, streamed: bool) -> dict:
    """Anthropic Messages -> {blocks: [{type, text | name and args_text}], stop, error}."""
    if not streamed:
        msg = json.loads(raw)
        if msg.get("type") == "error":
            return {"blocks": [], "stop": None, "error": msg.get("error")}
        blocks = []
        for b in msg.get("content", []):
            if b.get("type") == "tool_use":
                blocks.append({"type": "tool_use", "name": b.get("name"), "args_text": json.dumps(b.get("input"))})
            else:
                blocks.append({"type": b.get("type"), "text": b.get("text") or b.get("thinking") or ""})
        return {"blocks": blocks, "stop": msg.get("stop_reason"), "error": None}
    blocks, stop, error = {}, None, None
    for ev in sse_events(raw):
        et = ev.get("type")
        if et == "content_block_start":
            cb = ev.get("content_block", {})
            blocks[ev["index"]] = {"type": cb.get("type"), "name": cb.get("name"), "text": "", "args_text": ""}
        elif et == "content_block_delta":
            d, b = ev.get("delta", {}), blocks.setdefault(ev.get("index"), {"type": "?", "text": "", "args_text": ""})
            b["text"] += d.get("text") or d.get("thinking") or ""
            b["args_text"] += d.get("partial_json") or ""
        elif et == "message_delta":
            stop = ev.get("delta", {}).get("stop_reason") or stop
        elif et == "error" or "_unparsed" in ev:
            error = ev.get("error") or ev
    out = [blocks[k] for k in sorted(blocks, key=lambda k: (k is None, k))]
    for b in out:
        if b["type"] == "tool_use" and not b["args_text"]:
            b["args_text"] = "{}"
    return {"blocks": out, "stop": stop, "error": error}


def chat_response(raw: bytes, streamed: bool) -> dict:
    """OpenAI chat -> the same shape: one text block (reasoning and content) and the tool calls."""
    if not streamed:
        msg = json.loads(raw)
        if "error" in msg:
            return {"blocks": [], "stop": None, "error": msg["error"]}
        ch = (msg.get("choices") or [{}])[0]
        m = ch.get("message", {})
        blocks = [{"type": "text", "text": (m.get("reasoning_content") or "") + (m.get("content") or "")}]
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            blocks.append({"type": "tool_use", "name": f.get("name"), "args_text": f.get("arguments") or ""})
        return {"blocks": blocks, "stop": ch.get("finish_reason"), "error": None}
    text, calls, stop, error = "", {}, None, None
    for ev in sse_events(raw):
        if "_unparsed" in ev or "error" in ev:
            error = ev.get("error") or ev
            continue
        for ch in ev.get("choices") or []:
            d = ch.get("delta", {})
            text += (d.get("reasoning_content") or "") + (d.get("content") or "")
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"type": "tool_use", "name": None, "args_text": ""})
                f = tc.get("function", {})
                c["name"] = c["name"] or f.get("name")
                c["args_text"] += f.get("arguments") or ""
            stop = ch.get("finish_reason") or stop
    return {"blocks": [{"type": "text", "text": text}] + [calls[k] for k in sorted(calls)], "stop": stop, "error": error}


def offered_tools(req: dict) -> dict:
    out = {}
    for t in req.get("tools") or []:
        if "function" in t:  # OpenAI
            out[t["function"]["name"]] = t["function"].get("parameters") or {}
        elif "name" in t:
            out[t["name"]] = t.get("input_schema") or {}
    return out


def tool_errors(req: dict):
    """Tool results in the request that the agent marked as errors (only the last turn's, to count each once)."""
    tail = []
    for m in reversed(req.get("messages") or []):
        if m.get("role") == "assistant":
            break
        tail.append(m)
    for m in tail:
        if m.get("role") == "tool":  # OpenAI
            text = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
            if re.match(r"\s*(error|invalid|failed)", text or "", re.I):
                yield text
        elif isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result" and b.get("is_error"):
                    c = b.get("content")
                    yield c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))


def first_line(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "").strip()
    line = text.splitlines()[0] if text else ""
    return re.sub(r"\d+", "N", re.sub(r"/[\w./-]+", "PATH", line))[:120]


# ---------------------------------------------------------------- check

def check_response(req: dict, r: dict, name: str, counts: collections.Counter, examples: dict) -> None:
    if r["error"]:
        counts["http"] += 1
        examples["http"].append({"file": name, "error": r["error"]})
        return
    offered = offered_tools(req)
    calls = [b for b in r["blocks"] if b["type"] == "tool_use"]
    for b in r["blocks"]:
        if b["type"] != "tool_use" and (m := MARKUP.search(b.get("text") or "")):
            counts["leak"] += 1
            examples["leak"].append({"file": name, "block": b["type"], "around": b["text"][max(0, m.start() - 200):m.start() + 300]})
    for c in calls:
        try:
            args = json.loads(c["args_text"])
            assert isinstance(args, dict)
        except Exception:  # noqa: BLE001
            counts["bad_json"] += 1
            examples["bad_json"].append({"file": name, "name": c["name"], "args": c["args_text"][:500]})
            continue
        if offered and c["name"] not in offered:
            counts["unknown"] += 1
            examples["unknown"].append({"file": name, "name": c["name"], "offered": sorted(offered)[:30]})
            continue
        missing = [k for k in (offered.get(c["name"]) or {}).get("required", []) if k not in args]
        if missing:
            counts["missing"] += 1
            examples["missing"].append({"file": name, "name": c["name"], "missing": missing, "args": c["args_text"][:300]})
    if (r["stop"] in ("tool_use", "tool_calls")) != bool(calls) and r["stop"] not in ("max_tokens", "length"):
        counts["stop"] += 1
        examples["stop"].append({"file": name, "stop": r["stop"], "calls": len(calls)})
    counts["tool_calls"] += len(calls)


def check(d: Path) -> dict:
    """Counts, tool-error lines and examples for every recorded request under d."""
    counts, errs, examples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    paths = {}
    for idx in d.rglob("index.jsonl"):
        for rec in map(json.loads, idx.read_text().splitlines()):
            if rec.get("seq") is not None:
                paths[(idx.parent, rec["seq"])] = rec
    for rq in sorted(d.rglob("*.req.json.gz")):
        name = str(rq.relative_to(d))
        try:
            req = json.loads(gzip.decompress(rq.read_bytes()))
        except Exception:  # noqa: BLE001
            counts["unreadable"] += 1
            continue
        if not isinstance(req, dict) or "messages" not in req:
            continue
        counts["requests"] += 1
        for e in tool_errors(req):
            errs[first_line(e)] += 1
        rs = rq.with_name(rq.name.replace(".req.json.gz", ".resp.gz"))
        rec = paths.get((rq.parent, int(rq.name[:6]))) or {}
        if not rec:  # still in flight when the agent was stopped (a timeout): not the server's fault
            counts["unfinished"] += 1
            continue
        if not rs.exists() or rec.get("error"):
            counts["http"] += 1
            examples["http"].append({"file": name, "error": rec.get("error") or "no response"})
            continue
        if rec.get("status", 200) != 200:
            counts["http"] += 1
            examples["http"].append({"file": name, "error": f"status {rec['status']}"})
            continue
        raw = gzip.decompress(rs.read_bytes())
        parse = messages_response if rec.get("path", "").split("?")[0].endswith("/messages") else chat_response
        try:
            r = parse(raw, bool(req.get("stream")))
        except Exception as e:  # noqa: BLE001
            counts["http"] += 1
            examples["http"].append({"file": name, "error": f"unparseable response: {e}", "raw": raw[:300].decode(errors="replace")})
            continue
        check_response(req, r, name, counts, examples)
    return {"counts": {k: counts[k] for k in ("requests", "tool_calls", *ANOMALIES, "unfinished")},
            "tool_errors": dict(errs.most_common()), "examples": dict(examples)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proxy", help="record an agent's traffic")
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--upstream", required=True, help="the server, e.g. http://127.0.0.1:30004")
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check", help="tool-call health of recorded traffic")
    c.add_argument("dirs", nargs="+", type=Path)
    c.add_argument("--show", type=int, default=0, help="examples per anomaly and tool-error lines to print")
    args = ap.parse_args()
    if args.cmd == "proxy":
        Recorder(args.bind, args.port, args.upstream, args.out).serve_forever()
        return 0
    print("dir\t" + "\t".join(("requests", "tool_calls", *ANOMALIES, "unfinished")) + "\tagent-reported tool errors")
    for d in args.dirs:
        r = check(d)
        print(f"{d.name}\t" + "\t".join(str(v) for v in r["counts"].values()) + f"\t{sum(r['tool_errors'].values())}")
        for k, ex in r["examples"].items():
            for e in ex[:args.show]:
                print(f"  [{k}] {json.dumps(e)[:600]}")
        for line, n in list(r["tool_errors"].items())[:args.show]:
            print(f"  [tool error x{n}] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
