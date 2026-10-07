import gzip
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evals import agent_traffic, suite

TOOLS = [{"name": "Bash", "input_schema": {"type": "object", "required": ["command"]}}]


def sse(*events) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)


def tool_use_stream(args: str) -> bytes:
    return sse({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "Bash"}},
               {"type": "content_block_delta", "index": 0, "delta": {"partial_json": args}},
               {"type": "message_delta", "delta": {"stop_reason": "tool_use"}})


def leak_stream() -> bytes:  # a turn that ends inside thinking with the tool call in it
    return sse({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
               {"type": "content_block_delta", "index": 0, "delta": {"thinking": "ok\n<tool_call>\n<function=Bash>"}},
               {"type": "message_delta", "delta": {"stop_reason": "end_turn"}})


def upstream(replies):
    """A fake server: each POST gets the next (status, body)."""
    replies = iter(replies)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status, body = next(replies)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def post(port: int, body: dict) -> bytes:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages?beta=true", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        return e.read()


def test_proxy_records_and_check_counts_faults(tmp_path):
    up = upstream([(200, tool_use_stream('{"command": "ls"}')), (200, tool_use_stream('{"cmd": "ls"}')),
                   (200, tool_use_stream('{"command": ')), (200, leak_stream()),
                   (400, b'{"type":"error","error":{"type":"invalid_request_error","message":"image"}}')])
    proxy = agent_traffic.Recorder("127.0.0.1", 0, f"http://127.0.0.1:{up.server_address[1]}", tmp_path)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    body = {"stream": True, "tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]}
    first = post(proxy.server_address[1], body)
    for _ in range(4):
        post(proxy.server_address[1], body)
    proxy.shutdown()
    assert first == tool_use_stream('{"command": "ls"}')  # passed through unchanged
    assert gzip.decompress((tmp_path / "000000.resp.gz").read_bytes()) == first
    # a request the agent's timeout cut off: recorded, never answered
    (tmp_path / "000005.req.json.gz").write_bytes(gzip.compress(json.dumps(body).encode()))
    c = agent_traffic.check(tmp_path)["counts"]
    assert c == {"requests": 6, "tool_calls": 3, "leak": 1, "bad_json": 1, "unknown": 0, "missing": 1, "stop": 0,
                 "http": 1, "unfinished": 1}


def test_chat_and_agent_reported_errors(tmp_path):
    req = {"tools": [{"type": "function", "function": {"name": "bash", "parameters": {"required": ["command"]}}}],
           "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""},
                        {"role": "tool", "content": "Error: no such file /app/a.txt"}]}
    resp = {"choices": [{"message": {"content": "", "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]},
                         "finish_reason": "tool_calls"}]}
    (tmp_path / "000000.req.json.gz").write_bytes(gzip.compress(json.dumps(req).encode()))
    (tmp_path / "000000.resp.gz").write_bytes(gzip.compress(json.dumps(resp).encode()))
    (tmp_path / "index.jsonl").write_text(json.dumps({"seq": 0, "path": "/v1/chat/completions", "status": 200}) + "\n")
    r = agent_traffic.check(tmp_path)
    assert (r["counts"]["unknown"], r["counts"]["tool_calls"]) == (1, 1)
    assert r["tool_errors"] == {"Error: no such file PATH": 1}


def test_show_sums_the_traffic_of_agent_items(tmp_path, capsys):
    clean = {"requests": 5, "tool_calls": 4, **{k: 0 for k in agent_traffic.ANOMALIES}}
    items = [{"index": 0, "correct": True, "completion_tokens": 9, "traffic": clean},
             {"index": 1, "correct": False, "completion_tokens": 3, "traffic": {**clean, "leak": 1}}]
    run = tmp_path / "r"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"label": "fast", "engine": "img", "tasks": {"tbench-claude-code": {"max_tokens": None}}}))
    (run / "tbench-claude-code").mkdir()
    (run / "tbench-claude-code" / "s00000.json").write_text(json.dumps({"task": "tbench-claude-code", "start": 0, "items": items}))
    assert suite.summary(run)["tasks"]["tbench-claude-code"]["traffic"]["leak"] == 1
    suite.cmd_show([str(run)])
    assert "tbench-claude-code: 10 requests, 8 tool calls, tool-call faults: leak 1" in capsys.readouterr().out
