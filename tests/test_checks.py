"""evals.checks against a fake OpenAI-compatible server whose faults each test switches on."""
import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evals import checks

NAMES = [c.__name__ for c in checks.CHECKS]
ALL_PASS = {name: "pass" for name in NAMES}


class Fake(BaseHTTPRequestHandler):
    """/v1/models and /v1/chat/completions, streamed (one SSE event per few characters) or not. A reply without
    tools is n words, one token each: max_tokens of them with ignore_eos, else 6 (1000 for an essay) cut at
    max_tokens. With tools it calls add_tags, and after a tool result it quotes the result's confirmation code.
    self.server.faults switches faults on; self.server.seen keeps every POST body (and its Authorization)."""

    def log_message(self, *a):
        pass

    def send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/v1/models":
            return self.send(404, {"error": {"message": "not found"}})
        if status := self.server.faults.get("models_status"):
            return self.send(status, {"error": {"message": "model still loading"}})
        self.send(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

    def answer(self, body):
        """(message, finish_reason, completion tokens)."""
        f, msgs = self.server.faults, body["messages"]
        if body.get("tools"):
            if msgs[-1]["role"] == "tool" and not f.get("call_again"):
                code = json.loads(msgs[-1]["content"])["confirmation"]
                return {"role": "assistant", "content": f"Done: the confirmation code is {code}."}, "stop", 9
            tags = "urgent, finance, q3" if f.get("tags_as_string") else ["urgent", "finance", "q3"]
            call = {"id": "call_fake", "type": "function",
                    "function": {"name": "add_tags", "arguments": json.dumps({"doc_id": 42, "tags": tags})}}
            return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls", 20
        cap, natural = body["max_tokens"], 1000 if "essay" in msgs[-1]["content"] else 6
        if body.get("ignore_eos") and not f.get("ignore_eos_ignored"):
            n = cap
        else:
            n = natural if f.get("ignore_max_tokens") else min(cap, natural)
        message = {"role": "assistant", "content": " ".join(f"w{i}" for i in range(n))}
        if f.get("leak") == "<think>":
            message["content"] = "<think>17 times 23 is 391.</think>" + message["content"]
        elif f.get("leak"):
            message[f["leak"]] = "17 times 23 is 391."
        return message, "length" if n == cap else "stop", n

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append({"auth": self.headers.get("Authorization"), **body})
        f = self.server.faults
        if rejected := [k for k in f.get("reject", ()) if k in body]:
            return self.send(400, {"object": "error", "message": f"unknown field {rejected[0]}", "code": 400})
        message, finish, n = self.answer(body)
        usage = {"prompt_tokens": 30, "completion_tokens": n, "total_tokens": 30 + n}
        if not body.get("stream"):
            return self.send(200, {"id": "x", "object": "chat.completion", "model": body["model"],
                                   "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                   **({} if f.get("no_usage") else {"usage": usage})})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()  # HTTP/1.0: the stream ends when the connection closes

        def event(choices, **extra):
            chunk = {"id": "x", "object": "chat.completion.chunk", "choices": choices, **extra}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

        if f.get("garbage_stream"):
            return self.wfile.write(b"data: {not json\n\n")  # and nothing after: the client stops reading here
        event([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}], usage=None)
        for key in ("reasoning_content", "reasoning", "content"):
            text = message.get(key) or ""
            for i in range(0, len(text), 4):
                event([{"index": 0, "delta": {key: text[i:i + 4]}, "finish_reason": None}], usage=None)
        for i, call in enumerate(message.get("tool_calls") or []):
            event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                                          "function": {"name": "add_tags", "arguments": ""}}]}}])
            args = call["function"]["arguments"]
            for j in range(0, len(args), 5):
                event([{"index": 0, "delta": {"tool_calls": [{"index": i, "function": {"arguments": args[j:j + 5]}}]}}])
        event([{"index": 0, "delta": {}, "finish_reason": finish}], usage=None)
        if body.get("stream_options", {}).get("include_usage") and not f.get("no_stream_usage"):
            event([], usage=usage)
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def fake():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    server.daemon_threads = True
    server.faults, server.seen = {}, []
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def statuses(result):
    return {c["name"]: c["status"] for c in result["checks"]}


def detail(result, name):
    return next(c["detail"] for c in result["checks"] if c["name"] == name)


def closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_well_behaved_server_passes_everything(fake):
    result = checks.run(fake.url + "/", api_key="secret")
    assert result["model"] == "fake-model" and result["passed"]
    assert statuses(result) == ALL_PASS and list(statuses(result)) == NAMES
    assert len(fake.seen) == len(NAMES)
    for body in fake.seen:
        assert body["auth"] == "Bearer secret" and body["model"] == "fake-model"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["temperature"] == (0.7 if "top_k" in body else 0)
    assert detail(result, "sampling_params").startswith("accepted")
    follow = fake.seen[-1]["messages"]  # tool_result sends the server's own call back with the tool's result
    assert follow[-2]["tool_calls"][0]["id"] == follow[-1]["tool_call_id"] == "call_fake"
    assert checks.CODE in follow[-1]["content"] and checks.CODE not in json.dumps(fake.seen[:-1])


@pytest.mark.parametrize("leak", ["reasoning_content", "reasoning", "<think>"])
def test_leaked_reasoning_fails_thinking_off(fake, leak):
    fake.faults["leak"] = leak
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "thinking_off": "fail"} and not result["passed"]
    assert "391" in detail(result, "thinking_off")


def test_tags_as_a_string_fail_the_tool_calls_and_tool_result_still_runs(fake):
    fake.faults["tags_as_string"] = True
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "tool_call": "fail", "tool_stream": "fail"}
    assert "tags is \"urgent, finance, q3\" (str), not a JSON array" in detail(result, "tool_call")
    assert detail(result, "tool_stream").startswith("streamed: tags is")
    follow = fake.seen[-1]["messages"]
    assert follow[-2]["tool_calls"][0]["id"] == "call_0" and "written call" in detail(result, "tool_result")


def test_ignoring_ignore_eos_fails_ignore_eos(fake):
    fake.faults["ignore_eos_ignored"] = True
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "ignore_eos": "fail"}
    assert detail(result, "ignore_eos").startswith("completion_tokens 6, not 64 (finish_reason 'stop')")


def test_a_stream_without_usage_fails_stream_usage(fake):
    fake.faults["no_stream_usage"] = True
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "stream_usage": "fail"}
    assert "no chunk carried usage" in detail(result, "stream_usage")


def test_ignoring_max_tokens_fails_max_tokens(fake):
    fake.faults["ignore_max_tokens"] = True
    result = checks.run(fake.url)
    assert statuses(result)["max_tokens"] == "fail" and "completion_tokens 1000" in detail(result, "max_tokens")


def test_another_call_after_the_tool_result_fails_tool_result(fake):
    fake.faults["call_again"] = True
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "tool_result": "fail"}
    assert "another call add_tags" in detail(result, "tool_result")


def test_without_usage_token_counts_are_unchecked_not_passed(fake):
    fake.faults["no_usage"] = True
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "max_tokens": "unchecked", "ignore_eos": "unchecked"}
    assert not result["passed"]


def test_errors_fail_only_their_own_check(fake):
    fake.faults.update(reject=["top_k"], garbage_stream=True)
    result = checks.run(fake.url)
    assert statuses(result) == {**ALL_PASS, "sampling_params": "fail", "stream_usage": "fail", "tool_stream": "fail"}
    assert detail(result, "sampling_params") == "HTTP 400: unknown field top_k"
    assert detail(result, "stream_usage").startswith("the reply is not JSON")


@pytest.mark.parametrize("change, why", [
    ({}, None),
    ({"arguments": json.dumps({"doc_id": 42, "tags": [" Urgent", "FINANCE ", "q3", "extra"]})}, None),
    ({"arguments": json.dumps({"doc_id": "42", "tags": ["urgent", "finance", "q3"]})}, "doc_id is '42' (str)"),
    ({"arguments": json.dumps({"doc_id": True, "tags": ["urgent", "finance", "q3"]})}, "doc_id is True (bool)"),
    ({"arguments": json.dumps({"doc_id": 42.0, "tags": ["urgent", "finance", "q3"]})}, "doc_id is 42.0 (float)"),
    ({"arguments": json.dumps({"doc_id": 42, "tags": ["urgent", "finance"]})}, "miss ['q3']"),
    ({"arguments": json.dumps({"doc_id": 42, "tags": ["urgent", "finance", 3]})}, "not a JSON array of strings"),
    ({"arguments": '{"doc_id": 42, "tags": ['}, "arguments are not JSON"),
    ({"arguments": {"doc_id": 42, "tags": ["urgent", "finance", "q3"]}}, "arguments are a dict, not a JSON string"),
    ({"name": "tag"}, "called 'tag', not add_tags"),
    ({"finish": "stop"}, "finish_reason 'stop', not 'tool_calls'"),
    ({"tool_calls": []}, "no tool call"),
])
def test_call_problem_wants_typed_arguments(change, why):
    call = {"id": "c1", "name": "add_tags", "arguments": json.dumps({"doc_id": 42, "tags": sorted(checks.TAGS)})}
    reply = {"content": "", "tool_calls": [call], "finish": "tool_calls"}
    for k, v in change.items():
        (call if k in call else reply)[k] = v
    problem = checks.call_problem(reply)
    assert problem is None if why is None else why in problem, problem


def test_an_unreachable_url_cannot_run(capsys):
    url = f"http://127.0.0.1:{closed_port()}"
    with pytest.raises(checks.CannotRun, match="cannot reach the server"):
        checks.run(url, model="given")
    assert checks.main(["--url", url]) == 2
    assert "checks could not run" in capsys.readouterr().err


def test_the_command_prints_a_line_per_check_and_writes_out(fake, tmp_path, capsys):
    fake.faults["ignore_eos_ignored"] = True
    out = tmp_path / "pre" / "checks.json"
    assert checks.main(["--url", fake.url, "--model", "served-name", "--out", str(out)]) == 1
    lines = capsys.readouterr().out.splitlines()
    want = [f"{'FAIL' if name == 'ignore_eos' else 'PASS'} {name}" for name in NAMES]
    assert [text.split(":")[0] for text in lines[:-1]] == want
    assert lines[-1] == f"7/8 checks passed, 1 failed, 0 unchecked (model served-name at {fake.url})"
    saved = json.loads(out.read_text())
    assert saved["url"] == fake.url and saved["model"] == "served-name" and saved["passed"] is False
    assert {b["model"] for b in fake.seen} == {"served-name"}


@pytest.mark.parametrize("faults, code", [({}, 0), ({"no_stream_usage": True}, 1), ({"models_status": 503}, 2),
                                         (None, 2)])
def test_the_command_exit_codes(fake, faults, code):
    fake.faults.update(faults or {})
    url = fake.url if faults is not None else f"http://127.0.0.1:{closed_port()}"
    p = subprocess.run([sys.executable, "-m", "evals.checks", "--url", url], cwd=Path(__file__).parent.parent,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == code, p.stdout + p.stderr
    if faults and "models_status" in faults:
        assert "HTTP 503: model still loading" in p.stderr and not p.stdout
