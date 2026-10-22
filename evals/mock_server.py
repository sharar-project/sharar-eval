"""A GPU-free OpenAI chat server: streams one word per token, honours max_completion_tokens / max_tokens and
ignore_eos (otherwise it stops after 8 tokens), ends every other reply with a tool call when tools are offered, and
reports usage with cached prompt tokens. `sharar-eval bench prepare` sends AIPerf to it so AIPerf resolves datasets
with no engine serving; the tests exercise the suite end to end against it.

    python -m evals.mock_server PORT [LOG.jsonl]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = 'mock-model'
SEEN: set[str] = set()  # first-message prefixes already served, for cached_tokens


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    log_path = None

    def log_message(self, *args):
        pass

    def reply(self, code, body, kind='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/v1/models':
            return self.reply(200, {'object': 'list', 'data': [{'id': MODEL, 'object': 'model'}]})
        if self.path == '/get_server_info':
            return self.reply(200, {'engine': 'mock', 'requests_seen': len(SEEN)})
        if self.path in ('/health', '/metrics'):
            return self.reply(200, b'mock_up 1\n', 'text/plain')
        return self.reply(404, {'error': 'not found'})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/flush_cache':
            SEEN.clear()
            return self.reply(200, b'Cache flushed', 'text/plain')
        if self.path != '/v1/chat/completions':
            return self.reply(404, {'error': 'not found'})
        msgs = body['messages']
        prompt = sum(len(str(m.get('content') or '').split()) + 4 for m in msgs)
        key = json.dumps(msgs[:1])
        cached = prompt // 2 if key in SEEN else 0
        SEEN.add(key)
        cap = body.get('max_completion_tokens') or body.get('max_tokens') or 16
        n = cap if body.get('ignore_eos') else min(cap, 8)
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({'t': time.time(), 'messages': len(msgs), 'roles': [m['role'] for m in msgs],
                                    'first': str(msgs[0].get('content') or '')[:40],
                                    'last': hashlib.sha1(json.dumps(msgs[-1]).encode()).hexdigest()[:10],
                                    'tools': len(body.get('tools') or []), 'n': n,
                                    **{k: body.get(k) for k in ('max_completion_tokens', 'max_tokens', 'ignore_eos',
                                                                'temperature', 'top_p', 'top_k', 'chat_template_kwargs',
                                                                'accept_top_k')}}) + '\n')
        call = bool(body.get('tools')) and len(msgs) % 2 == 0
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()

        def send(obj):
            self.wfile.write(f'data: {json.dumps(obj)}\n\n'.encode())
            self.wfile.flush()

        base = {'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': MODEL}
        for i in range(n - (1 if call else 0)):
            send({**base, 'choices': [{'index': 0, 'delta': {'content': f'w{i} '}, 'finish_reason': None}]})
            time.sleep(0.001)
        if call:
            name = body['tools'][0]['function']['name']
            send({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': f'call_{len(msgs)}', 'type': 'function',
                  'function': {'name': name, 'arguments': '{"command": "ls"}'}}]}, 'finish_reason': None}]})
        send({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if call else ('length' if n == cap else 'stop')}]})
        send({**base, 'choices': [], 'usage': {'prompt_tokens': prompt, 'completion_tokens': n, 'total_tokens': prompt + n,
                                                'prompt_tokens_details': {'cached_tokens': cached}}})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        self.close_connection = True


def main():
    port = int(sys.argv[1])
    Handler.log_path = sys.argv[2] if len(sys.argv) > 2 else None
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
