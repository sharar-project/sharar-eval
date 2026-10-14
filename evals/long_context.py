"""Long-context serving check against a running server: prefill time, needle
retrieval and decode speed per context length.

For each length the prompt is source code (a fixed corpus, a different slice
per length so lengths never share a cached prefix) with one fact inserted at
a relative depth and a question about it at the end. Depths run deepest first,
so each later request reuses the cached prefix up to its own needle. A final
request reuses the whole context and asks for a long walkthrough with EOS
ignored, once exact and once with the server's relaxed default, to measure
decode speed and speculation at that context.

    sharar-eval long-context --url http://127.0.0.1:30002 --log out.jsonl
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

from evals.datasets import ROOT

DEFAULT_CORPUS = ROOT / 'long_context' / 'corpus.txt'  # pinned: sharar-eval datasets fetch
DEFAULT_TOKENIZER = ROOT / 'long_context' / 'tokenizer.json'
FILES = ['long_context/corpus.txt', 'long_context/tokenizer.json']


def corpus_ids(root, tok, need):
    """Token ids of the corpus file (or, for a directory, its *.py files in path order)."""
    root = Path(root)
    if root.is_file():
        ids = tok.encode(root.read_text()).ids
    else:
        ids = []
        for path in sorted(root.rglob('*.py')):
            text = path.read_text(errors='replace')
            ids += tok.encode(f'\n# ===== {path.relative_to(root)} =====\n{text}').ids
            if len(ids) >= need:
                break
    if len(ids) < need:
        raise SystemExit(f'corpus has {len(ids)} tokens, need {need}')
    return ids


def post(url, body, timeout=3600):
    """Stream a chat completion; return (first-token time, last-token time, text, usage, finish)."""
    req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    start, first, last, text, usage, finish = time.monotonic(), None, None, [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                piece = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
                if piece:
                    now = time.monotonic()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
                finish = choice.get('finish_reason') or finish
    return (first or time.monotonic()) - start, (last or time.monotonic()) - start, ''.join(text), usage, finish


def server_stats(url):
    with urllib.request.urlopen(url + '/get_server_info', timeout=30) as r:
        info = json.load(r)
    return info.get('spec_stats') or info.get('speculative_stats') or {}, info


def run(url, log, lengths=(10240, 32768, 65536, 131072, 262144), depths=(0.9, 0.5, 0.1), output_tokens=512,
        corpus=DEFAULT_CORPUS, tokenizer=DEFAULT_TOKENIZER, seed=7):
    """Append one row per needle and per decode measurement to `log` (JSONL); return the rows."""
    url = url.rstrip('/')
    tok = Tokenizer.from_file(str(tokenizer))
    margin = 256 + output_tokens  # template, question and retokenization drift
    ids = corpus_ids(corpus, tok, sum(lengths))
    rng = random.Random(seed)
    rows = []
    out = open(log, 'a')

    def emit(row):
        rows.append(row)
        out.write(json.dumps(row) + '\n')
        out.flush()
        print(json.dumps(row), flush=True)

    _, info = server_stats(url)
    served = info.get('context_length')
    emit({'kind': 'server', 'context_length': served, 'kv_pages': info.get('kv_pages'),
          'state_slots': info.get('state_slots'), 'ledger_gib': info.get('ledger_gib')})
    offset = 0
    for length in lengths:
        if served is not None and length > served:
            print(f'skip {length}: server context is {served}', flush=True)
            continue
        body_ids = ids[offset:offset + length - margin]
        offset += length
        name, code = f'{rng.choice(["amber", "cobalt", "saffron", "vermilion"])}-{rng.randrange(100, 999)}', f'{rng.randrange(10**7, 10**8)}'
        needle = f'\n\n# One of the special magic numbers for {name} is: {code}.\n\n'
        question = f'\n\nWhat is the special magic number for {name} mentioned in the code above? Reply with the number only.'
        filler = None
        for depth in depths:
            cut = int(len(body_ids) * depth)
            text = tok.decode(body_ids[:cut]) + needle + tok.decode(body_ids[cut:])
            filler = text
            ttft, _, answer, usage, finish = post(url, {
                'model': 'x', 'messages': [{'role': 'user', 'content': text + question}], 'max_tokens': 24,
                'temperature': 0, 'accept_top_k': 1, 'trace': False, 'stream': True,
                'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False}})
            emit({'kind': 'needle', 'length': length, 'depth': depth, 'prompt_tokens': (usage or {}).get('prompt_tokens'),
                  'cached_tokens': ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens'),
                  'ttft_s': round(ttft, 3), 'answer': answer.strip(), 'correct': code in answer, 'finish': finish})
        for mode, extra in (('exact', {'accept_top_k': 1}), ('relaxed-default', {})):
            before, _ = server_stats(url)
            ttft, last, text, usage, finish = post(url, {
                'model': 'x', 'messages': [{'role': 'user', 'content': filler + '\n\nWrite a detailed technical walkthrough of the code above, file by file.'}],
                'max_tokens': output_tokens, 'ignore_eos': True, 'temperature': 0, 'trace': False, 'stream': True,
                'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False}, **extra})
            after, _ = server_stats(url)
            n = (usage or {}).get('completion_tokens') or 0
            delta = {k: after.get(k, 0) - before.get(k, 0) for k in ('rounds', 'proposed', 'accepted', 'emitted', 'relaxed_accepts')}
            decode_s = last - ttft
            emit({'kind': 'decode', 'length': length, 'mode': mode, 'prompt_tokens': (usage or {}).get('prompt_tokens'),
                  'completion_tokens': n, 'ttft_s': round(ttft, 3), 'decode_s': round(decode_s, 3),
                  'decode_tps': round((n - 1) / decode_s, 2) if n > 1 and decode_s > 0 else None,
                  'spec': delta, 'tokens_per_round': round(delta['emitted'] / delta['rounds'], 3) if delta.get('rounds') else None,
                  'round_ms': round(1000 * decode_s / delta['rounds'], 2) if delta.get('rounds') else None, 'finish': finish})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--url', required=True)
    ap.add_argument('--lengths', default='10240,32768,65536,131072,262144', help='context budgets (prompt + output) in tokens')
    ap.add_argument('--depths', default='0.9,0.5,0.1')
    ap.add_argument('--output-tokens', type=int, default=512)
    ap.add_argument('--corpus', default=DEFAULT_CORPUS)
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--log', required=True)
    args = ap.parse_args()
    run(args.url, args.log, [int(x) for x in args.lengths.split(',')], [float(x) for x in args.depths.split(',')],
        args.output_tokens, args.corpus, args.tokenizer, args.seed)


if __name__ == '__main__':
    main()
