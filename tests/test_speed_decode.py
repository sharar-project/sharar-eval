import hashlib
import json

from evals.speed import decode

# engine-bench's suite/h2h.py prompts, caps and document words: the published decode results used exactly these
PROMPTS_SHA256 = "c6f08f9a6e9db764880a177d546d24f6018c84c321ebfcccc13eb595e154ed51"
WORDS_SHA256 = "b7d610a31839086c9f194b59c3b29b234888aa0291645ac6f44d2b379bd936f6"


def test_prompts_caps_and_words_are_h2hs():
    assert hashlib.sha256(json.dumps(decode.WORKLOADS, sort_keys=True).encode()).hexdigest() == PROMPTS_SHA256
    assert hashlib.sha256(" ".join(decode.WORDS).encode()).hexdigest() == WORDS_SHA256
    assert {t: cap for t, (_, cap) in decode.WORKLOADS.items()} == {"chat": 256, "prose": 1024, "code": 1024, "math": 768,
                                                                   "longctx": 256}


def test_decode_rows_cycle_the_prompts_with_unique_tags():
    rows = decode.rows("code", 10, tag=500)
    texts = [r["messages"][0]["content"] for r in rows]
    assert texts[0] == "[request 500] " + decode.CODE[0] and texts[9] == "[request 509] " + decode.CODE[1]
    assert len({r["session_id"] for r in rows}) == 10 and all(r["output_length"] == 1024 for r in rows)
    assert rows[0]["extra"] == {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}


def test_sampled_rows_carry_the_non_thinking_preset():
    assert decode.rows("chat", 1, 0.7, tag=1)[0]["extra"] == {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                                               "chat_template_kwargs": {"enable_thinking": False}}


def test_documents_are_fresh_per_request_and_per_build(snapshot):
    a, b = decode.rows("longctx", 2, tag=7)
    assert a["messages"][0]["content"] != b["messages"][0]["content"] and a["output_length"] == 256
    assert a["messages"][0]["content"].endswith(decode.QUESTION)
    warm, measured = decode.rows("prefill-8000", 0, tag=7)
    assert warm["output_length"] == measured["output_length"] == 1
    assert warm["messages"][0]["content"][20:] != measured["messages"][0]["content"][20:]
    assert decode.rows("prefill-8000", 0, tag=8)[1]["messages"] != measured["messages"]


def test_write_counts_rows(tmp_path):
    assert decode.write(tmp_path / "cell.jsonl", "math", 4, tag=3) == 4
    assert len((tmp_path / "cell.jsonl").read_text().splitlines()) == 4


def test_documents_are_cut_to_their_length_in_the_pinned_tokenizer(snapshot):
    from evals.speed import model
    tok, template = model.SnapshotTokenizer(snapshot), model.ChatTemplate(snapshot)
    assert [len(tok.encode(decode.sized_doc(n, seed))) for n in (4000, 12600) for seed in (1, 2)] == [4000, 4000, 12600, 12600]
    assert decode.sized_doc(4000, 1, snapshot.parent.parent.parent) == decode.sized_doc(4000, 1)  # --snapshot: a cache dir
    warm, measured = decode.rows("prefill-32000", 0, tag=123456789)
    rendered = template.render(measured["messages"], add_generation_prompt=True, enable_thinking=False)
    assert 32000 < len(tok.encode(rendered)) <= 32040  # the tag, the instruction and the chat template add the rest
