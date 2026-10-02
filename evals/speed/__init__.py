"""HTTP speed and serving benchmarks, moved from abdelfattah-lab/sharar (2026-10-02).

    python -m evals.speed --url http://127.0.0.1:30004            # AIPerf: cold-prefix BS1 TPS on the frozen 10K/1K workloads
    python -m evals.speed.sharegpt --freeze inputs.json | --inputs inputs.json --url ... --output DIR
    python -m evals.speed.competitive --freeze inputs.json | --inputs ... --url ... --profile sharar-... --output DIR
    python -m evals.speed.engine_matrix --freeze inputs.json | --inputs ... --url ... --profile ... --window 1 --output DIR
    python -m evals.speed.local --url ... --output DIR             # frozen local-use requests (cache, cancellation, mixed arrival)
    python -m evals.speed.agent --freeze inputs.json | --inputs ... --url ... --profile ... --output DIR

They tokenize on the client (evals/speed/model.py: the pinned Qwen3.8-27B snapshot) so every engine
receives identical token IDs, and talk to /generate (SGLang-native), /v1/completions or /v1/chat.
The frozen prompts (workloads.py, data/) are byte-identical to sharar's, which its in-process
benchmarks still use; tests/test_speed.py pins their digests.
"""
