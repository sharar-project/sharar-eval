"""Which warm-decode prompts TensorFold resumes from their kept state: each prompt is primed (1 token), then sent again
with max_tokens M; the reply's cached_tokens says whether the kept state was restored. One line per prompt and M.

    uv run python engines/tensorfold/restore_probe.py URL CONTEXT [M,...]     # e.g. http://127.0.0.1:8888 32k 1024,64
"""
import json
import sys
import urllib.request

from evals import efficiency


def chat(url, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=900))["usage"]
    return u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens")


def main():
    url, context = sys.argv[1], sys.argv[2]
    lengths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1024").split(",")]
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"]
    prompts = efficiency.speedbench_prompts(context, "low_entropy", 32, balanced=True, exclude=("jetbrains_lca",), seed=0)
    for m in lengths:
        for i, p in enumerate(prompts):
            chat(url, model, p, 1)
            n, cached = chat(url, model, p, m)
            print(json.dumps({"context": context, "prompt": i, "prompt_tokens": n, "max_tokens": m, "cached": cached,
                              "restored": bool(cached and cached > n // 2)}), flush=True)


if __name__ == "__main__":
    main()
