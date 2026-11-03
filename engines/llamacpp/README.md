# llama.cpp

## 1. What runs

One llama.cpp build for every model: ggml-org/llama.cpp `547208d`, `llama-server` only, built from source for sm_121
(section 2). Every setting serves one GGUF with a speculative drafter, q8_0 keys and values in one KV pool shared by
every slot (`-kvu`), flash attention, port 8081. Text only.

| Setting | Model | Checkpoint (revision) | Drafter | Drafts |
|---|---|---|---|---|
| `qwen3.8-flash-next-p` | Qwen3.8-Flash-Next | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF `ed59f92`, IQ3_S (two shards, 83.6 GB) | ggml-org/Qwen3.8-Flash-Next-GGUF `92796c0`, `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (MTP head, 4.1 GB) | 3 |
| `qwen3.8-27b-dflash2` | Qwen3.8-27B | ggml-org/Qwen3.8-27B-GGUF `a316ea5`, Q4_K_M | DFlash2 converted here from z-lab/Qwen3.8-27B-DFlash2 `50307d4` (Q8_0) | 4 |
| `nemotron-3.5-bolt-gguf` | Nemotron 3.5 Bolt | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF `e0fecb6`, IQ2 (22 GB) | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF `182d298` | 7 |
| `nemotron-3.5-super-gguf` | Nemotron 3.5 Super | ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF `99144d1`, Q4_0 (72 GB, the repo's lowest bit) | the same repo's LiLiCorr NVFP4 drafter | 3 |

All four are the **baselines** behind the published llama.cpp numbers (warm decode at 16k and the agent replay). The
GGUFs are not the checkpoints the other engines serve: llama.cpp cannot load those NVFP4 safetensors, so these are the
published GGUFs nearest to them. Bolt runs NVIDIA's own 128 GB recipe flags (`-b 4096 -ub 2048`, 7 drafts). Both
Nemotrons load as GGUF arch `nemotron_h_moe` (llama.cpp PRs #30520 and #30506, in `547208d`).

Each setting holds the sizes for both benchmarks:

| Setting | Warm decode: slot size, largest N | Agent replay pool (tokens) at 1 / 8 / 16 / 32 agents | Slots at 32 agents |
|---|---|---|---|
| `qwen3.8-flash-next-p` | 20,480; N 8 (16 and 32 do not load beside 88 GB of weights) | 81,920 / 245,760 / 393,216 / 393,216 | **16** |
| `qwen3.8-27b-dflash2` | 20,480; N 32 | 81,920 / 245,760 / 425,984 / 851,968 | 32 |
| `nemotron-3.5-bolt-gguf` | 22,528; N 32 | 81,920 / 262,144 / 425,984 / 851,968 | 32 |
| `nemotron-3.5-super-gguf` | 22,528; N 16 (N 32 stopped) | 81,920 / 262,144 / 425,984 / 425,984 | **16** |

Flags beyond the sizes: warm decode `-cram 0` (Bolt also `-b 4096 -ub 2048 --jinja`, Super `--jinja`); agent replay
`-cram 0 --jinja -ctxcp 3` (Flash-Next also `-ub 512`, Bolt `-b 4096 -ub 2048`).

## 2. One-time setup on a Spark

```bash
# llama.cpp 547208d (the tarball: git clone kept dropping on some Sparks)
curl -sfL -o ~/llama.cpp.tgz https://codeload.github.com/ggml-org/llama.cpp/tar.gz/547208d
mkdir -p ~/llama.cpp && tar -C ~/llama.cpp --strip-components=1 -xzf ~/llama.cpp.tgz
export PATH=/usr/local/cuda/bin:$PATH
cd ~/llama.cpp && cmake -B bld -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DLLAMA_CURL=OFF
nice -n 5 cmake --build bld -j 16 --target llama-server && ln -sfn bld ~/llama.cpp/build
hf() { uvx --from huggingface_hub hf "$@"; }      # files as positional arguments: --include with several patterns fetched one
```

Then the model's files, pinned (download with no engine serving):

```bash
# qwen3.8-flash-next-p: the IQ3_S shards and the MTP head (checked)
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --include 'IQ3_S/*'
mkdir -p ~/llamacpp-baseline && cd ~/llamacpp-baseline
curl -sfL -C - -o mtp.part https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/resolve/92796c044d1e70d66a373598709092f2989f4155/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  && mv mtp.part mtp-Qwen3.8-Flash-Next-Q8_0.gguf
echo "6be8a94e2d94107b7f38fdc957c3460ffeb6ff6302edef92c50738682f9dc0e0  mtp-Qwen3.8-Flash-Next-Q8_0.gguf" | sha256sum -c

# qwen3.8-27b-dflash2: the target, then the DFlash2 drafter converted with this build's converter. The published DFlash2
# GGUFs (incoai, the z-lab mirror, ggml-org) predate llama.cpp's DFlash2 support (PR #27342) and do not load.
hf download ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --revision a316ea527fde32c12487be7574e2db5ac2ad2a7e
hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
hf download Qwen/Qwen3.8-27B config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja \
  generation_config.json --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
HUB=~/.cache/huggingface/hub
cd ~/llama.cpp && uv venv -q .venv-convert --python 3.12 \
  && VIRTUAL_ENV=$PWD/.venv-convert uv pip install -q -r requirements/requirements-convert_hf_to_gguf.txt --index-strategy unsafe-best-match
.venv-convert/bin/python convert_hf_to_gguf.py $HUB/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --target-model-dir $HUB/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --outtype q8_0 --outfile ~/llamacpp-baseline/dflash2-Qwen3.8-27B-50307d4-Q8_0.gguf

# nemotron-3.5-bolt-gguf
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --revision e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf --revision 182d2987939e12f458298b5d5f6ef98912f4ad6c

# nemotron-3.5-super-gguf
hf download ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf \
  lilicorr-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf --revision 99144d1d7a318f045f1412a85bbb807f3c8a17a0
```

`LLAMA_DIR` (default `~/llama.cpp`), `LLAMACPP_HOME` (`~/llamacpp-baseline`) and `HF_HUB_CACHE` move these; `GGUF` and
`DRAFT_GGUF` point one start at other files. The prompt sets, once, with no engine serving:
`uv run sharar-eval efficiency prepare --context 16k`.

## 3. Run it

`llama-server` fixes its slot count (`-np`) at start, and N requests at once need N slots, so one server cannot run a
suite's N 1-32 row the way `common/run.sh` does for the other engines. Two drivers start a fresh server per N instead,
each sized from the setting, with sharar-eval's suite options unchanged:

```sh
CONTEXT=16k engines/llamacpp/warm.sh qwen3.8-27b-dflash2          # warm decode: N = 1, 2, 4, 8, 16, 32 (up to WARM_N_MAX)
engines/llamacpp/warm.sh qwen3.8-27b-dflash2 4 8                  # some N only
engines/llamacpp/agents.sh qwen3.8-27b-dflash2                    # agent replay, cross-engine-agents: 1, 8, 16, 32 agents
REPLAY=~/.local/share/sharar/replays/agentperf-default-v1-nonote engines/llamacpp/agents.sh qwen3.8-27b-dflash2 1 8 16
```

- **`warm.sh SETTING [N ...]`**: for each N, `serve.sh up` with `-np N -c N x SLOT_TOKENS` and the setting's
  `WARM_ARGS`, `sharar-eval efficiency warm-decode` with suite `cross-engine`'s options (`SUITE`) at that N and
  `CONTEXT` (default 16k), `serve.sh down`. The row ends at the first N whose server does not start or whose cell
  stops. Results: `~/engine-baselines/llamacpp/SETTING/<UTC time>-<host>-<context>/n<N>/` (`warm.txt`, `server.log`,
  `run/`); `OUT=` an earlier run's directory resumes it.
- **`agents.sh SETTING [AGENTS ...]`**: for each agent count, a server with one slot per agent (at most the setting's
  `AGENT_SLOTS_MAX`; the other agents wait for a slot) and the pool from `AGENT_POOLS`, `sharar-eval efficiency
  agent-replay` with suite `cross-engine-agents`' options (the thinking-off setting, tool time none) and `REPLAY` if
  set, `serve.sh down`. Results: `~/engine-baselines/llamacpp-agents/SETTING/<UTC time>-<host>[-<replay>]/a<N>/`.
- **By hand**: `LLAMA_SLOTS=4 LLAMA_CTX=81920 LLAMA_ARGS="-cram 0" engines/llamacpp/serve.sh up qwen3.8-27b-dflash2 DIR`,
  then any sharar-eval command against `http://127.0.0.1:8081`, then `engines/llamacpp/serve.sh down DIR`.

Both drivers reserve the Spark, refuse `AVOID_HOSTS` and start the memory guard, like `common/run.sh`.
For a few-minute check on one server, `LLAMA_ARGS="-cram 0" engines/common/run.sh llamacpp SETTING smoke` (suite
`smoke`: N 1 and 4 at 8k; the setting's default is 4 slots).

## 4. Sizing and memory

- **One pool, every slot.** With `-kvu` all slots share `-c` tokens of KV and any slot may grow into the whole pool. A
  slot keeps its conversation after the request ends, so the pool must hold every slot's content at once: N x the
  longest prompt and answer for warm decode, every agent's whole history for the agent replay (up to about 56k
  tokens per agent). A pool too small stops requests with "Context size has been exceeded".
- **Host prompt cache off (`-cram 0`).** With it on, llama.cpp moves a finished request's slot state into the host
  cache and empties the slot; a 16k state is about 0.8 GiB, so a 1 GiB cache keeps one, and primed prompts were lost
  (0% cache hit at N 4 and 8). With `-cram 0` every request finds its own slot ("selected slot by LCP similarity,
  f_sim_best = 1.000").
- **Context checkpoints (`-ctxcp`).** These hybrid models resume a prompt from a saved recurrent state. The agent replay
  keeps 3 a slot. **Never `-ctxcp 1`**: with one, llama.cpp cannot step back before the SWE-bench turn message and
  re-reads every SWE-bench prompt (0% cached on SWE-bench, against 95% with 3). A checkpoint is host memory: 112.6 MiB
  for Flash-Next, 174 MiB for Super.
- **Memory.** Flash-Next's 88 GB of weights (IQ3_S + head) leave about 7 GiB beside 8 warm slots, so warm decode ends
  at N 8, and 32 agents run on 16 slots. Super at 32 agents and 32 slots was killed by the host OOM killer two minutes
  in (32 x 3 checkpoints of 174 MiB), so it runs on 16 too. `common/lib.sh`'s guard stops a server under 2 GiB free.

## 5. Status and caveats (runs of 2026-10-29 and 10-30)

- Warm decode is 16k only, `cross-engine` options, one server per N:
  - Flash-Next N 1-8; N 16 and 32 do not load. Its published N 1 and 2 ran on one 2-slot server with `-cram 1024`
    (one state to bring back at most), N 4 and 8 on N slots with `-cram 0`, as `warm.sh` runs every N.
  - Qwen3.8-27B and Bolt N 1-32.
  - Super N 1-16: at N 32 the server started with 9 GiB free and its requests failed while the prompts were loading
    (the cell is recorded as stopped, not measured).
- Agent replay is `cross-engine-agents` at 1, 8, 16 and 32 agents on the AA traces as published, and 1, 8 and 16
  on the no-note copy (`docs/agent-replay.md`).
  - At 32 agents Flash-Next and Super run 16 slots with the other 16 agents waiting. Queued agents evict each other's
    slot contents, so about a quarter of each prompt comes from cache, and some turns fail with "Context size has been
    exceeded" (16-19 of 672). Label those two points with the cap.
  - Every other cell caches 92-93% with no failed turn.
- The replay holds each turn to its recorded length with `ignore_eos`. After a complete tool call llama.cpp starts
  another, which the length cap cuts, so many turns count a tool call whose arguments are not JSON
  (`tool: arguments not JSON`). The cut call is not used (the next turn sends the recorded reply), and timing is
  unaffected.
- `sharar-eval check` passes all eight checks; llama.cpp reports `usage.prompt_tokens_details.cached_tokens`.
- llama.cpp has no flush endpoint (`bench_v1.json`): within one N's server, prompts primed for earlier bursts stay in
  their slots until replaced.
- Stop a server with `serve.sh down DIR`: it runs under `setsid` and outlives a killed driver. Do not match it with
  `pgrep -f llama-server` from a shell whose own command line contains that text: the pattern matches the shell.
