# Hibana (formerly Sharar)

## 1. What runs

[Hibana](https://github.com/sharar-project/hibana) (formerly Sharar) `main` at `7387a45` (2026-10-27), a plain checkout (`versions.env`). That commit predates the
rename, so its command, settings and paths here are still `sharar`, `SHARAR_*` and `~/sharar-*`. The checkpoints are
its own pins at that commit (`sharar/model/*/config.py`), fetched on the first start:

| Setting | `sharar serve` | Port | Checkpoint | Changed from the profile |
|---|---|---|---|---|
| `qwen3.8-27b` | `27b --throughput` | 30001 | RadixArk/Qwen3.8-27B-NVFP4 `319f741`, 3-bit copies `72997f2`, DFlash2 `dedf8df` | max batch 32 (default 16); drafts 15 / 10 / 6 at 1 / 2 / 4+ requests, from the 2026-10-26 depth sweep |
| `qwen3.8-flash-next` | `flashnext --throughput` | 30004 | RadixArk/Qwen3.8-Flash-Next-NVFP4 `7b71922`, 3-bit copies `694e279` | nothing (up to 128 requests) |
| `nemotron-3.5-bolt` | `bolt --throughput` | 30006 | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB `cf2effc`, LiLiCorr draft | max batch 32 (default 4: at this commit Bolt's `--throughput` sets what `--fast` does) |
| `nemotron-3.5-super` | `nemotron --throughput` | 30005 | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 `6aae0d6`, 3-bit copies `2ca8b35` | max batch 32 (default 16) |
| `nemotron-3.5-bolt-w3` | `bolt --throughput` | 30006 | as `nemotron-3.5-bolt`, plus 3-bit copies from `SHARAR_W3_CHECKPOINT` (default `~/bolt-w3/ckpt-gptq`; the same files as shadowpa0327/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-sharar-w3 `f2b8327`) | max batch 32; Bolt served as the other three models are (Hibana pins no Bolt calibration). The agent-replay baseline for Bolt |
| `nemotron-3.5-bolt-w3-relaxed` | `bolt --throughput` | 30006 | as `-w3` | relaxed acceptance (top-3 within 2.0 logits, sampled requests too): outside Hibana's validated Bolt defaults, an experiment only |
| `<model>-fast` (each of the four) | `<target> --fast` | as above | as above | nothing: at most 4 requests; run with suite `cross-engine-n4` (N = 1, 2, 4) |
| `<model>-b64` (each of the four) | `<target> --throughput` | as above | as above | max batch 64 (Flash-Next: 64 instead of 128, which leaves ~7 GiB more KV pool); run with suite `cross-engine-n64`. 27B drafts as `qwen3.8-27b` (6 at 64). `nemotron-3.5-super-b64` runs another commit, below |

KV cache and recurrent state are the profiles' defaults at this commit: NVFP4 KV for 27B and Flash-Next, FP8 for
Bolt and Super; int8 recurrent state (27B, Flash-Next, Bolt), FP16 with stochastic rounding (Super). Hibana sizes its
own memory (no memory-share knob) and runs its own memory guard (4 GiB floor) beside `serve.sh`'s.

`nemotron-3.5-super-b64` serves `026767d` (branch `fix/spec-graphs-workspace-rows` of the private `abdelfattah-lab/sharar`, so this one setting needs access to it) from
`~/sharar-7387a45-ws`: `7387a45` plus one fix that sizes FlashInfer's speculative-graph scratch by verified rows,
without which Super does not start at max batch 64. It changes no computation.

**Bolt and the 3-bit copies.** Bolt's `--fast` / `--throughput` turn the 3-bit copies on, but only a local
calibration provides them: `SHARAR_W3_CHECKPOINT` (what `-w3` sets), else `$SHARAR_RESULTS/w3-checkpoints/bolt-current`
(`~/.local/share/sharar/...`) if that link exists, else none (served from the checkpoint encodings, about 25% slower
at one request). So `nemotron-3.5-bolt` and `-fast` serve 3-bit copies on a box that has that link: check the
preflight summary at the top of `server.log`.

## 2. One-time setup on a Spark

```sh
git clone https://github.com/sharar-project/hibana.git ~/sharar-main-7387a45
git -C ~/sharar-main-7387a45 checkout --detach 7387a450e14e032ce1d5cbee2edca41326bea6d3
# only for nemotron-3.5-super-b64
git clone https://github.com/abdelfattah-lab/sharar.git ~/sharar-7387a45-ws  # 026767d is on the private repo only
git -C ~/sharar-7387a45-ws checkout --detach 026767db640a50da9d7a817cce50f93ea614a891
# only for nemotron-3.5-bolt-w3 / -w3-relaxed: the calibrated copies (public), where SHARAR_W3_CHECKPOINT points
hf download shadowpa0327/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-sharar-w3 \
  --revision f2b83274fce544af35086f4fdce430158a34f000 --local-dir ~/bolt-w3/ckpt-gptq
```

On a Spark without GitHub access, bundle the repository on one that has it, copy the bundle over and fetch from it:
`git -C ~/sharar bundle create ~/sharar.bundle origin/main`, then on the other Spark `git init ~/sharar-main-7387a45`,
`git -C ~/sharar-main-7387a45 fetch ~/sharar.bundle 'refs/remotes/origin/main:refs/heads/main'` and the checkout
above. `serve.sh` serves `$SHARAR_DIR` (default `~/sharar-main-7387a45`) and refuses any other commit; `uv run` syncs
the checkout's `.venv` to `uv.lock` on each start. CUDA in `/usr/local/cuda`. Checkpoints go to
`~/.cache/huggingface` (`SHARAR_HF_HOME` overrides; a login shell's shared `HF_HOME` is ignored). The 3-bit copies of
27B, Flash-Next and Super are private Hugging Face repositories: log in once with a token that can read them
(`hf auth login`; never put a token on a command line).

## 3. Run it

```sh
engines/common/run.sh sharar nemotron-3.5-bolt                          # suite cross-engine, N = 1-32
engines/common/run.sh sharar nemotron-3.5-bolt-fast cross-engine-n4     # --fast, N = 1, 2 and 4
engines/common/run.sh sharar nemotron-3.5-bolt-b64 cross-engine-n64     # N = 64
engines/common/run.sh sharar nemotron-3.5-bolt-w3 cross-engine-agents   # the agent replay (Bolt's baseline setting)
engines/sharar/serve.sh up nemotron-3.5-bolt ~/sharar-bolt    # by hand: start and wait (log in ~/sharar-bolt/server.log)
engines/sharar/serve.sh down ~/sharar-bolt                    # one Ctrl-C: it drains and stores its cached prefixes
```

Each start gets an empty SSD-tier store under `$SHARAR_SSD_ROOT`. A warm start takes about 1.5 minutes; a first start
on a new machine downloads the checkpoint and compiles kernels (20 minutes or more, `START_TIMEOUT_S`, default 2 h).
The server runs under `setsid`: if `run.sh` is killed, stop it with `serve.sh down <the run's directory>`.
A vLLM container run on the same Spark leaves root-owned kernel caches unless its `serve.sh down` ran (it hands them
back); Hibana then fails with `PermissionError` on `~/.cache/b12x`: `sudo chown -R $USER ~/.cache/b12x`.
