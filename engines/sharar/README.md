# Sharar

## 1. What runs

abdelfattah-lab/sharar `origin/main` at `7387a45` (2026-10-27), a plain checkout (`versions.env`). The checkpoints
are Sharar's own pins at that commit (`sharar/model/*/config.py`), fetched on the first start:

| Setting | `sharar serve` | Port | Checkpoint | Changed from the profile |
|---|---|---|---|---|
| `qwen3.8-27b` | `27b --throughput` | 30001 | RadixArk/Qwen3.8-27B-NVFP4 `319f741`, 3-bit copies `72997f2`, DFlash2 `dedf8df` | max batch 32 (default 16); drafts 15 / 10 / 6 at 1 / 2 / 4+ requests, from the 2026-10-26 depth sweep |
| `qwen3.8-flash-next` | `flashnext --throughput` | 30004 | RadixArk/Qwen3.8-Flash-Next-NVFP4 `7b71922`, 3-bit copies `694e279` | nothing (up to 128 requests) |
| `nemotron-3.5-bolt` | `bolt --throughput` | 30006 | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB `cf2effc`, LiLiCorr draft | max batch 32 (default 16) |
| `nemotron-3.5-super` | `nemotron --throughput` | 30005 | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 `6aae0d6`, 3-bit copies `2ca8b35` | max batch 32 (default 16) |
| `<model>-fast` (each of the four) | `<target> --fast` | as above | as above | nothing: at most 4 requests; run with suite `cross-engine-n4` (N = 1, 2, 4) |

## 2. One-time setup on a Spark

```sh
git clone git@github.com:abdelfattah-lab/sharar.git ~/sharar-main-7387a45
git -C ~/sharar-main-7387a45 checkout --detach 7387a450e14e032ce1d5cbee2edca41326bea6d3
```

On a Spark without GitHub access, bundle the repository on one that has it, copy the bundle over and fetch from it:
`git -C ~/sharar bundle create ~/sharar.bundle origin/main`, then on the other Spark `git init ~/sharar-main-7387a45`,
`git -C ~/sharar-main-7387a45 fetch ~/sharar.bundle 'refs/remotes/origin/main:refs/heads/main'` and the checkout
above. `serve.sh` serves `$SHARAR_DIR` (default `~/sharar-main-7387a45`) and refuses any other commit; `uv run` syncs
the checkout's `.venv` to `uv.lock` on each start. The private 3-bit repositories need a Hugging Face token.

## 3. Run it

```sh
engines/common/run.sh sharar nemotron-3.5-bolt
engines/common/run.sh sharar nemotron-3.5-bolt-fast cross-engine-n4    # --fast, N = 1, 2 and 4
engines/sharar/serve.sh up nemotron-3.5-bolt ~/sharar-bolt    # by hand: start and wait (log in ~/sharar-bolt/server.log)
engines/sharar/serve.sh down ~/sharar-bolt                    # one Ctrl-C: it drains and stores its cached prefixes
```

Each start gets an empty SSD-tier store under `$SHARAR_SSD_ROOT`. A warm start takes about 1.5 minutes; a first start
on a new machine downloads the checkpoint and compiles kernels (20 minutes or more, `START_TIMEOUT_S`, default 2 h).
