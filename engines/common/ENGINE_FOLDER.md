# What an engine folder holds

Every engine is one folder, `engines/<engine>/`, with the same files, so `common/run.sh` can run any of them and an
agent can launch any of them from its README alone. sharar-eval itself never starts an engine: it measures whatever
OpenAI-compatible server answers at a URL. These folders are what starts one.

| File | What it is |
|---|---|
| `README.md` | The launch guide, for an agent with no other context: what runs (pinned versions, one row per model), one-time setup on a fresh Spark, how to run it (`common/run.sh`, and `serve.sh` by hand), what to expect (limits, failure modes). No machine-specific absolute paths (use `$HOME` and variables). |
| `versions.env` | What every setting shares, as shell variables (sourced; no secrets): `EVAL_ENGINE` (sharar-eval's `--engine`: an engine of `evals/workloads/bench_v1.json`, else `openai`), `URL` when it is the same for every model, and the pins: engine repository and commit, image digests. Install locations as overridable defaults, e.g. `EUGR_DIR=${EUGR_DIR:-$HOME/...}`. |
| `settings/<model>[-variant].env` | One base file per model, named the same in every engine folder (`qwen3.8-27b`, `qwen3.8-flash-next`, `nemotron-3.5-bolt`, `nemotron-3.5-super`), plus variants named `<model>-<change>` (e.g. `-fast`, `-d4`, `-maxfit`); sourced after `versions.env`, so a setting may also override its pins (another commit, recipe or binary): `SETTING_DESC` (one line), `ENGINE_LABEL` (what is serving, for `--engine-version`), `URL` if it differs by model, the checkpoint and its revision when the engine does not pin them itself, and the engine's own variables that `serve.sh` reads. The README lists every setting, what it changes and whether it is a baseline. |
| `serve.sh` | `serve.sh up SETTING DIR`: source `../common/defaults.env`, `../common/lib.sh`, `versions.env` and the setting; start a memory guard before the engine loads (`guard_pid` / `guard_container` in `lib.sh`, or the engine's own); start the engine in the background with its log in `DIR/server.log`; wait until it answers (`wait_health`); exit 0 when ready, non-zero after cleaning up when it fails. `serve.sh down DIR`: stop it and wait until no GPU process of it is left. Stop only what it started: a container by name, a process group by its recorded PID; never `pkill -f`. |
| other files | Only what the engine needs beyond this (a patch, a helper). |

The benchmark is not per engine: `common/run.sh ENGINE SETTING [SUITE]` runs sharar-eval's suite the same way for
every engine, with `common/defaults.env`.
