# lab — experiment control plane + tracking

Runs torch.compile CPU experiments on a worker over SSH and keeps a
git-committed ledger of every run. Portable: **any machine** with this repo
clone + SSH access to the worker is a control plane (Mac today, anything
later). State lives in git (`runs/`) and on the worker's disk (artifacts);
nothing depends on the machine you happen to be sitting at.

```
control plane (this clone)                 worker "xeon" (EMR, 2 NUMA)
─────────────────────────                  ────────────────────────────
edit code → commit → push ──────────────▶  git fetch/checkout <sha>
./labctl run exp --hypothesis "..."  ───▶  runner.py: flock → arms × repeats
./labctl status <id>                 ◀───  remote_state.json
./labctl sync <id>  (metrics → git)  ◀───  arms/*/rep*/metrics.json
./labctl ui   (browse the ledger)          artifacts stay here (rsync on demand)
```

## Quickstart

```bash
cd planning/lab
./labctl setup xeon              # once per worker (idempotent); --full-build for csrc work
./labctl run baseline-llama-latency --hypothesis "N/A — baseline"
./labctl status                  # all runs; add a run id for live remote state
./labctl sync <run-id>           # after it finishes: metrics → run.json → git
./labctl compare <run-a> <run-b> --format md   # PR-ready before/after table
./labctl ui                      # http://127.0.0.1:8800
```

## Rules the tooling enforces (don't fight them)

- **Hypothesis before run.** `labctl run` refuses without `--hypothesis`.
  "N/A — baseline" is valid; "let's see what happens" is not.
- **Clean worktree.** A run is keyed to a pushed SHA; dirty trees are refused.
- **`VLLM_CPU_CI_ENV` is forbidden** — it silently forces the eager backend
  and invalidates every compile measurement. Refused at launch and on-worker.
- **One run owns the box.** runner.py holds an exclusive flock for the whole
  run; concurrent submissions queue FIFO behind it.
- **Nothing writes to the worker's root fs** (it is full). bootstrap.sh pins
  all caches (uv, HF models, vLLM compile cache, Inductor/Triton, ccache,
  TMPDIR) under `remote_root` and aborts if any path resolves to the root fs.

## Anatomy

| Path | What |
|---|---|
| `labctl` | the CLI (uv script; agents and humans use the same commands) |
| `config/lab.yaml` | worker registry: ssh alias, `remote_root`, default env |
| `config/local.yaml` | optional per-machine overrides (gitignored) |
| `experiments/*.yaml` | experiment definitions (see below) |
| `runs/<id>/run.json` | the ledger: hypothesis, sha, env, host, metrics — committed |
| `runs/<id>/job.json` | exact resolved job shipped to the worker — committed |
| `worker/bootstrap.sh` | on-worker setup (`labctl setup` ships + runs it) |
| `worker/runner.py` | on-worker harness (lock, checkout, execute, state) |
| `server/` | the read-only web UI over `runs/` |
| `artifact-cache/` | rsynced big artifacts (gitignored) |
| `hosts/<worker>.json` | probed CPU/ISA/NUMA facts, stamped into each run |

## The iterative perf loop

This is the workflow the lab exists for. One iteration:

```bash
# 1. HYPOTHESIZE — from a profile, a code read, or a prior run's conclusion.
#    Write it down; it goes into --hypothesis verbatim.

# 2. CHANGE — edit vLLM source on this branch (python changes need no rebuild:
#    the worker install is editable, checkout is enough). Commit + let labctl push.
git commit -am "[EXPERIMENT] ..."

# 3. RUN — the worker checks out exactly your SHA and benchmarks it.
./labctl run baseline-qwen-serve --hypothesis "max_autotune improves ... because ..."

# 4. OBSERVE
./labctl status <new-run-id>        # live arm/rep state + runner log tail
./labctl sync <new-run-id>          # when done: metrics -> run.json -> git

# 5. COMPARE — across runs = across commits. This is the before/after evidence.
./labctl compare <baseline-run-id> <new-run-id> --format md

# 6. CONCLUDE — verdict into the ledger (also: negative results are results).
./labctl conclude <new-run-id> "CONFIRMED/REFUTED — ..."
```

Notes per step:

- **Rebuilds (C++/csrc changes)**: add a `setup:` line to the experiment —
  it runs on the worker after checkout, before any arm:
  `setup: VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation`
  (TMPDIR/ccache are already pinned to nvme by lab.env).
- **Compile observability**: the compile happens in the server warmup — read
  `arms/<arm>/server.log` (synced under `runs/<id>/remote/`). For deeper views
  add env to the experiment: `TORCH_TRACE: /mnt/nvme/pramod/torch_compile_perf/tmp/trace`
  (inspect with `tlparse`) or `VLLM_DEBUG_DUMP_PATH` (depyf dump of generated
  code). List those dirs in `artifacts:` and pull with `labctl sync --artifacts`.
- **Op-level profiling**: add `--profiler-config '{"profiler":"torch",
  "torch_profiler_dir":"{rep_dir}/profile"}'` to `server_command`, drive
  traffic with `--profile` on `vllm bench serve`, add `profile/*` to
  `artifacts:`; view the trace in ui.perfetto.dev.
- **Compile time**: runner greps it per arm (`compile_time_s` in run.json —
  on CPU's DYNAMO_TRACE_ONCE path this is the first warmup run's duration).
  A warm `~/.cache`-equivalent on nvme means cache hits; delete
  `{remote_root}/.cache/vllm/torch_compile_cache` on the worker to force a
  cold compile measurement.
- **Fair comparisons**: keep repeats/env identical between the two runs;
  the compare view shows arms side by side, so compare inductor-arm vs
  inductor-arm across SHAs (the eager arm doubles as a sanity control —
  it should NOT move when you only touched compile behavior).

## Defining an experiment

```yaml
name: my-experiment            # must match filename
repeats: 3
arms:                          # each arm runs repeats× at the same SHA
  inductor: {extra_args: []}
  eager:    {extra_args: [--enforce-eager]}
env: {VLLM_CPU_OMP_THREADS_BIND: "auto"}   # overlays worker default_env
command: >                     # {metrics_json}/{rep_dir} resolved per repeat,
  vllm bench latency ... --output-json {metrics_json} {arm_extra_args}
# server_command: ...          # serve-type: runner manages lifecycle+health
metrics:                       # dotted paths into the benchmark's JSON output
  avg_latency_s: avg_latency
primary_metric: avg_latency_s
forbidden_env: [VLLM_CPU_CI_ENV]
```

Metrics come from the benchmark's own JSON (`vllm bench latency|throughput
--output-json`, `vllm bench serve --save-result`) — never from scraping logs.
The one sanctioned log-scrape is startup compile time (runner greps the
`torch.compile takes N s` line, tolerating absence).

## For agents

Everything is file-first: read `runs/*/run.json` directly, launch with
`labctl run`, never edit `run.json` by hand mid-run (the worker owns status
until `sync`). After `sync`, write the verdict with `labctl conclude`. Before
building anything on top, re-read `planning/CLAUDE.md` for mission rules.
