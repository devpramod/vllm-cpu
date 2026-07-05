# Mission: vLLM CPU torch.compile contributions

This `planning/` directory is the scaffolding for landing accepted upstream
PRs that improve vLLM's CPU torch.compile path. It lives on the
`torch_compile_tutorial` branch of the `devpramod/vllm-cpu` fork **only**.

> **Hard rule:** `planning/` must NEVER appear in a branch cut for an
> upstream PR to vllm-project/vllm. Always branch PRs from a clean
> `main`/upstream ref, never from `torch_compile_tutorial`.

Note: this file auto-loads only when a Claude session's working directory is
inside `planning/`. When working from the repo root, read it explicitly.

## Orientation for any agent (or human) picking this up

Read in this order:

1. **PROGRESS.md** — current stage, next actions, log. Update it after every
   work session; it is the single source of truth for state.
2. **CURRICULUM.md** — the 6-stage contribution ladder (bug fixes → CI
   coverage → per-op benchmarking → fusion passes → piecewise research →
   FusedMoE MK refactor #36739), the profile→hypothesize→fix→benchmark loop,
   the torch.compile learning track, and the maintainer-channel question list.
3. **FINDINGS.md** — distilled codebase facts (file:line) and the upstream
   issue/PR landscape as of 2026-07-05. Re-verify issue/PR state before
   relying on it.
4. **step0/** — executable environment-setup and baseline-benchmark kit for
   the x86 server (`setup_server.sh`, `run_baselines.sh`,
   `results-template.md`). Results land in `step0/results/`.

## Fixed decisions (don't re-litigate)

- Gold-standard models: **meta-llama/Llama-3.1-8B-Instruct** (dense, tp2) and
  **Qwen/Qwen3-30B-A3B** (MoE, tp1) — both are vLLM's own CPU perf-CI
  reference models. ("Qwen3.6-35B-A3B" was ruled out: served by Qwen3.5 hybrid
  GDN classes, a hairier path deferred to later.)
- Hardware: x86_64 Linux server (AMX preferred). Pramod's local Mac is for
  reading/writing code only — the CPU backend's macOS ARM build has open
  breakage (#41537, #41437) and no perf-relevant ISA.
- The user (Pramod, devpramod) is PyTorch-fluent but new to torch.compile and
  wants to *learn* — when executing stages, explain the compile-stack
  concepts encountered (Dynamo guards/breaks, functionalization, Inductor
  codegen) rather than just doing the work.
- Pramod **personally knows bigPYJ1151** (vLLM CPU maintainer) — align scope
  with him directly before each stage, then leave a public issue trail.

## Hard rules

- The repo-root `AGENTS.md` governs all contribution work: duplicate-work
  checks before every PR, no low-value one-off PRs, uv-only Python, PR
  descriptions must carry test commands + results + AI-assistance disclosure,
  and the human must understand and defend every line.
- **Never set `VLLM_CPU_CI_ENV`** when benchmarking — it silently forces the
  eager backend and invalidates any compile measurement.
- Every experiment gets a results log (`step0/results-template.md` format)
  with hypothesis written *before* the run; before/after numbers from these
  logs are what make PRs defensible.
- No-fly: files touched by open PR #40777; anything covered by an open
  bigPYJ1151 PR (`gh pr list --repo vllm-project/vllm --author bigPYJ1151 --state open`).
