# vLLM CPU torch.compile Contribution Curriculum

Gold-standard paths: **meta-llama/Llama-3.1-8B-Instruct** (dense, TP=2) and
**Qwen/Qwen3-30B-A3B** (MoE, TP=1). Both are vLLM's own CPU perf-CI reference
models (`.buildkite/performance-benchmarks/tests/*-cpu.json`), so improvements
to them are numbers maintainers already track.

Hardware: x86_64 Linux server (ideally AMX-capable Xeon; AVX512 minimum).
Local Mac is for reading code and writing PRs only — the CPU backend's perf
paths (AMX, oneDNN, SGL kernels) are x86 Linux.

## Ground rules (every PR, per AGENTS.md)

1. Before starting anything: `gh issue view <n> --repo vllm-project/vllm --comments`,
   `gh pr list --repo vllm-project/vllm --state open --search "<keywords>"`.
   Link the check results in the PR description.
2. No single-typo/one-liner PRs; bundle mechanical cleanup with substantive work.
3. Every PR description: why it doesn't duplicate existing work, exact test
   commands + before/after results (from your results log), AI-assistance disclosure.
4. You must be able to defend every line — the learning track below exists so you can.
5. All Python via `uv` / `.venv/bin/python`, never system pip.
6. No-fly zones: PR #40777 (CPU compile args for Qwen3-VL + sampler guards);
   anything covered by an open PR from **bigPYJ1151** (the CPU maintainer —
   check `gh pr list --repo vllm-project/vllm --author bigPYJ1151 --state open`
   before each stage).

### Maintainer channel

You know bigPYJ1151 personally — use it. Scope and align each stage with him
directly *before* investing effort, then summarize the agreed direction in a
public issue comment so the paper trail AGENTS.md expects still exists (and so
other contributors don't duplicate *your* work). Direct access does not waive
the duplicate-work checks or the PR-description requirements; it replaces
"file an RFC and hope" with "ask, then file the issue already aligned."

Questions worth asking him up front (each one can save weeks):

1. **Stage 2**: What CI time budget would he accept for an Inductor-backend
   CPU compile smoke test? Would he rather flip one existing model test to
   inductor than add a new step?
2. **Stages 4–5**: Were the CUDA-gating of fusion passes and the
   `VLLM_COMPILE → DYNAMO_TRACE_ONCE` downgrade (cpu.py:161-185) ever
   *measured* on CPU, or assumed? Any past experiments with `compile_sizes` /
   piecewise on CPU he can share?
3. **Stage 3**: Does he already have x86 measurements of custom CPU kernels vs
   Inductor `forward_native` (the analog of the ARM gelu data behind
   cpu.py:190-207)? Which ops does he suspect are worth it?
4. **Stage 6**: Is #36739 (FusedMoE → MK flow) still wanted, what MK shape
   does he prefer, and can you claim it?
5. What does *he* consider the top CPU torch.compile pain point right now —
   compile time at startup, steady-state perf, or coverage/bugs?

## The one mental model to hold

On GPU, vLLM uses `VLLM_COMPILE` (mode 3): piecewise Dynamo graphs split at
attention ops, custom Inductor fusion passes, CUDA graphs. **On CPU none of
that runs.** `vllm/platforms/cpu.py:161-185` silently downgrades to
`DYNAMO_TRACE_ONCE` (mode 2) + whole-graph Inductor C++/OpenMP codegen, with:

- `custom_ops=["none"]` — Inductor compiles each op's `forward_native` instead
  of dispatching to hand-written CPU kernels (see `vllm/model_executor/custom_op.py`).
- Zero vLLM fusion passes — all are CUDA/ROCm/XPU-gated at import in
  `vllm/compilation/passes/pass_manager.py:35-48`; only the generic passes
  (noop-elimination, IR lowering, clone-elimination, fix-functionalization) run.
- No graph capture/replay analog — warmup-time compile is the whole story
  (`vllm/v1/worker/cpu_model_runner.py:143-149`).
- Attention (`cpu_attention_with_kv_cache`) and MoE (`cpu_fused_moe*`) are
  opaque C++ ops the compiler never sees inside.
- CI never exercises any of this: `VLLM_CPU_CI_ENV=1` forces `backend=eager`,
  and `tests/compile/` is not in the CPU pipeline.

Every stage below either measures a consequence of this or closes a gap it creates.

History (know before challenging): the downgrade lineage runs through
PR #19539 (refine CPU default config), #26355 (CompilationConfig overhaul),
and #37391 (OpenMP thread reallocation fix — why `cpp.dynamic_threads` and
`TORCHINDUCTOR_COMPILE_THREADS=1` exist). Read these before Stage 4–5.

## Learning track (read alongside stages)

| Stage | torch.compile concept | Read |
|---|---|---|
| 0–1 | Dynamo capture, guards, graph breaks, recompiles | `docs/design/torch_compile.md`; `docs/design/debug_vllm_compile.md`; PyTorch "torch.compile: the missing manual"; `vllm/compilation/decorators.py` (`@support_torch_compile`) |
| 2 | What Inductor CPU codegen emits (C++/OpenMP); reading it via TORCH_TRACE + tlparse, depyf | `vllm/compilation/monitor.py`; `vllm/config/compilation.py` (`CompilationMode`, `PassConfig`) |
| 3 | Functionalization, custom-op dispatch, `torch.library` | `vllm/model_executor/custom_op.py` (`dispatch_forward`, `maybe_compile`, `default_on`) |
| 4 | FX post-grad passes, pattern matching | `vllm/compilation/passes/pass_manager.py`; `passes/vllm_inductor_pass.py`; one fusion pass end-to-end (`passes/fusion/qk_norm_rope_fusion.py`) |
| 5 | VllmBackend internals: piecewise split, compile cache, compile_sizes | `vllm/compilation/backends.py` (`split_graph`, `CompilerManager`); `caching.py`; `compiler_interface.py` |
| 6 | Modular-kernel abstraction for MoE | `vllm/model_executor/layers/fused_moe/` (MK flow, issue #36286); `cpu_fused_moe.py` |

---

## Step 0 — Environment + baselines (no PR; every future PR cites these numbers)

Scripts: `step0/setup_server.sh`, `step0/run_baselines.sh`, `step0/results-template.md`.

On the x86 server, check ISA first — AMX presence changes which MoE path runs:

```bash
lscpu | grep -oE 'amx[^ ]*|avx512[^ ]*' | sort -u
```

Setup (see `step0/setup_server.sh` for the full script):

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install
# Python-only iteration (precompiled CPU wheel variant):
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
  uv pip install --editable .
# When touching csrc/cpu/ (Stage 3+), full source build:
uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
huggingface-cli login   # Llama-3.1-8B is gated; request access first
```

Baseline matrix — each config 3×, record medians. **Never set `VLLM_CPU_CI_ENV`**
(it forces eager and invalidates the measurement):

| Model | Bench | Configs |
|---|---|---|
| Llama-3.1-8B-Instruct, bf16, tp2 | `vllm bench latency` + `throughput` | default (inductor) vs `--enforce-eager` |
| Qwen3-30B-A3B, bf16, tp1 | `vllm serve` + `vllm bench serve` (random 128/128, 200 prompts) | default vs `--enforce-eager` |

```bash
VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_SGL_KERNEL=1 vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 -tp 2 \
  --distributed-executor-backend mp --block-size 128 \
  --num-iters-warmup 5 --num-iters 15   # [--enforce-eager for the eager arm]
```

First traces (the three observation levels you'll use forever):

```bash
# Op level — torch profiler, view in ui.perfetto.dev:
vllm serve <model> --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'
vllm bench serve --profile --num-prompts 2 ...
# Compile level — Dynamo/Inductor structured logs:
TORCH_TRACE=./trace_dir vllm bench latency ... && tlparse ./trace_dir/*
# Codegen level — generated C++ and graphs via depyf:
VLLM_DEBUG_DUMP_PATH=./depyf_dump vllm bench latency ...
```

Also record **compile time and cache-hit time** (logged at startup by
`vllm/compilation/monitor.py`) — CPU Inductor compile is minutes-long and is
itself a perf target.

Results log discipline: one `results/YYYY-MM-DD-<topic>.md` per experiment —
commit SHA, CPU model/ISA, env vars, exact command, median±spread, trace
filename. This is what makes Stage 2+ PRs acceptable.

## Stage 1 — Reproduce and fix open CPU bugs (first PRs, low risk)

Teaches: how the CPU runner wraps the model; reading Dynamo/Inductor stack traces.

Candidates (re-check for claimants first):

- **#46470** — InternLM2 CPU embedding IndexError, v0.23 regression. Bisectable,
  self-contained. Difficulty: low. Acceptance: high (labeled bug+cpu).
- **#47014** — spec-decode crash on CPU. Touches the Triton-fallback
  monkeypatching in `vllm/v1/worker/cpu_model_runner.py:62-119`. Difficulty:
  medium. Acceptance: high.
- **#46693** — Qwen3.5 `mamba_utils.batch_memcpy` crash without Triton. Same
  fallback machinery. Difficulty: medium.
- **#46347** — KV-cache-space-dependent accuracy drop on EPYC. Great profiling
  practice; a clean minimal-repro comment is itself a valued contribution.

Also: finish the git archaeology on `vllm/platforms/cpu.py:155-190`
(`git log -L 155,190:vllm/platforms/cpu.py`) so you know why each CPU compile
decision was made before proposing to change any of it.

## Stage 2 — CPU compile test coverage (the boring work others skip)

Gap: the shipped CPU compile path (Inductor) is untested upstream — CI forces
eager and `tests/compile/` never runs on CPU.

Contribution: a CPU compile smoke test (tiny model through
`DYNAMO_TRACE_ONCE`+inductor; assert output parity with eager and zero
recompiles) + a CPU CI step or inductor-enabled variant of one model test.
**Ask bigPYJ1151 directly what CI budget he'd accept** (maintainer-channel
question 1), then open the issue already aligned. Measure first: how long one
Inductor compile of a tiny model takes in the CI container. Difficulty:
low-medium. Acceptance: high once pre-aligned.

Rider cleanup candidate (never standalone): stale `use_sdpa_prefill` removal
noted at `vllm/v1/attention/backends/cpu_attn.py:121-122`.

## Stage 3 — Per-op: Inductor codegen vs hand-written CPU kernel (repeatable PR series)

Gap: `custom_ops=["none"]` means RMSNorm, RoPE, SiluAndMul run as
Inductor-compiled `forward_native` even where hand-written C++ kernels exist
(`csrc/cpu/layernorm.cpp`, `pos_encoding.cpp`, `activation.cpp`). ARM already
force-enables three gelu ops (`cpu.py:190-207`) because measurement showed the
kernel wins — nobody has published the same matrix for x86.

Loop per op: microbenchmark both paths at decode-realistic shapes (batch×hidden
for 8B and 30B-A3B) → confirm in a torch-profiler trace of a real run → if the
custom op wins meaningfully, PR adding it to the platform default `custom_ops`
with numbers; if Inductor wins, that's a useful issue comment. Difficulty: low
per op; each is a self-contained defensible PR. Acceptance: high **with numbers
on ≥2 ISAs** (AMX + AVX512 if possible; state which you tested). Duplicate
check: PR #40777's files and `gh pr list --search "custom_ops cpu"`.

This is where you learn to read Inductor's generated C++ (depyf dump) — it
pays for Stages 4–5.

## Stage 4 — Enable viable fusion passes on CPU

Gap: every fusion pass is import-gated to CUDA-alike
(`passes/pass_manager.py:35-48`); enabling their `PassConfig` flags on CPU
raises `NameError`. Some are semantically platform-neutral pattern rewrites —
start with `QKNormRoPEFusionPass` (Llama and Qwen3 both have qk-norm+rope);
also inspect `ScatterSplitReplacementPass` / `SplitCoalescingPass`.

Order: check pass preconditions (does it rewrite to a CUDA-only custom op? if
so it needs a CPU lowering — bigger job) → un-gate locally → verify the pattern
matches on the CPU graph (`VLLM_PATTERN_MATCH_DEBUG`) → correctness (`lm_eval`
or Stage 2 parity test) → benchmark. Difficulty: medium-high. Acceptance:
medium — ask bigPYJ1151 first whether the gating was measured or assumed
(maintainer-channel question 2), then file the issue with your numbers. Expect
some candidates to die on measurement; that's the curriculum working.

Sibling experiment (cheap, same stage): the Inductor config dict at
`cpu.py:176-185`. Measure `max-autotune` (+ `freezing`, see
`cpu_model_runner.py:219-231`) and epilogue-fusion on/off on both gold models.
A data-backed tweak to that dict is a small, very acceptable PR — it's the
maintainer's own dict and PR #40777 shows he iterates on it; coordinate, don't
collide.

## Stage 5 — Revisit the big downgrade: piecewise / compile_sizes on CPU

Research stage, not a guaranteed PR. `VLLM_COMPILE` (piecewise +
compile-size specialization + vLLM compile-cache semantics) is unexplored on
CPU — mode 2 gives one dynamic-shape graph, so Inductor can't shape-specialize
hot decode sizes. Prototype `-cc.mode=3` with `splitting_ops` (attention is
already opaque, so splitting may be near-free) and/or `compile_sizes` for
common decode batch sizes. Measure compile-time cost vs steady-state gain on
both models. The downgrade may exist precisely because piecewise bought
nothing without CUDA graphs — one conversation with bigPYJ1151
(maintainer-channel question 2) plus the Stage 1 archaeology tells you whether
that was measured or assumed, before you write any code. Even a negative
result written up on an issue is a real contribution.

## Stage 6 — Capstone: CPU FusedMoE → modular-kernel refactor (#36739)

Unclaimed, stale, filed by the CPU maintainer, directly on the Qwen3-30B-A3B
path. Refactor `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` (three
strategies: grouped-GEMM `cpu_fused_moe`, oneDNN torch fallback, `SGLFusedMOE`)
into the MK flow (ref #36286). First action: claim it with bigPYJ1151 directly
(preferred MK shape, scope), then comment on #36739 to un-stale it and record
the agreed approach publicly. Keep green:
`tests/kernels/moe/test_cpu_fused_moe.py`, `test_cpu_quant_fused_moe.py`.
Difficulty: high (needs full C++ build + AMX box to exercise all paths).
Acceptance: high — it's literally requested. Do it only after Stages 1–3 built
credibility and MoE-path understanding. Related: #31985 (unwrap FusedMoE
custom op — torch.compile-relevant).

---

## The standing feedback loop (every stage)

1. **Profile** at the level matching your hypothesis: torch-profiler trace (op
   level), TORCH_TRACE/tlparse (compile level), depyf dump (codegen level).
2. **Hypothesize**: one falsifiable sentence, written in the results log
   *before* changing code.
3. **Fix**: smallest change that tests it.
4. **Benchmark**: the Step-0 matrix subset the change could affect, 3× medians;
   `lm_eval --model vllm` (or the `-m cpu_model` generation tests) for
   correctness on anything touching numerics.
5. **Decide**: PR (log becomes the test section), issue comment
   (negative/ambiguous result), or drop.

## Realism flags

Likely rejected: enabling piecewise on CPU without maintainer pre-alignment
(Stage 5); un-gating fusion passes whose fused target kernels don't exist on
CPU (Stage 4); CI additions without a cost discussion (Stage 2); `custom_ops`
default changes benchmarked on only one CPU (Stage 3).

Likely accepted: Stage 1 bug fixes with repro+test; Stage 3 with cross-ISA
numbers; the `inductor_compile_config` tweak with data; Stage 6 (#36739).

bigPYJ1151 gates all of it — you have a direct line, so use the
maintainer-channel question list before each stage, and keep the public issue
trail current with numbers so the rest of the community sees the alignment.
