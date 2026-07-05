# Codebase & Upstream Findings (explored 2026-07-05, vllm-cpu @ cc1d020d0)

Distilled from a full exploration of the vLLM checkout. A future agent should
trust-but-verify line numbers (code moves); the structural facts are durable.

## How torch.compile actually works on CPU

- GPU default is `VLLM_COMPILE` (mode 3): piecewise Dynamo graphs split at
  attention, custom Inductor fusion passes, CUDA graphs.
- **CPU silently downgrades** in `vllm/platforms/cpu.py:161-185`:
  `VLLM_COMPILE → DYNAMO_TRACE_ONCE` (mode 2), `backend="inductor"`
  (or `"eager"` when `VLLM_CPU_CI_ENV` is set — CI never tests Inductor).
  Whole-graph Inductor C++/OpenMP codegen; no piecewise, no compile_sizes
  specialization, no graph capture/replay (warmup compile is everything —
  `vllm/v1/worker/cpu_model_runner.py:143-149`, freezing toggle at 219-231).
- Inductor config hand-tuned dict at `cpu.py:176-184`:
  `{dce, size_asserts:False, nan_asserts:False, epilogue_fusion:True,
  cpp.dynamic_threads:True}`; `ir_enable_torch_wrap=False`. LoRA forces mode
  NONE (`cpu.py:187-188`). ARM force-enables `+gelu,+gelu_tanh,+gelu_and_mul`
  (`cpu.py:190-207`) — measured wins; no x86 equivalent published.
- `custom_ops=["none"]` on CPU (resolved in `vllm/config/vllm.py:1182-1189`):
  RMSNorm/RoPE/SiluAndMul etc. run as Inductor-compiled `forward_native`, NOT
  the hand-written kernels in `csrc/cpu/{layernorm,pos_encoding,activation}.cpp`.
  Dispatch logic: `vllm/model_executor/custom_op.py` (`dispatch_forward`,
  `maybe_compile`, `default_on`).
- **All fusion passes are CUDA/ROCm/XPU-gated at import** in
  `vllm/compilation/passes/pass_manager.py:35-48`; enabling their PassConfig
  flags on CPU raises NameError. Only generic passes run on CPU:
  NoOpElimination, VllmIRLoweringPass, UnsafeCloneElimination, PostCleanup,
  FixFunctionalization.
- Attention is an opaque op (`cpu.py:402` `opaque_attention_op`); calls C++
  `cpu_attention_with_kv_cache` (`vllm/v1/attention/backends/cpu_attn.py:312-401`).
  Stale-flag rider: `use_sdpa_prefill` removal TODO at `cpu_attn.py:121-122`.
- MoE on CPU: `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` — three
  strategies: grouped-GEMM `cpu_fused_moe` (AMX/NEON/vec), oneDNN torch
  fallback (`cpu_fused_moe_torch`), `SGLFusedMOE` (gated by
  `VLLM_CPU_SGL_KERNEL`). Routing is pure torch.
- Compile cache: `~/.cache/vllm/torch_compile_cache/<hash>` (`backends.py:1062-1074`);
  disable via `VLLM_DISABLE_COMPILE_CACHE`.
- OpenMP interactions are why compile config is conservative:
  `TORCHINDUCTOR_COMPILE_THREADS=1`, `cpp.dynamic_threads` (PR #37391).
  Downgrade lineage: PRs #19539 → #26355 → #37391.

## Observation tooling (the three levels)

1. Op level: `--profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'`
   + `vllm bench serve --profile`; view in ui.perfetto.dev.
   (Old `VLLM_TORCH_PROFILER_DIR` env var is gone.)
2. Compile level: `TORCH_TRACE=<dir>` then `tlparse` — graph breaks, guards,
   recompiles. Compile time logged at startup by `vllm/compilation/monitor.py`.
3. Codegen level: `VLLM_DEBUG_DUMP_PATH=<dir>` (depyf) — generated C++.
   Also `VLLM_PATTERN_MATCH_DEBUG` for fusion-pass matching.

Docs: `docs/design/torch_compile.md`, `docs/design/debug_vllm_compile.md`,
`docs/contributing/profiling.md`.

## Benchmark ground truth (CPU perf CI)

`.buildkite/performance-benchmarks/tests/{latency,throughput,serving}-tests-cpu*.json`:
Llama-3.1-8B-Instruct (tp2, bf16, block 128, SGL kernel on, KVCACHE_SPACE=40)
and Qwen/Qwen3-30B-A3B (tp1 serving, random 128/128, 200 prompts). CPU CI
(`.buildkite/scripts/hardware_ci/run-cpu-test.sh`) sets `VLLM_CPU_CI_ENV=1`
→ eager; `tests/compile/` is absent from the CPU pipeline entirely.

## Upstream landscape (as of 2026-07-05 — RE-CHECK before every PR)

- Open PR **#40777** (bigPYJ1151): "[CPU] Optimize CPU t.compile arguments" —
  narrow (Qwen3-VL compile fix + sampler guards). No-fly zone; shows he
  iterates on the cpu.py compile dict.
- Open issue **#36739** (bigPYJ1151): CPU FusedMoE → modular-kernel (MK) flow
  refactor, ref #36286. Stale, unclaimed. Curriculum capstone.
- Related: **#31985** (unwrap FusedMoE custom op, torch.compile-labeled).
- Stage-1 bug candidates: **#46470** (InternLM2 CPU embedding IndexError,
  v0.23 regression), **#47014** (spec-decode crash on CPU), **#46693**
  (Qwen3.5 mamba_utils.batch_memcpy crash w/o Triton), **#46347**
  (KVCACHE_SPACE-dependent accuracy drop on EPYC).
- Triton-fallback monkeypatching (relevant to #47014/#46693):
  `vllm/v1/worker/cpu_model_runner.py:62-119`.
- "Qwen3.6" is served by the Qwen3.5 hybrid GDN classes (`qwen3_5.py`, commit
  #41025); there is no qwen3_6 module. Qwen3-30B-A3B = `Qwen3MoeForCausalLM`
  (`qwen3_moe.py:432`, bare `@support_torch_compile`) — standard attention,
  cleanest compile story, and already the CPU perf-CI MoE reference.

## Build facts

- Precompiled CPU wheel for Python-only work:
  `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu uv pip install -e .`
- C++ work needs full source build (see `step0/setup_server.sh --full-build`).
- CPU kernel tests: `tests/kernels/attention/test_cpu_attn.py`,
  `tests/kernels/moe/test_cpu_fused_moe.py`, `test_cpu_quant_fused_moe.py`,
  `tests/kernels/test_onednn.py`.
