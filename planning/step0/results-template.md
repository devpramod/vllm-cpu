# <YYYY-MM-DD> — <topic, e.g. "baselines" or "rmsnorm custom-op vs inductor">

## Environment
- vLLM commit: `<sha>` (branch: `<branch>`)
- CPU: `<model name from lscpu>` — ISA: `<amx_bf16/avx512f/... from lscpu>`
- Sockets/NUMA: `<from lscpu>`
- Env vars: `VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_SGL_KERNEL=1 ...` (list every VLLM_/TORCH_ var set)
- torch version: `<pip show torch>`

## Hypothesis
<One falsifiable sentence, written BEFORE running. For baselines: "N/A — baseline.">

## Commands
```bash
<exact command(s), copy-pasteable>
```

## Results (median of N=<runs>, spread in parens)

| Config | Metric | Value |
|---|---|---|
| llama8B tp2, inductor | avg latency (s) | `<x.xx (±y.yy)>` |
| llama8B tp2, eager | avg latency (s) | |
| llama8B tp2, inductor | throughput (tok/s) | |
| qwen30B-A3B tp1, inductor | output tok/s | |
| qwen30B-A3B tp1, inductor | TTFT p50/p99 (ms) | |
| ... | compile time at startup (s) | `<from monitor.py log line>` |
| ... | compile cache-hit boot time (s) | |

## Traces
- torch profiler: `<path>.json` (view: ui.perfetto.dev) — key observation: <...>
- tlparse: `<trace_dir>` — graph breaks: <n>, recompiles: <n>
- depyf dump: `<path>` (if codegen-level question)

## Conclusion
<Hypothesis confirmed/refuted/ambiguous. Next action: PR / issue comment / drop.>
