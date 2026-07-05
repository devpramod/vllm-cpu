# Learning path: torch.compile on vLLM CPU, one step at a time

Personal, sequential, hands-on. Each step has **Do** (commands), **Read**
(code/docs), **Understand** (the torch.compile concept), and **Done when**.
Work top to bottom; update PROGRESS.md as you go. CURRICULUM.md holds the
contribution ladder this feeds into; this file is *your* daily driver.

---

## Step 1 — Drive one full loop yourself (today)

A max-autotune experiment (`[EXPERIMENT] CPU: enable inductor max_autotune`)
is already running/finished on xeon. Close the loop with your own hands:

**Do**
```bash
cd planning/lab
./labctl list                                  # find the 63a5a60 run + baseline dcccd8a
./labctl status 2026-07-05-63a5a60-baseline-qwen-serve
./labctl sync   2026-07-05-63a5a60-baseline-qwen-serve
./labctl compare 2026-07-05-dcccd8a-baseline-qwen-serve \
                 2026-07-05-63a5a60-baseline-qwen-serve --format md
./labctl conclude 2026-07-05-63a5a60-... "CONFIRMED/REFUTED — <your verdict>"
./labctl ui                                    # see it all rendered
```
**Read** `lab/README.md` ("The iterative perf loop"), the run's `run.json`.
**Understand** why every run is pinned to a commit SHA and why the hypothesis
is written before the run — this is what makes numbers PR-defensible.
**Done when** you've written the conclusion yourself and can explain the
compare table's every cell.

## Step 2 — Understand what "compile" means on vLLM CPU

**Do** Nothing runs; this is a reading step with one puzzle.
**Read** in order:
1. `docs/design/torch_compile.md` (vLLM's own design doc)
2. `vllm/platforms/cpu.py:156-190` — the downgrade block you experimented on
3. `vllm/config/compilation.py` — `CompilationMode` (0/1/2/3), find what
   `DYNAMO_TRACE_ONCE` promises vs `VLLM_COMPILE`
4. `vllm/compilation/decorators.py` — how `@support_torch_compile` wraps
   `Qwen3MoeModel` (`qwen3_moe.py:432`) and what happens in mode 2
**Understand** Dynamo's job (bytecode → FX graph, guards, graph breaks) vs
Inductor's job (FX graph → C++/OpenMP kernels); why CPU skips piecewise.
**Puzzle to solve** (open thread from our smoke run): the compile arm logged
`"Inductor compilation was disabled by user settings"` — find the code that
emits it (grep in `vllm/config/`) and explain why it fires on CPU mode 2 even
though `backend="inductor"`. Write the answer into PROGRESS.md.
**Done when** you can explain to bigPYJ1151, without notes, what actually
happens between "Warming up model for the compilation" and "startup complete".

## Step 3 — Watch the compiler work (observability tooling)

**Do** Rerun the qwen experiment with compiler introspection, via the lab:
add to `experiments/baseline-qwen-serve.yaml` (temporarily, on a branch commit):
```yaml
env:
  TORCH_TRACE: /mnt/nvme/pramod/torch_compile_perf/tmp/ttrace
  VLLM_DEBUG_DUMP_PATH: "{rep_dir}/depyf"
```
run with `--repeats 1`, then `./labctl sync <id> --artifacts` and:
```bash
uvx tlparse artifact-cache/<id>/...ttrace...   # graph breaks, guards, recompiles
less artifact-cache/<id>/.../depyf/...         # the actual generated C++
```
**Read** `docs/design/debug_vllm_compile.md`; skim one generated C++ kernel
and find the OpenMP pragma and the fused ops in it.
**Understand** guards (what makes a recompile), functionalization (why the
graph has no in-place ops), and what Inductor's epilogue fusion did.
**Done when** you can point at one fused kernel in the depyf dump and name
the model ops it came from.

## Step 4 — Profile a real serving run (find where time goes)

**Do** Copy `baseline-qwen-serve.yaml` → `profile-qwen-serve.yaml`; add
`--profiler-config '{"profiler":"torch","torch_profiler_dir":"{rep_dir}/profile"}'`
to `server_command`, `--profile --num-prompts 8` to the client command,
`artifacts: ["profile/*"]`. Run, sync `--artifacts`, open the trace in
ui.perfetto.dev.
**Understand** the decode-step anatomy on CPU: attention (opaque C++ op —
`cpu_attention_with_kv_cache`), MoE (`cpu_fused_moe*`), and the
inductor-compiled glue between them. Rank the top-5 time consumers.
**Done when** you have a written list: "op X: N% of decode step" — this list
IS the Stage 3+ target list in CURRICULUM.md.

## Step 5 — Finish curriculum Step 0 (real baselines) + open the channel

**Do**
```bash
./labctl run baseline-llama-latency    --hypothesis "N/A — baseline"   # tp2 dense
./labctl run baseline-llama-throughput --hypothesis "N/A — baseline"
./labctl run baseline-qwen-serve       --hypothesis "N/A — baseline"   # 3 reps now
# sync + conclude each
```
(Llama is HF-gated: `huggingface-cli login` on xeon first.)
In parallel: message bigPYJ1151 with the 5 questions in CURRICULUM.md
("Maintainer channel") and record answers in PROGRESS.md.
**Done when** PROGRESS.md Stage 0 says DONE and the answers section is filled.

## Step 6 — First contribution candidate (Stage 1 of CURRICULUM.md)

**Do** Re-run the duplicate checks, pick one of #46470 / #47014 / #46693,
reproduce it (CPU-only bugs may not need xeon — some reproduce anywhere
Linux). Fix it. Test per AGENTS.md. PR with the disclosure block.
**Understand** whatever the bug teaches — that's the point of starting here.
**Done when** the PR is open and CI is green.

## Step 7 and beyond

You now have: the loop (Steps 1, 3-5), the mental model (Step 2), the
profile-derived target list (Step 4), maintainer alignment (Step 5), and a
first PR (Step 6). Proceed through CURRICULUM.md Stages 2→6 — each stage's
experiments run through the lab, each PR cites lab run IDs as evidence.

---

**Rhythm suggestion**: Steps 1-2 in one sitting (~2h). Step 3-4 one evening
each. Step 5's runs are wall-clock-long but hands-off. Don't skip the
puzzle in Step 2 — it's the difference between using the loop and
understanding what it measures.
