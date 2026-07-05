# Progress Tracker

Update this file as work happens — it is the single source of truth for
"where are we." One line per event, newest first, in the Log section.
An agent resuming work: read CLAUDE.md → this file → CURRICULUM.md stage.

## Stage status

| Stage | What | Status |
|---|---|---|
| 0 | Environment + baselines on xeon | **IN PROGRESS** — xeon bootstrapped, lab proven E2E (MoE smoke run done: compile +6.3% tok/s vs eager); next: full 3-repeat baselines for all three experiments |
| 0b | Talk to bigPYJ1151 (question list in CURRICULUM.md "Maintainer channel") | NOT STARTED — user-driven |
| 1 | Reproduce/fix open CPU bugs (#46470, #47014, #46693, #46347) | NOT STARTED |
| 2 | CPU compile test coverage (Inductor smoke test + CI) | NOT STARTED — needs Stage 0b CI-budget answer |
| 3 | Per-op custom-kernel vs Inductor benchmarking series | NOT STARTED — needs Stage 0 baselines |
| 4 | Fusion-pass enablement on CPU + inductor config dict experiments | NOT STARTED |
| 5 | Piecewise/compile_sizes on CPU (research) | NOT STARTED |
| 6 | Capstone: FusedMoE → MK refactor (#36739) | NOT STARTED — claim via maintainer first |

## Next actions (in order)

1. `cd planning/lab && ./labctl setup xeon` (idempotent; enforces the
   /mnt/nvme-only rule; add `--full-build` only when csrc work starts).
2. Smoke-run the lab E2E: `./labctl run baseline-llama-latency
   --hypothesis "N/A — baseline" --repeats 1`, then `status` / `sync`.
3. Full baselines: all three `baseline-*` experiments at default repeats;
   `sync` + `conclude` each. This completes curriculum Step 0.
4. User: message bigPYJ1151 with the 5 maintainer-channel questions
   (CURRICULUM.md); record answers in this file.
5. Re-run duplicate checks (issue/PR numbers in FINDINGS.md may be stale).
6. Pick the first Stage 1 bug and start the loop.

## Answers from bigPYJ1151

(record here when received)

- CI budget for Inductor smoke test: —
- Was the DYNAMO_TRACE_ONCE downgrade / fusion-pass gating measured or assumed: —
- Existing x86 custom-op vs Inductor measurements: —
- #36739 still wanted / preferred MK shape / can we claim: —
- His current top CPU compile pain point: —

## Log

- 2026-07-05: First clean E2E lab run on xeon (run dcccd8a): Qwen3-30B-A3B
  tp1, 200 random 128/128 prompts — compile 205.1 tok/s vs eager 193.0
  (+6.3%), TPOT median 822.7 vs 875.4 ms. Single repeat, directional. Smoke
  runs 1-4 flushed out and fixed 5 lab bugs (tenant port collision,
  zero-metric false success, ssh submit hang, zombie server port, stale
  runner deployment) — all documented in the run conclusions. Xeon:
  Platinum 8568Y+ (AMX), all caches verified on /mnt/nvme. Open thread:
  "Inductor compilation was disabled by user settings" warning on the
  compile arm — investigate what mode 2 actually engages (Stage 1 learning).
- 2026-07-05: Built `planning/lab/` — portable control plane (labctl CLI,
  git-committed run ledger, web UI) + xeon worker harness (flock'd runner,
  nvme-pinned bootstrap for /mnt/nvme/pramod/torch_compile_perf). Retired
  step0 scripts into lab experiments. Verified: validate/dry-run/rails/UI
  pass locally; E2E on xeon pending (next action 1).
- 2026-07-05: Planning session. Explored codebase, checked upstream landscape,
  wrote CURRICULUM.md / FINDINGS.md / step0 kit / this tracker. Models locked:
  Llama-3.1-8B-Instruct (tp2) + Qwen3-30B-A3B (tp1). Hardware: x86 Linux
  server (pending access). Scripts pass `bash -n`.
