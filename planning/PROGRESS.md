# Progress Tracker

Update this file as work happens — it is the single source of truth for
"where are we." One line per event, newest first, in the Log section.
An agent resuming work: read CLAUDE.md → this file → CURRICULUM.md stage.

## Stage status

| Stage | What | Status |
|---|---|---|
| 0 | Environment + baselines on x86 server | **NOT STARTED** — blocked on server access; scripts ready in `step0/` |
| 0b | Talk to bigPYJ1151 (question list in CURRICULUM.md "Maintainer channel") | NOT STARTED — user-driven |
| 1 | Reproduce/fix open CPU bugs (#46470, #47014, #46693, #46347) | NOT STARTED |
| 2 | CPU compile test coverage (Inductor smoke test + CI) | NOT STARTED — needs Stage 0b CI-budget answer |
| 3 | Per-op custom-kernel vs Inductor benchmarking series | NOT STARTED — needs Stage 0 baselines |
| 4 | Fusion-pass enablement on CPU + inductor config dict experiments | NOT STARTED |
| 5 | Piecewise/compile_sizes on CPU (research) | NOT STARTED |
| 6 | Capstone: FusedMoE → MK refactor (#36739) | NOT STARTED — claim via maintainer first |

## Next actions (in order)

1. User: get SSH access to the x86 Linux server; run `step0/setup_server.sh`.
2. User: message bigPYJ1151 with the 5 maintainer-channel questions
   (CURRICULUM.md); record answers in this file.
3. Agent+user: run `step0/run_baselines.sh`, fill in
   `step0/results/<date>-baselines.md` from `results-template.md`.
4. Re-run duplicate checks (issue/PR numbers in FINDINGS.md may be stale).
5. Pick the first Stage 1 bug and start the loop.

## Answers from bigPYJ1151

(record here when received)

- CI budget for Inductor smoke test: —
- Was the DYNAMO_TRACE_ONCE downgrade / fusion-pass gating measured or assumed: —
- Existing x86 custom-op vs Inductor measurements: —
- #36739 still wanted / preferred MK shape / can we claim: —
- His current top CPU compile pain point: —

## Log

- 2026-07-05: Planning session. Explored codebase, checked upstream landscape,
  wrote CURRICULUM.md / FINDINGS.md / step0 kit / this tracker. Models locked:
  Llama-3.1-8B-Instruct (tp2) + Qwen3-30B-A3B (tp1). Hardware: x86 Linux
  server (pending access). Scripts pass `bash -n`.
