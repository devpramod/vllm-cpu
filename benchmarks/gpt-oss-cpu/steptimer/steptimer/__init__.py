"""vLLM general plugin: wall-clock timers around model-runner step phases.

Active only when STEPTIMER_DIR is set. Each worker process writes cumulative
stats to $STEPTIMER_DIR/<pid>.json. Only calls made inside a real
execute_model/sample_tokens step are counted (dummy/warmup runs are not).
Touching $STEPTIMER_DIR/RESET clears the stats (each new mtime resets once).
With STEPTIMER_RF=1, each phase is also wrapped in a torch.profiler
record_function scope named "ST::<phase>".
"""

import contextlib
import functools
import json
import os
import time
from collections import defaultdict

_stats: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
_depth = 0
_steps = 0
_reset_mtime = 0.0


def _dir() -> str:
    return os.environ["STEPTIMER_DIR"]


def _dump() -> None:
    with open(os.path.join(_dir(), f"{os.getpid()}.json"), "w") as f:
        json.dump({k: {"calls": v[0], "total_s": v[1]} for k, v in _stats.items()},
                  f, indent=1)


def _maybe_reset() -> None:
    global _reset_mtime
    try:
        m = os.stat(os.path.join(_dir(), "RESET")).st_mtime
    except FileNotFoundError:
        return
    if m != _reset_mtime:
        _stats.clear()
        _reset_mtime = m


def _scope(key: str):
    if os.environ.get("STEPTIMER_RF") == "1":
        import torch

        return torch.profiler.record_function(f"ST::{key}")
    return contextlib.nullcontext()


def _wrap(cls, name: str, key: str, step: bool = False) -> None:
    orig = getattr(cls, name)

    @functools.wraps(orig)
    def timed(*args, **kwargs):
        global _depth, _steps
        if not step and _depth == 0:
            return orig(*args, **kwargs)
        _depth += step
        t0 = time.perf_counter()
        try:
            with _scope(key):
                return orig(*args, **kwargs)
        finally:
            s = _stats[key]
            s[0] += 1
            s[1] += time.perf_counter() - t0
            _depth -= step
            if step and key == "sample_tokens":
                _steps += 1
                _maybe_reset()
                if _steps % 5 == 0:
                    _dump()

    setattr(cls, name, timed)


def register() -> None:
    if not os.environ.get("STEPTIMER_DIR"):
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _wrap(GPUModelRunner, "execute_model", "execute_model", step=True)
    _wrap(GPUModelRunner, "sample_tokens", "sample_tokens", step=True)
    _wrap(GPUModelRunner, "_model_forward", "target_forward")
    _wrap(GPUModelRunner, "_sample", "sample")
    _wrap(GPUModelRunner, "propose_draft_token_ids", "drafter_total")

    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    _wrap(GptOssForCausalLM, "compute_logits", "target_logits")
    try:
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

        _wrap(DFlashQwen3ForCausalLM, "compute_logits", "drafter_logits")
    except ImportError:
        pass
