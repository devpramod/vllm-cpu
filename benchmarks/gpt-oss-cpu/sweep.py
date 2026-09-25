#!/usr/bin/env python
"""Sequential vLLM CPU serving sweep for gpt-oss-20b.

One server config runs at a time (the machine is exclusive); each config is
benchmarked over every (workload, concurrency) point of the selected phase.
Finished points are skipped on rerun, so an interrupted sweep can be resumed.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VLLM = str(VENV / "bin/vllm")
MODEL = ROOT / "models/gpt-oss-20b"
DRAFTER = ROOT / "models/gpt-oss-20b-DFlash"
SHAREGPT = ROOT / "data/ShareGPT_V3_unfiltered_cleaned_split.json"
PORT = 8100
SERVED_NAME = "gpt-oss-20b"

LD_PRELOAD = ":".join(
    [
        str(VENV / "lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4"),
        str(VENV / "lib/libiomp5.so"),
    ]
)

BASE_SERVE_ARGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "4096",
    "--max-num-seqs", "32",
    "--max-num-batched-tokens", "4096",
]
NO_PREFIX_CACHING = ["--no-enable-prefix-caching"]

# name -> (extra serve args, number of NUMA nodes used)
CONFIGS: dict[str, tuple[list[str], int]] = {
    "TP2": (["-tp", "2"], 2),
    "TP1": ([], 1),
    "TP2_EP": (["-tp", "2", "--enable-expert-parallel"], 2),
    "PP2": (["-pp", "2"], 2),
    "PP3": (["-pp", "3"], 3),
    "TP4": (["-tp", "4"], 4),
    "TP4_EP": (["-tp", "4", "--enable-expert-parallel"], 4),
    "PP6": (["-pp", "6"], 6),
    "TP2_PP3": (["-tp", "2", "-pp", "3"], 6),
    "DP6": (["-dp", "6"], 6),
    "DP3_TP2": (["-dp", "3", "-tp", "2"], 6),
    "DP3_TP2_EP": (["-dp", "3", "-tp", "2", "--enable-expert-parallel"], 6),
    "TP4_DFLASH": (["-tp", "4", "--disable-hybrid-kv-cache-manager",
                    "--speculative-config", json.dumps(
        {"model": str(DRAFTER), "method": "dflash", "num_speculative_tokens": 7}
    )], 4),
    "TP4_NOHYBRID": (["-tp", "4", "--disable-hybrid-kv-cache-manager"], 4),
}
# Hybrid KV cache manager stays on; needs VLLM_KV_GROUP_SIZE=20 (local patch).
for _tp, _k in [(4, 3), (4, 4), (4, 5), (4, 7), (1, 7), (2, 7)]:
    CONFIGS[f"TP{_tp}_DFLASH_HYB_K{_k}"] = (["-tp", str(_tp),
        "--speculative-config",
        json.dumps({"model": str(DRAFTER), "method": "dflash",
                    "num_speculative_tokens": _k})], _tp)
CONFIGS["TP4_DFLASH_HYB_K7_LAR"] = (["-tp", "4", "--speculative-config",
    json.dumps({"model": str(DRAFTER), "method": "dflash",
                "num_speculative_tokens": 7,
                "use_local_argmax_reduction": True})], 4)


def random_wl(inp: int, out: int) -> list[str]:
    return [
        "--dataset-name", "random",
        "--random-input-len", str(inp),
        "--random-output-len", str(out),
        "--ignore-eos",
    ]


def shared_prefix_wl(prefix: int, inp: int, out: int) -> list[str]:
    """Random prompts of `inp` total tokens whose first `prefix` are shared."""
    return random_wl(inp - prefix, out) + ["--random-prefix-len", str(prefix)]


def sharegpt_shape_wl(inp: int, out: int) -> list[str]:
    """ShareGPT prompts within +/-20% of `inp` tokens (make_sharegpt_shapes.py)."""
    return [
        "--dataset-name", "custom",
        "--dataset-path", str(ROOT / f"data/sharegpt_in{inp}.jsonl"),
        "--skip-chat-template",
        "--custom-output-len", str(out),
        "--ignore-eos",
    ]


CHAT = ["--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
        "--skip-chat-template", "--temperature", "0"]


def hf_wl(path: str, split: str, subset: str | None = None) -> list[str]:
    args = CHAT + ["--dataset-name", "hf", "--dataset-path", path,
                   "--hf-split", split, "--hf-output-len", "2048"]
    return args + (["--hf-subset", subset] if subset else [])


WORKLOADS: dict[str, list[str]] = {
    "gsm8k": hf_wl("openai/gsm8k", "test", "main"),
    "humaneval": hf_wl("openai/openai_humaneval", "test"),
    "mtbench": hf_wl("philschmid/mt-bench", "train"),
    "sharegpt_chat": CHAT + ["--dataset-name", "sharegpt",
                             "--dataset-path", str(SHAREGPT)],
    "rand_1k_1k": random_wl(1024, 1024),
    "rand_1k_128": random_wl(1024, 128),
    "rand_128_1k": random_wl(128, 1024),
    "sharegpt": [
        "--dataset-name", "sharegpt",
        "--dataset-path", str(SHAREGPT),
        "--ignore-eos",
    ],
    "rand_128_128": random_wl(128, 128),
    "rand_2k_128": random_wl(2048, 128),
    "rand_pfx512_1k_128": shared_prefix_wl(512, 1024, 128),
    "rand_pfx512_2k_128": shared_prefix_wl(512, 2048, 128),
    "sgpt_128_128": sharegpt_shape_wl(128, 128),
    "sgpt_128_1k": sharegpt_shape_wl(128, 1024),
    "sgpt_1k_128": sharegpt_shape_wl(1024, 128),
    "sgpt_2k_128": sharegpt_shape_wl(2048, 128),
}

# Distinct prompts available; points never sample more than this.
DATASET_SIZE = {"humaneval": 164, "mtbench": 80}

PHASES = {
    "p1": dict(
        configs=[c for c in CONFIGS if c != "TP1"],
        workloads=["rand_1k_1k", "sharegpt"],
        concurrency=[1, 8, 32],
        prompts_per_user=3,
        min_prompts=6,
    ),
    "p2": dict(
        configs=[],
        workloads=["rand_1k_1k", "rand_1k_128", "rand_128_1k", "sharegpt"],
        concurrency=[1, 2, 4, 8, 16, 32],
        prompts_per_user=4,
        min_prompts=8,
    ),
    "p3": dict(
        configs=["TP4", "TP2_PP3"],
        workloads=["rand_1k_1k", "sharegpt"],
        concurrency=[8, 32],
        prompts_per_user=3,
        min_prompts=6,
    ),
    "p4": dict(
        configs=["TP4_DFLASH", "TP4"],
        workloads=["gsm8k", "humaneval", "mtbench", "sharegpt_chat"],
        concurrency=[1, 2, 4, 8, 16, 32],
        prompts_per_user=4,
        min_prompts=8,
        accuracy=True,
    ),
    "p5": dict(
        configs=["TP4"],
        workloads=["rand_128_128", "rand_128_1k", "rand_1k_128", "rand_2k_128",
                   "rand_pfx512_1k_128", "rand_pfx512_2k_128",
                   "sgpt_128_128", "sgpt_128_1k", "sgpt_1k_128", "sgpt_2k_128"],
        concurrency=[1, 2, 4, 8, 16, 32],
        prompts_per_user=4,
        min_prompts=8,
        prefix_caching=True,
    ),
    "p6": dict(
        configs=["TP4", "TP4_DFLASH_HYB_K7"],
        workloads=["gsm8k", "humaneval", "mtbench"],
        concurrency=[1, 2, 4, 8, 16, 32],
        prompts_per_user=4,
        min_prompts=16,
        accuracy=True,
        prefix_caching=True,
    ),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def server_env(name: str, nodes: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        LD_PRELOAD=LD_PRELOAD,
        VLLM_CPU_KVCACHE_SPACE="40",
        VLLM_CPU_OMP_THREADS_BIND="auto",
        CPU_VISIBLE_MEMORY_NODES=",".join(str(n) for n in range(nodes)),
        HF_HUB_OFFLINE="1",
        VLLM_SERVER_DEV_MODE="1",
    )
    if "DFLASH_HYB" in name:
        env["VLLM_KV_GROUP_SIZE"] = "20"
    return env


def reset_prefix_cache() -> None:
    req = urllib.request.Request(f"http://localhost:{PORT}/reset_prefix_cache",
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60):
        pass


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2):
            return True
    except Exception:
        return False


def start_server(name: str, args: list[str], nodes: int, logf: Path):
    if healthy():
        log(f"{name}: port {PORT} is already in use, not starting")
        return None
    cmd = [VLLM, "serve", str(MODEL), "--served-model-name", SERVED_NAME,
           "--port", str(PORT), *BASE_SERVE_ARGS, *args]
    log(f"start {name}: {' '.join(cmd[2:])}")
    (logf.parent).mkdir(parents=True, exist_ok=True)
    fh = open(logf, "w")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, env=server_env(name, nodes), stdout=fh,
                            stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 1200
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"{name}: server exited with {proc.returncode} during startup")
            return None
        if healthy():
            log(f"{name}: ready")
            return proc
        time.sleep(5)
    log(f"{name}: startup timed out")
    stop_server(proc)
    return None


def stop_server(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass
    for _ in range(30):
        if not healthy():
            break
        time.sleep(1)


def run_bench(bench_args: list[str], out_dir: Path, fname: str, logf: Path,
              timeout: int) -> bool:
    cmd = [VLLM, "bench", "serve", "--backend", "vllm", "--model", SERVED_NAME,
           "--tokenizer", str(MODEL), "--port", str(PORT), "--seed", "0",
           "--disable-tqdm", "--save-result", "--result-dir", str(out_dir),
           "--result-filename", fname,
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,90,99", *bench_args]
    with open(logf, "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env=dict(os.environ, HF_HUB_OFFLINE="1"))
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            log(f"  bench timed out after {timeout}s")
            return False
    return rc == 0 and (out_dir / fname).exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    ap.add_argument("--configs", nargs="*", help="override config list")
    ap.add_argument("--workloads", nargs="*")
    ap.add_argument("--concurrency", nargs="*", type=int)
    ap.add_argument("--serve-args", default="",
                    help="extra serve args appended to every config")
    ap.add_argument("--tag", default="", help="suffix for the results dir")
    ap.add_argument("--point-timeout", type=int, default=2400)
    ap.add_argument("--no-accuracy", action="store_true")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if healthy():
        sys.exit(f"something is already serving on port {PORT}")
    (ROOT / "logs" / f"{a.phase}.pid").write_text(str(os.getpid()))

    ph = PHASES[a.phase]
    configs = a.configs or ph["configs"]
    workloads = a.workloads or ph["workloads"]
    concs = a.concurrency or ph["concurrency"]
    if not configs:
        sys.exit("no configs selected; pass --configs")
    prefix_caching = ph.get("prefix_caching", False)
    extra = ([] if prefix_caching else NO_PREFIX_CACHING) + a.serve_args.split()
    out_root = ROOT / "results" / (a.phase + (f"_{a.tag}" if a.tag else ""))

    for name in configs:
        cfg_args, nodes = CONFIGS[name]
        cdir = out_root / name
        todo = [(w, c) for w in workloads for c in concs
                if not (cdir / f"{w}_c{c}.json").exists()]
        if not todo:
            log(f"{name}: all points done, skipping")
            continue
        proc = start_server(name, cfg_args + extra, nodes, cdir / "server.log")
        if proc is None:
            (cdir / "FAILED_STARTUP").touch()
            continue
        try:
            run_bench(random_wl(128, 128) + ["--max-concurrency", "4",
                      "--num-prompts", "8"], cdir, "warmup.json",
                      cdir / "warmup.log", 900)
            acc_out = cdir / "gsm8k_accuracy.json"
            if ph.get("accuracy") and not a.no_accuracy and not acc_out.exists():
                log(f"  {name} GSM8K accuracy check")
                with open(cdir / "gsm8k_accuracy.log", "w") as fh:
                    subprocess.run([str(VENV / "bin/python"),
                                    str(ROOT / "acc_gsm8k.py"),
                                    "--port", str(PORT), "--out", str(acc_out)],
                                   stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=3600, check=False)
            for w, c in todo:
                n = max(ph["min_prompts"], ph["prompts_per_user"] * c)
                n = min(n, DATASET_SIZE.get(w, n))
                if prefix_caching:
                    reset_prefix_cache()
                fname = f"{w}_c{c}.json"
                log(f"  {name} {w} conc={c} prompts={n}")
                t0 = time.time()
                detail = ["--save-detailed"] if ph.get("accuracy") else []
                ok = run_bench(
                    WORKLOADS[w] + detail + ["--max-concurrency", str(c),
                                    "--num-prompts", str(n),
                                    "--metadata", f"config={name}",
                                    f"workload={w}", f"serve_extra={a.serve_args}"],
                    cdir, fname, cdir / f"{w}_c{c}.log", a.point_timeout)
                log(f"  -> {'ok' if ok else 'FAILED'} in {time.time() - t0:.0f}s")
                if not ok and proc.poll() is not None:
                    log(f"{name}: server died, moving on")
                    break
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
