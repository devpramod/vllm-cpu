#!/usr/bin/env python
"""Per-TP DFlash vs baseline: phase timings at 1 and 16 users plus a torch
profiler trace at 1 user. One server per config; results in results/profile_tp.
"""

import argparse
import json
import shutil
import subprocess
import time
import urllib.request

import sweep

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", required=True)
a = ap.parse_args()

ROOT = sweep.ROOT / "results" / "profile_tp"


def post(path: str) -> None:
    req = urllib.request.Request(f"http://localhost:{sweep.PORT}{path}",
                                 method="POST")
    urllib.request.urlopen(req, timeout=1800).read()


def gsm8k(c: int, n: int, extra: list[str] | None = None) -> list[str]:
    return sweep.WORKLOADS["gsm8k"] + ["--max-concurrency", str(c),
                                       "--num-prompts", str(n)] + (extra or [])


def snapshot(out_dir, name: str) -> None:
    time.sleep(2)
    stats = {p.stem: json.loads(p.read_text())
             for p in out_dir.glob("[0-9]*.json")}
    (out_dir / f"steptimer_{name}.json").write_text(json.dumps(stats, indent=1))


for name in a.configs:
    args, nodes = sweep.CONFIGS[name]
    out = ROOT / name
    shutil.rmtree(out, ignore_errors=True)
    (out / "trace").mkdir(parents=True)
    prof = {"profiler": "torch", "torch_profiler_dir": str(out / "trace"),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_use_gzip": False, "ignore_frontend": True,
            "delay_iterations": 5, "max_iterations": 60}
    env = sweep.server_env(nodes) | {"STEPTIMER_DIR": str(out),
                                     "STEPTIMER_RF": "1"}
    if "DFLASH_HYB" in name:
        env["VLLM_KV_GROUP_SIZE"] = "20"
    cmd = [sweep.VLLM, "serve", str(sweep.MODEL), "--served-model-name",
           sweep.SERVED_NAME, "--port", str(sweep.PORT),
           *sweep.BASE_SERVE_ARGS, *args, "--profiler-config", json.dumps(prof)]
    assert not sweep.healthy(), "port busy"
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 1200
        while not sweep.healthy():
            assert proc.poll() is None and time.time() < deadline, "startup failed"
            time.sleep(5)
        sweep.log(f"{name}: ready")
        sweep.run_bench(sweep.random_wl(128, 64) + ["--max-concurrency", "4",
                        "--num-prompts", "8"], out, "warmup.json",
                        out / "warmup.log", 600)
        for c, n in [(1, 8), (16, 48)]:
            (out / "RESET").touch()
            sweep.run_bench(gsm8k(c, n), out, f"bench_c{c}.json",
                            out / f"bench_c{c}.log", 1800)
            snapshot(out, f"c{c}")
            sweep.log(f"{name}: c={c} done")
        post("/start_profile")
        sweep.run_bench(gsm8k(1, 2, ["--hf-output-len", "256"]), out,
                        "bench_profile.json", out / "bench_profile.log", 1800)
        post("/stop_profile")
        sweep.log(f"{name}: trace written")
    except Exception as e:  # keep going with the next config
        sweep.log(f"{name}: FAILED {e!r}")
    finally:
        sweep.stop_server(proc)
        time.sleep(10)
