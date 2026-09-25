#!/usr/bin/env python
"""Run a GSM8K load with the steptimer plugin enabled and save per-phase timings."""

import argparse
import shutil
import subprocess
import time

import sweep

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--prompts", type=int, default=8)
a = ap.parse_args()

args, nodes = sweep.CONFIGS[a.config]
tag = f"{a.config}_c{a.concurrency}"
out_dir = sweep.ROOT / "results" / "profile" / tag
shutil.rmtree(out_dir, ignore_errors=True)
out_dir.mkdir(parents=True)

env = sweep.server_env(nodes) | {"STEPTIMER_DIR": str(out_dir)}
serve = [sweep.VLLM, "serve", str(sweep.MODEL), "--served-model-name",
         sweep.SERVED_NAME, "--port", str(sweep.PORT),
         *sweep.BASE_SERVE_ARGS, *args]
assert not sweep.healthy(), "port busy"
log = open(out_dir / "server.log", "w")
proc = subprocess.Popen(serve, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
try:
    while not sweep.healthy():
        assert proc.poll() is None, "server died"
        time.sleep(5)
    sweep.log(f"{tag}: ready")
    sweep.run_bench(sweep.random_wl(128, 64) + ["--max-concurrency", "4",
                    "--num-prompts", "8"], out_dir, "warmup.json",
                    out_dir / "warmup.log", 600)
    (out_dir / "RESET").touch()
    bench = sweep.WORKLOADS["gsm8k"] + ["--max-concurrency", str(a.concurrency),
                                        "--num-prompts", str(a.prompts)]
    sweep.run_bench(bench, out_dir, "bench.json", out_dir / "bench.log", 1800)
    sweep.log(f"{tag}: bench done")
finally:
    sweep.stop_server(proc)
