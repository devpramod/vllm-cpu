"""LightOnOCR-2-1B single-instance sweep. Results: results/ocr/<config>/<wl>_c<c>.json.

Lanes run independent configs on disjoint sockets so they can execute in parallel.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
import sweep as S  # noqa: E402

MODEL = BENCH / "models" / "LightOnOCR-2-1B"
RESULTS = BENCH / "results" / "ocr"
BASE_ARGS = ["--dtype", "bfloat16", "--max-model-len", "8192",
             "--limit-mm-per-prompt", '{"image":1}', "--mm-processor-cache-gb", "0",
             "--no-enable-prefix-caching"]
CONC = [1, 2, 4, 8, 16, 32, 64, 128]
SYN_CONC = [1, 4, 16, 64, 128]
GREEDY = ["--temperature", "0"]

# name -> (serve args, NUMA nodes)
CONFIGS = {
    "TP1": ([], [0]),
    "TP1_MNBT4K": (["--max-num-batched-tokens", "4096"], [0]),
    "TP1_MNBT8K": (["--max-num-batched-tokens", "8192"], [0]),
    "TP1_MNBT16K": (["--max-num-batched-tokens", "16384"], [0]),
    "TP2": (["-tp", "2"], [3, 4]),
    "TP2_DATA": (["-tp", "2", "--mm-encoder-tp-mode", "data"], [3, 4]),
    "TP4": (["-tp", "4"], [0, 1, 2, 3]),
}

# name -> (page set, harness args, concurrencies)
FULL = {
    "omnidoc": ("omnidoc", ["--save-text"], CONC),
    "olmocr": ("olmocr", ["--save-text"], CONC),
    "cord": ("cord", ["--save-text"], CONC),
    "omnidoc_greedy": ("omnidoc", ["--save-text", *GREEDY], [1, 16, 128]),
    "syn768_o512": ("synth_768x768", ["--ignore-eos", "--max-tokens", "512"],
                    SYN_CONC),
    "syn1024_o512": ("synth_1024x1024", ["--ignore-eos", "--max-tokens", "512"],
                     SYN_CONC),
    "syn1540_o512": ("synth_1190x1540", ["--ignore-eos", "--max-tokens", "512"],
                     SYN_CONC),
    "syn1024_o128": ("synth_1024x1024", ["--ignore-eos", "--max-tokens", "128"],
                     SYN_CONC),
    "syn1024_o2048": ("synth_1024x1024", ["--ignore-eos", "--max-tokens", "2048"],
                      SYN_CONC),
}
MNBT = {
    "omnidoc": ("omnidoc", ["--save-text"], [8, 32, 128]),
    "cord": ("cord", ["--save-text"], [32, 128]),
    "syn1024_o128": FULL["syn1024_o128"][:2] + ([16, 128],),
}
LANES = {
    "s0": [("TP1", FULL), ("TP1_MNBT4K", MNBT), ("TP1_MNBT8K", MNBT),
           ("TP1_MNBT16K", MNBT)],
    "s1": [("TP2", FULL), ("TP2_DATA", FULL)],
    "all": [("TP4", FULL)],
}
LANE_PORT = {"s0": 8110, "s1": 8111, "all": 8112}
LANE_CLIENT_CPUS = {"s0": "64-95", "s1": "160-191", "all": "160-191"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2):
            return True
    except Exception:
        return False


def start(name, port):
    args, nodes = CONFIGS[name]
    env = S.server_env(name, len(nodes))
    env["CPU_VISIBLE_MEMORY_NODES"] = ",".join(map(str, nodes))
    cmd = [S.VLLM, "serve", str(MODEL), "--served-model-name", "lightonocr",
           "--port", str(port), *BASE_ARGS, *args]
    logf = RESULTS / name / "server.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    fh = open(logf, "w")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.time() + 1200
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"{name}: server exited with {proc.returncode} during startup")
            return None
        if healthy(port):
            log(f"{name}: ready")
            return proc
        time.sleep(5)
    stop(proc)
    return None


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def bench(port, cpus, page_set, hargs, c, out):
    cmd = (["taskset", "-c", cpus] if cpus else []) + [str(S.VENV / "bin/python"),
           str(BENCH / "ocr/bench_ocr.py"), "--port", str(port), "--set", page_set,
           "--concurrency", str(c), "--pages-per-user", "2", "--min-pages", "8",
           "--out", str(out), "--records", str(out.with_suffix(".rec.jsonl")),
           *hargs]
    with open(out.with_suffix(".log"), "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def run_config(name, workloads, port, cpus):
    todo = [(wl, c) for wl, (_, _, cs) in workloads.items() for c in cs
            if not (RESULTS / name / f"{wl}_c{c}.json").exists()]
    if not todo:
        log(f"{name}: all done")
        return
    proc = start(name, port)
    if proc is None:
        return
    try:
        # One warm-up page so the first measured point excludes graph/alloc setup.
        bench(port, cpus, "cord", ["--num-pages", "2"], 1,
              RESULTS / name / "warmup.json")
        for wl, c in todo:
            page_set, hargs, _ = workloads[wl]
            out = RESULTS / name / f"{wl}_c{c}.json"
            t = time.time()
            rc = bench(port, cpus, page_set, hargs, c, out)
            log(f"{name} {wl} c={c}: rc={rc} {time.time() - t:.0f}s")
            if not healthy(port):
                log(f"{name}: server unhealthy, aborting config")
                break
    finally:
        stop(proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=list(LANES), required=True)
    ap.add_argument("--configs", nargs="*")
    a = ap.parse_args()
    for name, wls in LANES[a.lane]:
        if a.configs and name not in a.configs:
            continue
        run_config(name, wls, LANE_PORT[a.lane], LANE_CLIENT_CPUS[a.lane])
    log(f"lane {a.lane} done")


if __name__ == "__main__":
    main()
