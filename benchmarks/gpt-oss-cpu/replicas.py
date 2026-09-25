#!/usr/bin/env python
"""Multi-replica sweep: several vLLM instances on disjoint NUMA nodes behind
vllm-router. Users (concurrency) are counted at the router.

Results: results/p7/<layout>/<policy>/<workload>_c<c>.json
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

import sweep as S

ROUTER = str(S.VENV / "bin/vllm-router")
MT_DIR = S.ROOT / "multi_turn"
MT_DATA = S.ROOT / "data/sharegpt_multiturn.json"
BASE_PORT = 8200

# layout -> [(tp, numa nodes)]; sockets are nodes 0-2 and 3-5.
LAYOUTS: dict[str, list[tuple[int, list[int]]]] = {
    "TP1x6": [(1, [n]) for n in range(6)],
    "TP2x3": [(2, [0, 1]), (2, [3, 4]), (2, [2, 5])],
    "TP2x2": [(2, [0, 1]), (2, [3, 4])],
    "TP2x2_TP1x2": [(2, [0, 1]), (2, [3, 4]), (1, [2]), (1, [5])],
    "TP4_TP1x2": [(4, [0, 1, 2, 3]), (1, [4]), (1, [5])],
    "TP4_TP2": [(4, [0, 1, 2, 3]), (2, [4, 5])],
    "TP4x1": [(4, [0, 1, 2, 3])],
}
# workload -> routing policies to run it under
PLAN = {
    "rand_128_1k": ["cache_aware"],
    "rand_1k_128": ["cache_aware"],
    "multiturn": ["round_robin", "cache_aware"],
}
CONCS = [1, 2, 4, 8, 16, 32]


def log(msg: str) -> None:
    S.log(msg)


def ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False


def post(url: str, body: dict | None = None, timeout: int = 120) -> None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout):
        pass


def killpg(proc) -> None:
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


def start_replicas(layout: str, out: Path) -> list[tuple[subprocess.Popen, int]]:
    reps = []
    for i, (tp, nodes) in enumerate(LAYOUTS[layout]):
        port = BASE_PORT + i
        if ok(f"http://localhost:{port}/health"):
            raise RuntimeError(f"port {port} already in use")
        cmd = [S.VLLM, "serve", str(S.MODEL), "--served-model-name",
               S.SERVED_NAME, "--port", str(port), *S.BASE_SERVE_ARGS,
               "-tp", str(tp)]
        env = S.server_env(f"TP{tp}", len(nodes))
        env["CPU_VISIBLE_MEMORY_NODES"] = ",".join(map(str, nodes))
        logf = out / f"replica{i}_tp{tp}_n{'-'.join(map(str, nodes))}.log"
        fh = open(logf, "w")
        fh.write(" ".join(cmd) + f"\nnodes={nodes}\n")
        fh.flush()
        log(f"  replica {i}: TP{tp} on nodes {nodes}, port {port}")
        reps.append((subprocess.Popen(cmd, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT,
                                      start_new_session=True), port))
    deadline = time.time() + 1500
    pending = {port for _, port in reps}
    while pending and time.time() < deadline:
        for proc, port in reps:
            if proc.poll() is not None:
                raise RuntimeError(f"replica on port {port} exited "
                                   f"({proc.returncode})")
            if port in pending and ok(f"http://localhost:{port}/health"):
                pending.discard(port)
        time.sleep(5)
    if pending:
        raise RuntimeError(f"replicas on ports {sorted(pending)} not ready")
    return reps


def start_router(policy: str, ports: list[int], out: Path) -> subprocess.Popen:
    if ok(f"http://localhost:{S.PORT}/health"):
        raise RuntimeError(f"port {S.PORT} already in use")
    cmd = [ROUTER, "--host", "127.0.0.1", "--port", str(S.PORT), "--policy",
           policy, "--worker-urls", *[f"http://127.0.0.1:{p}" for p in ports]]
    fh = open(out / "router.log", "a")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(f"router exited ({proc.returncode})")
        if ok(f"http://localhost:{S.PORT}/v1/models"):
            return proc
        time.sleep(1)
    killpg(proc)
    raise RuntimeError("router not ready")


def warmup(ports: list[int]) -> None:
    for p in ports:
        for _ in range(2):
            post(f"http://localhost:{p}/v1/completions",
                 {"model": S.SERVED_NAME, "prompt": "Hello " * 100,
                  "max_tokens": 32}, timeout=600)


def run_multiturn(c: int, out_json: Path, logf: Path, timeout: int) -> bool:
    n = max(32, 8 * c)
    cmd = [str(S.VENV / "bin/python"), "benchmark_serving_multi_turn.py",
           "-i", str(MT_DATA), "-m", str(S.MODEL),
           "--served-model-name", S.SERVED_NAME,
           "-u", f"http://localhost:{S.PORT}", "-p", str(c),
           "-k", str(2 * c), "-n", str(n), "--seed", "0",
           "--request-timeout-sec", "900", "--no-early-stop",
           "--stats-json-output", str(out_json)]
    with open(logf, "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=MT_DIR, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, HF_HUB_OFFLINE="1"))
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            log("  multi-turn timed out")
            return False
    return rc == 0 and out_json.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    ap.add_argument("--workloads", nargs="*", default=list(PLAN))
    ap.add_argument("--concurrency", nargs="*", type=int, default=CONCS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--point-timeout", type=int, default=2400)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    (S.ROOT / "logs/p7.pid").write_text(str(os.getpid()))
    root = S.ROOT / "results" / ("p7" + (f"_{a.tag}" if a.tag else ""))

    for layout in a.layouts:
        ldir = root / layout
        policies = ["round_robin"] if len(LAYOUTS[layout]) == 1 else None

        def todo_for(policy: str) -> list[tuple[str, int]]:
            return [(w, c) for w in a.workloads for c in a.concurrency
                    if policy in (policies or PLAN[w])
                    and not (ldir / policy / f"{w}_c{c}.json").exists()]

        all_policies = sorted({p for w in a.workloads
                               for p in (policies or PLAN[w])})
        if not any(todo_for(p) for p in all_policies):
            log(f"{layout}: all points done, skipping")
            continue
        ldir.mkdir(parents=True, exist_ok=True)
        log(f"start layout {layout}")
        reps = []
        try:
            reps = start_replicas(layout, ldir)
            ports = [p for _, p in reps]
            warmup(ports)
            log(f"{layout}: {len(ports)} replicas ready")
            for policy in all_policies:
                todo = todo_for(policy)
                if not todo:
                    continue
                pdir = ldir / policy
                pdir.mkdir(exist_ok=True)
                router = start_router(policy, ports, ldir)
                try:
                    for w, c in todo:
                        for p in ports:
                            post(f"http://localhost:{p}/reset_prefix_cache")
                        log(f"  {layout} {policy} {w} conc={c}")
                        t0 = time.time()
                        fname = f"{w}_c{c}.json"
                        if w == "multiturn":
                            done = run_multiturn(c, pdir / fname,
                                                 pdir / f"{w}_c{c}.log",
                                                 a.point_timeout)
                        else:
                            n = max(8, 4 * c)
                            done = S.run_bench(
                                S.WORKLOADS[w] + [
                                    "--max-concurrency", str(c),
                                    "--num-prompts", str(n), "--metadata",
                                    f"layout={layout}", f"policy={policy}",
                                    f"workload={w}"],
                                pdir, fname, pdir / f"{w}_c{c}.log",
                                a.point_timeout)
                        log(f"  -> {'ok' if done else 'FAILED'} "
                            f"in {time.time() - t0:.0f}s")
                        if not done and any(pr.poll() is not None
                                            for pr, _ in reps):
                            raise RuntimeError("a replica died")
                finally:
                    killpg(router)
        except Exception as e:
            log(f"{layout}: {e}")
            (ldir / "FAILED").write_text(str(e))
        finally:
            for proc, _ in reps:
                killpg(proc)
            time.sleep(10)


if __name__ == "__main__":
    main()
