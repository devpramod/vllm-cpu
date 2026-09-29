"""LightOnOCR-2-1B replicas behind vllm-router.

Results: results/ocr/replicas/<layout>/<policy>/<wl>_c<c>.json
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "ocr"))
import replicas as R  # noqa: E402
import sweep as S  # noqa: E402
import sweep_ocr as O  # noqa: E402

BASE_PORT = 8200


def half(node: int, i: int) -> str:
    lo = 32 * node + 16 * i
    return f"{lo}-{lo + 14}"


# layout -> [(numa node, OMP bind list or "auto")]
LAYOUTS = {
    "TP1x6": [(n, "auto") for n in range(6)],
    "TP1x12": [(n, half(n, i)) for n in range(6) for i in range(2)],
}
WORKLOADS = {k: O.FULL[k] for k in ("omnidoc", "cord")}
POLICIES = ["round_robin", "power_of_two"]
CONCS = [1, 4, 16, 32, 64, 128, 256]


def start_replicas(layout, out):
    reps = []
    for i, (node, bind) in enumerate(LAYOUTS[layout]):
        port = BASE_PORT + i
        if R.ok(f"http://localhost:{port}/health"):
            raise RuntimeError(f"port {port} already in use")
        cmd = [S.VLLM, "serve", str(O.MODEL), "--served-model-name", "lightonocr",
               "--port", str(port), *O.BASE_ARGS]
        env = S.server_env("TP1", 1)
        env.update(CPU_VISIBLE_MEMORY_NODES=str(node), VLLM_CPU_OMP_THREADS_BIND=bind)
        fh = open(out / f"replica{i}_n{node}.log", "w")
        fh.write(" ".join(cmd) + f"\nnode={node} bind={bind}\n")
        fh.flush()
        reps.append((subprocess.Popen(cmd, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT,
                                      start_new_session=True), port))
    pending = {p for _, p in reps}
    deadline = time.time() + 1500
    while pending and time.time() < deadline:
        for proc, port in reps:
            if proc.poll() is not None:
                raise RuntimeError(f"replica on port {port} exited "
                                   f"({proc.returncode})")
            if port in pending and R.ok(f"http://localhost:{port}/health"):
                pending.discard(port)
        time.sleep(5)
    if pending:
        raise RuntimeError(f"replicas on ports {sorted(pending)} not ready")
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    a = ap.parse_args()
    root = O.RESULTS / "replicas"
    for layout in a.layouts:
        ldir = root / layout
        todo = {p: [(w, c) for w in WORKLOADS for c in CONCS
                    if not (ldir / p / f"{w}_c{c}.json").exists()]
                for p in a.policies}
        if not any(todo.values()):
            continue
        ldir.mkdir(parents=True, exist_ok=True)
        reps = []
        try:
            reps = start_replicas(layout, ldir)
            ports = [p for _, p in reps]
            for p in ports:
                O.bench(p, "", "cord", ["--num-pages", "2"], 1,
                        ldir / f"warmup_{p}.json")
            O.log(f"{layout}: {len(ports)} replicas ready")
            for policy, points in todo.items():
                if not points:
                    continue
                pdir = ldir / policy
                pdir.mkdir(exist_ok=True)
                try:
                    router = R.start_router(policy, ports, ldir)
                except RuntimeError as e:
                    O.log(f"{layout} {policy}: {e}")
                    continue
                try:
                    for w, c in points:
                        page_set, hargs, _ = WORKLOADS[w]
                        t = time.time()
                        rc = O.bench(S.PORT, "", page_set, hargs, c,
                                     pdir / f"{w}_c{c}.json")
                        O.log(f"{layout} {policy} {w} c={c}: rc={rc} "
                              f"{time.time() - t:.0f}s")
                        if any(pr.poll() is not None for pr, _ in reps):
                            raise RuntimeError("a replica died")
                finally:
                    R.killpg(router)
        except Exception as e:  # noqa: BLE001
            O.log(f"{layout}: {e}")
            (ldir / "FAILED").write_text(str(e))
        finally:
            for proc, _ in reps:
                R.killpg(proc)
            time.sleep(10)


if __name__ == "__main__":
    main()
