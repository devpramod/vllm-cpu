#!/usr/bin/env python3
"""On-worker run harness. stdlib only; executed by the repo venv python.

Reads a fully-resolved job.json (shipped by labctl), takes an exclusive
flock on the runs dir for the whole run (benchmarks own the box), checks out
the requested SHA, executes every arm x repeat, and maintains
<run_dir>/remote_state.json for labctl status/sync to read.

Placeholders left for this script to resolve per repeat:
  {rep_dir}      absolute dir for this repeat's outputs
  {metrics_json} {rep_dir}/metrics.json
"""

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

COMPILE_RE = re.compile(r"torch\.compile takes ([0-9.]+)|Compilation took ([0-9.]+)")


def dotted(obj, path):
    try:
        return reduce(lambda o, k: o[int(k)] if isinstance(o, list) else o[k],
                      path.split("."), obj)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, job):
        self.job = job
        self.run_dir = Path(job["run_dir"])
        self.repo = Path(job["repo_dir"])
        self.state_path = self.run_dir / "remote_state.json"
        self.state = {
            "id": job["id"],
            "status": "queued",
            "current": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "arms": {a: {"status": "pending", "compile_time_s": None, "reps": []}
                     for a in job["arms"]},
        }
        self.env = dict(os.environ)
        self.env.update({k: str(v) for k, v in job.get("env", {}).items()})
        venv_bin = str(self.repo / ".venv" / "bin")
        self.env["PATH"] = venv_bin + os.pathsep + self.env.get("PATH", "")
        for var in job.get("forbidden_env", []):
            if var in self.env:
                raise SystemExit(f"forbidden env var set on worker: {var}")

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def sh(self, cmd, log_path, cwd=None):
        with open(log_path, "a") as log:
            log.write(f"\n$ {cmd}\n")
            log.flush()
            return subprocess.run(cmd, shell=True, cwd=cwd or self.repo,
                                  env=self.env, stdout=log, stderr=log).returncode

    def checkout(self):
        for cmd in [f"git fetch origin", f"git checkout --detach {self.job['sha']}"]:
            rc = self.sh(cmd, self.run_dir / "runner.git.log")
            if rc != 0:
                raise RuntimeError(f"git step failed: {cmd}")

    def port_open(self):
        u = urllib.parse.urlparse(self.job["health_url"])
        try:
            with socket.create_connection((u.hostname, u.port or 80), timeout=2):
                return True
        except OSError:
            return False

    def stop_server(self, server):
        """Kill the server group and wait for the PORT to close, not just the
        shell: vLLM traps SIGTERM for graceful shutdown and can hold the port
        long after the wrapper shell is reaped."""
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.time() + 120
        while time.time() < deadline and self.port_open():
            time.sleep(5)
        if self.port_open():
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.time() + 60
            while time.time() < deadline and self.port_open():
                time.sleep(5)
        if self.port_open():
            raise RuntimeError("server port still open after SIGKILL")

    def wait_healthy(self, proc, url, timeout_s, log_path):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}, see {log_path}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(10)
        raise RuntimeError(f"server not healthy after {timeout_s}s")

    def grep_compile_time(self, *log_paths):
        for p in log_paths:
            p = Path(p)
            if not p.exists():
                continue
            for m in COMPILE_RE.finditer(p.read_text(errors="replace")):
                return float(m.group(1) or m.group(2))
        return None

    def resolve(self, template, rep_dir):
        return (template
                .replace("{rep_dir}", str(rep_dir))
                .replace("{metrics_json}", str(rep_dir / "metrics.json")))

    def run_arm(self, name, arm):
        arm_state = self.state["arms"][name]
        arm_state["status"] = "running"
        arm_dir = self.run_dir / "arms" / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        server = None
        server_log = arm_dir / "server.log"
        try:
            if arm.get("server_command"):
                if self.port_open():
                    raise RuntimeError(
                        "health port already in use before server start "
                        "(tenant process or leftover server from a previous arm)")
                with open(server_log, "w") as log:
                    server = subprocess.Popen(
                        arm["server_command"], shell=True, cwd=self.repo,
                        env=self.env, stdout=log, stderr=log,
                        start_new_session=True)
                self.wait_healthy(server, self.job["health_url"],
                                  self.job.get("health_timeout_s", 5400), server_log)
            for rep in range(1, self.job["repeats"] + 1):
                self.state["current"] = f"{name}/rep{rep}"
                self.save()
                rep_dir = arm_dir / f"rep{rep}"
                rep_dir.mkdir(exist_ok=True)
                cmd = self.resolve(arm["command"], rep_dir)
                rc = self.sh(cmd, rep_dir / "bench.log")
                metrics = rep_dir / "metrics.json"
                ok = rc == 0 and metrics.exists()
                why = f"rc={rc}"
                sanity = self.job.get("sanity_metric")
                if ok and sanity:
                    v = dotted(json.loads(metrics.read_text()), sanity)
                    if not isinstance(v, (int, float)) or v <= 0:
                        ok = False
                        why = f"sanity: {sanity}={v} (benchmark ran but measured nothing real)"
                arm_state["reps"].append({"rep": rep, "ok": ok})
                self.save()
                if not ok:
                    raise RuntimeError(f"{name}/rep{rep} failed ({why})")
            first_rep_log = arm_dir / "rep1" / "bench.log"
            arm_state["compile_time_s"] = self.grep_compile_time(server_log, first_rep_log)
            arm_state["status"] = "done"
        except Exception:
            arm_state["status"] = "failed"
            raise
        finally:
            if server:
                self.stop_server(server)
            self.save()

    def execute(self):
        lock_path = Path(self.job["runs_dir"]) / ".lock"
        lock = open(lock_path, "w")
        self.save()
        fcntl.flock(lock, fcntl.LOCK_EX)  # blocks: FIFO queue behind current run
        lock.truncate(0)
        lock.write(f"{self.job['id']} pid={os.getpid()}\n")
        lock.flush()
        self.state["status"] = "running"
        self.state["started_at"] = now()
        self.save()
        try:
            self.checkout()
            for name, arm in self.job["arms"].items():
                self.run_arm(name, arm)
            self.state["status"] = "done"
        except Exception as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            raise
        finally:
            self.state["current"] = None
            self.state["finished_at"] = now()
            self.save()
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="path to job.json")
    args = ap.parse_args()
    job = json.loads(Path(args.job).read_text())
    Run(job).execute()


if __name__ == "__main__":
    sys.exit(main())
