#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn", "jinja2", "pyyaml"]
# ///
"""Lab UI — read-only ledger over planning/lab/runs/. No DB; git is the store.

Run via: ./labctl ui   (or: uv run server/app.py --port 8800)
"""

import argparse
import json
import subprocess
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

LAB = Path(__file__).resolve().parent.parent
RUNS = LAB / "runs"
CACHE = LAB / "artifact-cache"

app = FastAPI(title="torch.compile CPU lab")
templates = Jinja2Templates(directory=str(LAB / "server" / "templates"))
if CACHE.is_dir():
    app.mount("/artifacts", StaticFiles(directory=str(CACHE)), name="artifacts")


_subjects: dict = {}


def commit_subject(sha):
    if sha not in _subjects:
        r = subprocess.run(["git", "log", "-1", "--format=%s", sha],
                           capture_output=True, text=True, cwd=LAB)
        _subjects[sha] = r.stdout.strip() if r.returncode == 0 else ""
    return _subjects[sha]


def load_runs():
    runs = []
    for p in sorted(RUNS.glob("*/run.json"), reverse=True):
        r = json.loads(p.read_text())
        job_file = p.parent / "job.json"
        if job_file.exists():
            job = json.loads(job_file.read_text())
            for an, arm in r.get("arms", {}).items():
                arm.setdefault("server_command",
                               job.get("arms", {}).get(an, {}).get("server_command"))
        if not r.get("git", {}).get("subject"):
            r.setdefault("git", {})["subject"] = commit_subject(
                r.get("git", {}).get("sha", ""))
        exp_file = LAB / "experiments" / f"{r['experiment']}.yaml"
        r["_primary"] = (yaml.safe_load(exp_file.read_text()).get("primary_metric")
                         if exp_file.exists() else None)
        r["_artifacts"] = sorted(
            str(f.relative_to(CACHE)) for f in (CACHE / r["id"]).rglob("*")
            if f.is_file()) if (CACHE / r["id"]).is_dir() else []
        runs.append(r)
    return runs


def fork_url():
    return "https://github.com/devpramod/vllm-cpu"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "runs": load_runs(), "fork": fork_url()})


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str):
    run = next((r for r in load_runs() if r["id"] == run_id), None)
    return templates.TemplateResponse(request, "run.html", {
        "run": run, "run_id": run_id, "fork": fork_url()})


@app.get("/compare", response_class=HTMLResponse)
def compare(request: Request, a: str = "", b: str = ""):
    runs = load_runs()
    ra = next((r for r in runs if r["id"] == a), None)
    rb = next((r for r in runs if r["id"] == b), None)
    rows = []
    if ra and rb:
        metrics = sorted({m for r in (ra, rb) for arm in r.get("arms", {}).values()
                          for m in arm.get("metrics", {})})
        for m in metrics:
            for arm in sorted(set(ra["arms"]) | set(rb["arms"])):
                va = ra["arms"].get(arm, {}).get("metrics", {}).get(m, {}).get("median")
                vb = rb["arms"].get(arm, {}).get("metrics", {}).get(m, {}).get("median")
                delta = (f"{(vb - va) / va * 100:+.1f}%"
                         if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va
                         else "")
                rows.append({"metric": m, "arm": arm, "a": va, "b": vb, "delta": delta})
    return templates.TemplateResponse(request, "compare.html", {
        "runs": runs, "a": ra, "b": rb, "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
