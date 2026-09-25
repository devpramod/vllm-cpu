#!/usr/bin/env python
"""Collect sweep result JSONs into a CSV and per-workload Pareto plots.

Usage: summarize.py results/p1 [results/p2 ...]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIELDS = [
    "completed", "failed", "duration", "output_throughput",
    "total_token_throughput", "request_throughput",
    "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p50_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms", "mean_e2el_ms",
]


def load(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(d.glob("*/*_c*.json")):
            r = json.loads(f.read_text())
            wl, conc = f.stem.rsplit("_c", 1)
            row = {"phase": d.name, "config": f.parent.name, "workload": wl,
                   "concurrency": int(conc)}
            row.update({k: r.get(k) for k in FIELDS})
            if r.get("mean_tpot_ms"):
                row["tok_s_per_user"] = 1000.0 / r["mean_tpot_ms"]
            for k in ("spec_decode_acceptance_length",
                      "spec_decode_acceptance_rate"):
                if k in r:
                    row[k] = r[k]
            rows.append(row)
    return pd.DataFrame(rows)


def pareto_plot(df: pd.DataFrame, out: Path) -> None:
    for (phase, wl), g in df.groupby(["phase", "workload"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        for cfg, gc in g.groupby("config"):
            gc = gc.sort_values("concurrency")
            ax.plot(gc["tok_s_per_user"], gc["output_throughput"], "o-",
                    label=cfg)
            for _, r in gc.iterrows():
                ax.annotate(str(r["concurrency"]),
                            (r["tok_s_per_user"], r["output_throughput"]),
                            fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_xlabel("per-user output tok/s (1000 / mean TPOT)")
        ax.set_ylabel("aggregate output tok/s")
        ax.set_title(f"gpt-oss-20b CPU, {phase}, {wl} (labels = concurrency)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"pareto_{phase}_{wl}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    df = load(dirs)
    if df.empty:
        sys.exit("no results found")
    out = dirs[0]
    df = df.sort_values(["workload", "concurrency", "output_throughput"],
                        ascending=[True, True, False])
    df.to_csv(out / "summary.csv", index=False)
    pareto_plot(df, out)

    cols = ["config", "output_throughput", "tok_s_per_user", "mean_ttft_ms",
            "p99_ttft_ms", "failed"]
    pd.set_option("display.width", 200)
    for (wl, c), g in df.groupby(["workload", "concurrency"]):
        print(f"\n== {wl}  concurrency={c}")
        print(g[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
