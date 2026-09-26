#!/usr/bin/env python
"""Build results/report.html: a self-contained HTML report (plots inlined)."""

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
CONCS = [1, 2, 4, 8, 16, 32, 64, 128]
SLOS = [20, 10]  # tok/s/user thresholds for capacity planning

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


# ----------------------------------------------------------------- loading

def load(path: Path):
    return json.load(open(path)) if path.exists() else None


def bench_m(b):
    if b is None or not b.get("completed"):
        return None
    return dict(thr=b["output_throughput"], pu=1000 / b["mean_tpot_ms"],
                ttft=b["mean_ttft_ms"], ttft99=b["p99_ttft_ms"],
                n=b["completed"], al=b.get("spec_decode_acceptance_length"))


def mt_m(d):
    if not d:
        return None
    s = min(r["start_time_ms"] for r in d)
    e = max(r["start_time_ms"] + r["latency_ms"] for r in d)
    tp = [r["tpot_ms"] for r in d if r["tpot_ms"] > 0]
    tt = sorted(r["ttft_ms"] for r in d)
    return dict(thr=sum(r["output_num_tokens"] for r in d) / ((e - s) / 1e3),
                pu=1000 / np.mean(tp), ttft=float(np.mean(tt)),
                ttft99=tt[min(len(tt) - 1, int(0.99 * len(tt)))], n=len(d),
                al=None)


def tp4(w, c):
    d = R / ("p5/TP4" if c <= 32 else "p5_mns128/TP4")
    return bench_m(load(d / f"{w}_c{c}.json"))


def spec(cfg, w, c):
    d = R / (f"p6/{cfg}" if c <= 32 else f"p6_mns128/{cfg}")
    return bench_m(load(d / f"{w}_c{c}.json"))


def replica(layout, w, c, policy="cache_aware"):
    if layout == "TP4x1":
        policy = "round_robin"
    d = R / ("p7" if c <= 32 else "p7_mns128") / layout / policy
    f = d / f"{w}_c{c}.json"
    if w == "multiturn":
        return mt_m(load(f))
    return bench_m(load(f))


def p1(cfg, w, c):
    return bench_m(load(R / f"p1/{cfg}/{w}_c{c}.json"))


# ----------------------------------------------------------------- plotting

FIG_DIR = R / "report_figs"
_fig_count = 0


def fig_html(fig, caption=""):
    global _fig_count
    _fig_count += 1
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"fig{_fig_count:02d}.png", bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}"/>{cap}</figure>'


PALETTE = plt.get_cmap("tab10").colors


def grouped_bars(ax, groups, series, getter, ylabel, fmt="{:.0f}",
                 annotate=True, colors=None):
    """groups: x categories; series: list of (key, label)."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (key, label) in enumerate(series):
        vals = [getter(key, g) for g in groups]
        ys = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, ys, width, label=label,
                      color=(colors or PALETTE)[i % 10])
        if annotate:
            for b, v in zip(bars, vals):
                if v is not None:
                    ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2,
                                b.get_height()), ha="center", va="bottom",
                                fontsize=6, rotation=90, xytext=(0, 2),
                                textcoords="offset points")
    ax.set_xticks(x, [str(g) for g in groups])
    ax.set_ylabel(ylabel)


def table(headers, rows, cls=""):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                  for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def f0(v):
    return "–" if v is None else f"{v:,.0f}"


def f1(v):
    return "–" if v is None else f"{v:,.1f}"


def capacity(getter, slo):
    """Largest measured concurrency whose per-user speed meets `slo`."""
    best = None
    for c in CONCS:
        m = getter(c)
        if m and m["pu"] >= slo:
            best = (c, m)
    return best


# ----------------------------------------------------------------- sections

RAND = [("rand_128_128", "128/128"), ("rand_128_1k", "128/1024"),
        ("rand_1k_128", "1024/128"), ("rand_2k_128", "2048/128")]
SGPT = [("sgpt_128_128", "128/128"), ("sgpt_128_1k", "128/1024"),
        ("sgpt_1k_128", "1024/128"), ("sgpt_2k_128", "2048/128")]
PFX = [("rand_1k_128", "1024/128 no shared prefix"),
       ("rand_pfx512_1k_128", "1024/128, 512 shared"),
       ("rand_2k_128", "2048/128 no shared prefix"),
       ("rand_pfx512_2k_128", "2048/128, 512 shared")]
LAYOUTS = [("TP4x1", "TP4 ×1"), ("TP2x3", "TP2 ×3"), ("TP1x6", "TP1 ×6"),
           ("TP2x2", "TP2 ×2 (4 nodes)"), ("TP2x2_TP1x2", "TP2 ×2 + TP1 ×2"),
           ("TP4_TP1x2", "TP4 + TP1 ×2"), ("TP4_TP2", "TP4 + TP2")]
LAYOUT_WL = [("rand_128_1k", "Random 128 in / 1024 out (decode-heavy)"),
             ("rand_1k_128", "Random 1024 in / 128 out (prefill-heavy)"),
             ("multiturn", "Multi-turn ShareGPT chat")]
SPEC_WL = [("gsm8k", "GSM8K"), ("humaneval", "HumanEval"),
           ("mtbench", "MT-Bench")]


def section_tp4():
    out = []
    for title, wls in [("Random prompts", RAND), ("ShareGPT prompts", SGPT)]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
        grouped_bars(axes[0], CONCS, wls,
                     lambda w, c: (tp4(w, c) or {}).get("pu"),
                     "tok/s per user")
        axes[0].set_xlabel("concurrent users")
        axes[0].set_title(f"{title}: per-user decode speed")
        axes[0].legend(title="in/out tokens", fontsize=7)
        for i, (w, lab) in enumerate(wls):
            xs = [c for c in CONCS if tp4(w, c)]
            axes[1].plot(xs, [tp4(w, c)["thr"] for c in xs], "o-", label=lab,
                         color=PALETTE[i])
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(CONCS, [str(c) for c in CONCS])
        axes[1].set_xlabel("concurrent users")
        axes[1].set_ylabel("output tok/s (all users)")
        axes[1].set_title(f"{title}: total throughput")
        out.append(fig_html(fig))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for i, (w, lab) in enumerate(RAND):
        xs = [c for c in CONCS if tp4(w, c)]
        axes[0].plot(xs, [tp4(w, c)["ttft"] / 1e3 for c in xs], "o-",
                     label=lab, color=PALETTE[i])
        axes[1].plot([tp4(w, c)["thr"] for c in xs],
                     [tp4(w, c)["pu"] for c in xs], "o-", label=lab,
                     color=PALETTE[i])
        for c in xs:
            axes[1].annotate(str(c), (tp4(w, c)["thr"], tp4(w, c)["pu"]),
                             fontsize=6, xytext=(3, 3),
                             textcoords="offset points")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(CONCS, [str(c) for c in CONCS])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("concurrent users")
    axes[0].set_ylabel("mean TTFT (s, log)")
    axes[0].set_title("Time to first token")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("output tok/s (all users)")
    axes[1].set_ylabel("tok/s per user")
    axes[1].set_title("Throughput vs per-user speed (labels = users)")
    for s in SLOS:
        axes[1].axhline(s, ls="--", c="grey", lw=0.8)
    out.append(fig_html(fig, "Random prompts, TP4. Dashed lines: 20 and 10 "
                        "tok/s/user service levels."))

    fig, ax = plt.subplots(figsize=(13, 3.4))
    grouped_bars(ax, CONCS, PFX, lambda w, c: (tp4(w, c) or {}).get("thr"),
                 "output tok/s (all users)",
                 colors=[PALETTE[2], "#9edc9e", PALETTE[3], "#f2a3a3"])
    ax.set_xlabel("concurrent users")
    ax.set_title("Prefix caching: effect of a 512-token shared prefix "
                 "(same total prompt length)")
    ax.legend(fontsize=7)
    out.append(fig_html(fig))

    rows = []
    for w, lab in RAND + SGPT:
        kind = "Random" if w.startswith("rand") else "ShareGPT"
        cells = [f"{kind} {lab}"]
        for c in [1, 8, 32, 64, 128]:
            m = tp4(w, c)
            cells.append("–" if not m else
                         f"{m['pu']:.0f} / {m['thr']:,.0f} / {m['ttft'] / 1e3:.2f}s")
        rows.append(cells)
    out.append(table(["Workload (in/out)"] + [f"{c} users" for c in
                     [1, 8, 32, 64, 128]], rows))
    out.append('<p class="note">Cells: tok/s per user / total output tok/s '
               '/ mean TTFT.</p>')
    return "\n".join(out)


def section_parallel():
    cfgs = [("TP4", "TP4"), ("TP2", "TP2"), ("TP2_PP3", "TP2×PP3"),
            ("PP2", "PP2"), ("PP3", "PP3"), ("PP6", "PP6")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    grouped_bars(axes[0], [1, 8, 32], cfgs,
                 lambda k, c: (p1(k, "rand_1k_1k", c) or {}).get("pu"),
                 "tok/s per user")
    axes[0].set_title("Per-user speed, 1024/1024")
    axes[0].set_xlabel("concurrent users")
    axes[0].legend(fontsize=7, ncol=2)
    grouped_bars(axes[1], [1, 8, 32], cfgs,
                 lambda k, c: (p1(k, "rand_1k_1k", c) or {}).get("thr"),
                 "output tok/s (all users)")
    axes[1].set_title("Total throughput, 1024/1024")
    axes[1].set_xlabel("concurrent users")
    return fig_html(fig, "Single instance, prefix caching off (phase 1). "
                    "EP and DP configurations failed to start (see "
                    "Issues).")


def section_spec():
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (w, lab) in zip(axes, SPEC_WL):
        concs = [c for c in CONCS if spec("TP4", w, c)]
        grouped_bars(ax, concs, [("TP4", "Baseline"),
                                 ("TP4_DFLASH_HYB_K7", "DFlash k=7")],
                     lambda k, c: (spec(k, w, c) or {}).get("pu"),
                     "tok/s per user", colors=["#8c8c8c", PALETTE[0]])
        for i, c in enumerate(concs):
            b, d = spec("TP4", w, c), spec("TP4_DFLASH_HYB_K7", w, c)
            if b and d:
                ax.annotate(f"×{d['pu'] / b['pu']:.2f}", (i, max(b["pu"],
                            d["pu"]) + 12), ha="center", fontsize=7,
                            color="darkred")
        ax.set_title(lab)
        ax.set_xlabel("concurrent users")
    axes[0].legend(fontsize=7)
    out.append(fig_html(fig, "Per-user speed with and without DFlash "
                        "(TP4). Red: per-user speedup."))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    for i, (w, lab) in enumerate(SPEC_WL):
        concs = [c for c in CONCS if spec("TP4", w, c)
                 and spec("TP4_DFLASH_HYB_K7", w, c)]
        axes[0].plot(concs, [spec("TP4_DFLASH_HYB_K7", w, c)["thr"] /
                             spec("TP4", w, c)["thr"] for c in concs],
                     "o-", label=lab, color=PALETTE[i])
        axes[1].plot(concs, [spec("TP4_DFLASH_HYB_K7", w, c)["al"]
                             for c in concs], "o-", label=lab,
                     color=PALETTE[i])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONCS, [str(c) for c in CONCS])
        ax.set_xlabel("concurrent users")
    axes[0].axhline(1, c="k", lw=0.8)
    axes[0].set_ylabel("throughput speedup (×)")
    axes[0].set_title("DFlash total-throughput speedup")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("mean acceptance length")
    axes[1].set_title("DFlash acceptance length (tokens per verify step)")
    out.append(fig_html(fig))

    rows = []
    for w, lab in SPEC_WL:
        for c in CONCS:
            b, d = spec("TP4", w, c), spec("TP4_DFLASH_HYB_K7", w, c)
            if not (b and d):
                continue
            rows.append([lab, c, b["n"], f"{b['pu']:.0f}", f"{d['pu']:.0f}",
                         f"×{d['pu'] / b['pu']:.2f}", f0(b["thr"]),
                         f0(d["thr"]), f"×{d['thr'] / b['thr']:.2f}",
                         f"{d['al']:.2f}"])
    out.append(table(["Dataset", "Users", "Requests", "Base tok/s/user",
                      "DFlash tok/s/user", "Per-user ×", "Base tok/s",
                      "DFlash tok/s", "Throughput ×", "Accept. len"], rows))
    acc = [(cfg, load(R / f"p6/{cfg}/gsm8k_accuracy.json"))
           for cfg in ["TP4", "TP4_DFLASH_HYB_K7"]]
    out.append("<p>GSM8K accuracy (250 questions, greedy): " + ", ".join(
        f"{'DFlash' if 'DFLASH' in c else 'baseline'} "
        f"{a['accuracy'] * 100:.1f}%" for c, a in acc if a) + ".</p>")
    return "\n".join(out)


def section_replicas():
    out = []
    for w, lab in LAYOUT_WL:
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
        grouped_bars(axes[0], [c for c in CONCS], LAYOUTS,
                     lambda L, c: (replica(L, w, c) or {}).get("pu"),
                     "tok/s per user", annotate=False)
        axes[0].set_title(f"{lab}: per-user speed")
        axes[0].set_xlabel("concurrent users (total, at the router)")
        axes[0].legend(fontsize=6, ncol=2)
        for s in SLOS:
            axes[0].axhline(s, ls="--", c="grey", lw=0.8)
        for i, (L, llab) in enumerate(LAYOUTS):
            pts = [(replica(L, w, c), c) for c in CONCS]
            pts = [(m, c) for m, c in pts if m]
            axes[1].plot([m["thr"] for m, _ in pts], [m["pu"] for m, _ in pts],
                         "o-", label=llab, color=PALETTE[i], ms=3)
            for m, c in pts:
                if c >= 32:
                    axes[1].annotate(str(c), (m["thr"], m["pu"]), fontsize=6,
                                     xytext=(2, 2), textcoords="offset points")
        axes[1].set_xlabel("output tok/s (all users)")
        axes[1].set_ylabel("tok/s per user")
        axes[1].set_title("Throughput vs per-user speed (labels ≥32 users)")
        for s in SLOS:
            axes[1].axhline(s, ls="--", c="grey", lw=0.8)
        out.append(fig_html(fig))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=False)
    for ax, (w, lab) in zip(axes, LAYOUT_WL):
        grouped_bars(ax, [32, 64, 128], LAYOUTS,
                     lambda L, c: (replica(L, w, c) or {}).get("thr"),
                     "output tok/s (all users)", annotate=False)
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("concurrent users")
    axes[0].legend(fontsize=6, ncol=2)
    out.append(fig_html(fig, "Total throughput at high load. At 64 and 128 "
                        "users every instance runs with "
                        "--max-num-seqs 128."))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    for ax, key, ylab in [(axes[0], "thr", "output tok/s (all users)"),
                          (axes[1], "ttft", "mean TTFT (ms)")]:
        groups = [L for L, _ in LAYOUTS if L != "TP4x1"]
        x = np.arange(len(groups))
        for j, c in enumerate([8, 32, 128]):
            for k, pol in enumerate(["round_robin", "cache_aware"]):
                vals = [(replica(L, "multiturn", c, pol) or {}).get(key, 0)
                        for L in groups]
                ax.bar(x + (j * 2 + k - 2.5) * 0.13, vals, 0.13,
                       color=PALETTE[j], alpha=0.45 if k == 0 else 1.0,
                       label=f"{c} users, {pol.replace('_', '-')}")
        ax.set_xticks(x, [dict(LAYOUTS)[g] for g in groups], fontsize=7,
                      rotation=15)
        ax.set_ylabel(ylab)
    axes[0].set_title("Multi-turn: round-robin (light) vs cache-aware (dark)")
    axes[1].set_title("Multi-turn TTFT by routing policy")
    axes[0].legend(fontsize=6, ncol=3)
    out.append(fig_html(fig))
    return "\n".join(out)


def section_capacity():
    rows = []
    cands = [("TP4 ×1", lambda w: (lambda c: replica("TP4x1", w, c)))]
    cands += [(lab, (lambda L: lambda w: (lambda c: replica(L, w, c)))(L))
              for L, lab in LAYOUTS if L != "TP4x1"]
    for w, wlab in LAYOUT_WL:
        for lab, g in cands:
            cells = [wlab if lab == "TP4 ×1" else "", lab]
            for s in SLOS:
                cap = capacity(g(w), s)
                cells.append("–" if not cap else
                             f"{cap[0]} users, {cap[1]['thr']:,.0f} tok/s, "
                             f"TTFT {cap[1]['ttft'] / 1e3:.2f}s")
            rows.append(cells)
    t = table(["Workload", "Deployment"] +
              [f"Max users at ≥{s} tok/s/user" for s in SLOS], rows, "cap")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, (w, wlab) in zip(axes, LAYOUT_WL):
        labs = [lab for lab, _ in cands]
        x = np.arange(len(labs))
        for k, s in enumerate(SLOS):
            vals = []
            for _, g in cands:
                cap = capacity(g(w), s)
                vals.append(cap[1]["thr"] if cap else 0)
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38,
                   label=f"≥{s} tok/s/user", color=PALETTE[k + 4])
        ax.set_xticks(x, labs, rotation=30, fontsize=7, ha="right")
        ax.set_title(wlab, fontsize=8)
        ax.set_ylabel("tok/s delivered within SLO")
    axes[0].legend(fontsize=7)
    return fig_html(fig, "Highest total throughput each deployment "
                    "sustains while every user still gets at least the "
                    "given decode speed (measured points only).") + t


def section_profile():
    rows = [["Baseline step (1 token)", "11.6", "Forward 9.2, logits 1.3"],
            ["DFlash step (8 tokens verified)", "29.3",
             "Verify 18.5, drafter 7.4 (logits 1.2, argmax 1.1), "
             "sampling 1.1"]]
    t = table(["TP4, 1 user", "ms/step", "Breakdown (ms)"], rows)
    t2 = table(["Operation (TP1)", "Achieved", "Verdict"], [
        ["MoE verify, 8 tokens (≈21 of 32 experts)", "≈224 GB/s (90% of "
         "249 GB/s node peak)", "Memory-bound; no kernel headroom"],
        ["lm_head, 201k vocab (target and drafter)", "≈247 GB/s (99%)",
         "Only int8 weights or a smaller draft vocab help"],
        ["Drafter dense bf16 layers", "208–234 GB/s", "Quantize to gain"],
        ["argmax over 201k vocab", "300–570 µs/call", "Single-threaded; "
         "a chunked version is 7–15× faster"],
        ["TP all-reduce", "1.35 ms/step base, 2.3 ms DFlash (TP4)",
         "Latency-bound"]])
    return t + t2


def build():
    css = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;
margin:24px auto;padding:0 18px;color:#222;line-height:1.45}
h1{font-size:26px;margin-bottom:4px}h2{border-bottom:2px solid #e5e5e5;
padding-bottom:4px;margin-top:38px}h3{margin-top:22px}
figure{margin:14px 0}figure img{max-width:100%}figcaption{font-size:12px;
color:#666}table{border-collapse:collapse;font-size:12px;margin:10px 0;
width:100%}th,td{border:1px solid #ddd;padding:4px 7px;text-align:right}
th{background:#f4f6f8}td:first-child,th:first-child{text-align:left}
table.cap td:nth-child(2){text-align:left}.note{font-size:12px;color:#666}
.box{background:#f6f9fc;border-left:4px solid #3b7dd8;padding:10px 16px;
margin:14px 0}.warn{border-left-color:#d88b3b;background:#fdf8f2}
ul{margin:6px 0}li{margin:3px 0}code{background:#f2f2f2;padding:0 3px}
"""
    body = f"""
<h1>gpt-oss-20b on CPU with vLLM: serving benchmark report</h1>
<p class="note">2× Xeon 6972P (192 cores, SNC3 → 6 NUMA nodes of 32 cores,
1.5 TB), vLLM 0.30.0 CPU wheel, bf16 activations with MXFP4 MoE weights.
Generated from <code>results/</code> by <code>report.py</code>.</p>

<h2>Key findings and deployment guidance</h2>
{FINDINGS}

<h2>1. Method</h2>
<ul>
<li><b>tok/s per user</b> = 1000 / mean time per output token (TPOT), the
decode speed each user sees after the first token. TTFT is reported
separately. Total throughput = output tokens / wall time.</li>
<li>Concurrency = number of simultaneous users (closed loop). 4 prompts per
user up to 32 users, 2 per user at 64 and 128 (at least 8).</li>
<li>Prefix caching on; the cache is reset before every point, so points never
reuse each other's prompts. Random and ShareGPT runs force exact output
lengths (<code>--ignore-eos</code>); ShareGPT prompts are filtered to ±20%
of the target input length.</li>
<li>Server: <code>--max-model-len 4096 --max-num-batched-tokens 4096</code>,
<code>--max-num-seqs 32</code> up to 32 users and <code>128</code> above.
KV cache 40 GB per instance, OpenMP threads bound per NUMA node.</li>
<li>Replicas: one <code>vllm serve</code> per replica on disjoint NUMA nodes,
behind <code>vllm-router</code> (cache-aware policy unless stated).</li>
<li>Speculative decoding: DFlash drafter <code>z-lab/gpt-oss-20b-DFlash</code>,
7 draft tokens, chat endpoint, greedy decoding, real output lengths.</li>
</ul>

<h2>2. Single instance (TP4): per-user speed vs concurrency</h2>
{section_tp4()}

<h3>Parallelism choice for a single instance</h3>
{section_parallel()}

<h2>3. Speculative decoding (DFlash)</h2>
{section_spec()}

<h2>4. Multiple replicas behind vllm-router</h2>
{section_replicas()}

<h2>5. Capacity at a per-user speed target</h2>
{section_capacity()}

<h2>6. Why DFlash gains are modest on CPU (profiling)</h2>
{section_profile()}

<h2>7. Issues found</h2>
{ISSUES}
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>gpt-oss-20b CPU serving report</title><style>{css}</style></head><body>{body}</body></html>"


FINDINGS = """<div class="box"><p>Findings are filled in once all runs
complete.</p></div>"""

ISSUES = """<ul>
<li>Expert parallelism crashes on CPU: <code>CPUExpertsMxfp4</code> ignores
<code>expert_map</code> and passes global expert IDs to a kernel holding only
local experts (out-of-bounds write in <code>moe_align_block_size</code>).</li>
<li>Data parallelism on CPU fails: <code>all_gatherv</code> is not
implemented; with 6 EP ranks the MXFP4 loader also fails because 32 experts
don't divide by 6.</li>
<li>DFlash on gpt-oss needs the CPU Triton fallbacks from PR #58217 and a
KV-cache group-size override (the drafter's layers are split across groups
otherwise). <code>--disable-hybrid-kv-cache-manager</code> corrupts gpt-oss
output on CPU (GSM8K 61.6%).</li>
<li><code>benchmarks/multi_turn</code>: inverted English-only filter in the
ShareGPT converter, crash on conversations ending with a user turn, hang at
exit, early stop by default, and TTFT/token counting that ignores reasoning
output.</li>
</ul>"""


if __name__ == "__main__":
    out = R / "report.html"
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
