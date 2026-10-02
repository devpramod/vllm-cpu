#!/usr/bin/env python
"""Build results/ocr/report.html for LightOnOCR-2-1B (plots inlined)."""

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
import report as G  # noqa: E402
from report import f1, fig_html, grouped_bars, plt, table  # noqa: E402

R = BENCH / "results" / "ocr"
G.FIG_DIR = R / "report_figs"
CONCS = [1, 2, 4, 8, 16, 32, 64, 128]
RCONCS = [1, 4, 16, 32, 64, 128, 256]
SYN_CONCS = [1, 4, 16, 64, 128]
SINGLE = [("TP1", "TP1 (1 node)"), ("TP2", "TP2 (2 nodes)"), ("TP4", "TP4 (4 nodes)")]
SETS = [("omnidoc", "OmniDocBench (full pages)"), ("olmocr", "olmOCR-bench (PDF pages)"),
        ("cord", "CORD-v2 (receipts)")]
NODES = {"TP1": 1, "TP2": 2, "TP2_DATA": 2, "TP4": 4, "TP1x6": 6, "TP1x12": 6}


def m(path):
    d = G.load(path)
    if not d or not d.get("completed"):
        return None
    c = d["concurrency"]
    return dict(
        pps=d["pages_per_s"], sus=c / d["latency_mean_s"], lat=d["latency_mean_s"],
        p90=d["latency_p90_s"], ttft=d["ttft_mean_s"], pu=d["tok_s_user"],
        thr=d["output_tok_s"], out=d["out_tokens_mean"], inp=d["prompt_tokens_mean"],
        cap=d["finish_length"], n=d["completed"])


def single(cfg, wl, c):
    return m(R / cfg / f"{wl}_c{c}.json")


def rep(layout, policy, wl, c):
    return m(R / "replicas" / layout / policy / f"{wl}_c{c}.json")


def best_rep(layout, wl, c):
    a, b = rep(layout, "round_robin", wl, c), rep(layout, "power_of_two", wl, c)
    return a if b is None or (a and a["sus"] >= b["sus"]) else b


def deploy(key, wl, c):
    """Unified getter over single instances and replica layouts (round robin)."""
    if key.startswith("TP1x"):
        return rep(key, "round_robin", wl, c)
    return single(key, wl, c)


def bars(getter, groups, series, metric, ylabel, title, fmt="{:.2f}", log=False):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    grouped_bars(ax, groups, series,
                 lambda k, g: (v := getter(k, g)) and v[metric], ylabel, fmt=fmt)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("concurrent users")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(series))
    return fig


# ----------------------------------------------------------------- sections

def section_single():
    out = []
    for wl, label in SETS:
        g = lambda k, c, wl=wl: single(k, wl, c)  # noqa: E731
        out.append(f"<h3>{label}</h3>")
        out.append(fig_html(bars(g, CONCS, SINGLE, "sus", "pages / s",
                                 f"{label}: sustained pages/s"),
                            "Sustained pages/s = users ÷ mean latency per page."))
        out.append(fig_html(bars(g, CONCS, SINGLE, "lat", "seconds",
                                 f"{label}: mean latency per page", fmt="{:.0f}",
                                 log=True)))
        out.append(fig_html(bars(g, CONCS, SINGLE, "pu", "tok/s per user",
                                 f"{label}: decode speed per user", fmt="{:.0f}")))
        rows = []
        for k, name in SINGLE + [("TP2_DATA", "TP2, encoder data-parallel")]:
            for c in CONCS:
                v = single(k, wl, c)
                if v:
                    rows.append([name, c, f"{v['sus']:.3f}", f"{v['pps']:.3f}",
                                 f1(v["lat"]), f1(v["p90"]), f1(v["ttft"]),
                                 f1(v["pu"]), f"{v['thr']:.0f}", f"{v['out']:.0f}",
                                 v["cap"]])
        out.append("<details><summary>Table</summary>" + table(
            ["config", "users", "sustained pages/s", "measured pages/s",
             "latency s", "p90 latency s", "TTFT s", "tok/s/user", "output tok/s",
             "mean output tokens", "pages hitting 4096 cap"], rows) + "</details>")
    return "\n".join(out)


def section_encoder():
    rows = []
    for wl, label in SETS:
        for c in [1, 8, 32, 128]:
            a, b = single("TP2", wl, c), single("TP2_DATA", wl, c)
            if a and b:
                rows.append([label, c, f"{a['sus']:.3f}", f"{b['sus']:.3f}",
                             f1(a["ttft"]), f1(b["ttft"]),
                             f"{100 * (b['sus'] / a['sus'] - 1):+.0f}%"])
    return ("<p>Running the Pixtral vision encoder data-parallel across the two TP "
            "ranks (<code>--mm-encoder-tp-mode data</code>) instead of "
            "tensor-parallel makes no measurable difference: the encoder is a small "
            "part of each page's cost, and TTFT is unchanged.</p>"
            + table(["dataset", "users", "TP2 pages/s", "TP2 data-parallel encoder "
                     "pages/s", "TP2 TTFT s", "data-parallel TTFT s", "difference"],
                    rows))


def section_sampling():
    groups = [(k, c) for k, _ in SINGLE for c in (1, 16, 128)]
    labels = [f"{k}\n{c} u" for k, c in groups]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
    for ax, metric, yl in [(axes[0], "sus", "pages / s"),
                           (axes[1], "pu", "tok/s per user")]:
        grouped_bars(ax, list(range(len(groups))),
                     [("omnidoc", "card: temperature 0.2, top_p 0.9"),
                      ("omnidoc_greedy", "greedy")],
                     lambda wl, i: (v := single(groups[i][0], wl, groups[i][1]))
                     and v[metric], yl, fmt="{:.2f}" if metric == "sus" else "{:.0f}")
        ax.set_xticks(range(len(groups)), labels, fontsize=7)
        ax.legend(fontsize=8)
    axes[0].set_title("OmniDocBench pages/s")
    axes[1].set_title("OmniDocBench decode speed per user")
    acc = json.load(open(R / "accuracy_omnidoc.json"))
    arow = []
    for k, _ in SINGLE:
        for wl, name in [("omnidoc", "card"), ("omnidoc_greedy", "greedy")]:
            s = acc.get(f"results/ocr/{k}/{wl}_c128.rec.jsonl")
            if s:
                arow.append([k, name, s["pages"], f"{s['ned_mean']:.3f}",
                             f"{s['ned_median']:.3f}"])
    s = acc.get("results/ocr/replicas/TP1x6/round_robin/omnidoc_c256.rec.jsonl")
    if s:
        arow.append(["TP1 ×6 replicas", "card", s["pages"], f"{s['ned_mean']:.3f}",
                     f"{s['ned_median']:.3f}"])
    return (fig_html(fig, "Same server, same pages; only the request's sampling "
                     "parameters differ.")
            + "<p>The model card recommends temperature 0.2 with top_p 0.9. On CPU the "
            "top-p filter sorts the 151k-token vocabulary every step and adds "
            "about 8.5 ms per token (12.4 → 20.9 ms at one user on TP1; temperature "
            "alone costs nothing). Greedy decoding is 1.8–2.7× faster per page at one "
            "user and 1.1–1.4× at 128 users, with the same accuracy:</p>"
            + table(["config", "sampling", "pages scored", "mean edit distance",
                     "median edit distance"], arow)
            + "<p class='note'>Edit distance: page-level normalized Levenshtein "
            "distance on OmniDocBench text blocks (lower is better). Tables, display "
            "equations, figures, headers, footers and page numbers are removed from "
            "both reference and output. This approximates, but is not, the official "
            "OmniDocBench text metric. 196 of the 200 sampled English pages have "
            "reference text.</p>")


def section_synthetic():
    sizes = [("syn768_o512", "768×768"), ("syn1024_o512", "1024×1024"),
             ("syn1540_o512", "1190×1540")]
    outs = [("syn1024_o128", "128"), ("syn1024_o512", "512"),
            ("syn1024_o2048", "2048")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, metric, yl, t in [(axes[0], "ttft", "seconds", "TTFT at 16 users"),
                              (axes[1], "sus", "pages / s", "pages/s at 16 users"),
                              (axes[2], "pu", "tok/s per user",
                               "decode speed at 16 users")]:
        grouped_bars(ax, [s for s, _ in sizes], SINGLE,
                     lambda k, wl: (v := single(k, wl, 16)) and v[metric], yl,
                     fmt="{:.1f}")
        ax.set_xticks(range(3), [lbl for _, lbl in sizes])
        ax.set_title(t)
    axes[0].legend(fontsize=7)
    rows = []
    for wl, lbl in sizes:
        v = single("TP1", wl, 1)
        rows.append([lbl, f"{v['inp']:.0f}", f1(v["ttft"]),
                     f1(single("TP4", wl, 1)["ttft"])])
    fig2, ax2 = plt.subplots(figsize=(10, 3.2))
    grouped_bars(ax2, SYN_CONCS, [(wl, f"{lbl} output tokens") for wl, lbl in outs],
                 lambda wl, c: (v := single("TP4", wl, c)) and v["sus"], "pages / s",
                 fmt="{:.2f}")
    ax2.set_xlabel("concurrent users")
    ax2.set_title("TP4, 1024×1024 noise image: pages/s by output length")
    ax2.legend(fontsize=8)
    return ("<p>Random-noise images with a fixed output length "
            "(<code>ignore_eos</code>) separate the cost of the image from the cost "
            "of the text. Prompt tokens grow with image area, and TTFT with them; "
            "output length dominates the per-page cost of real documents.</p>"
            + fig_html(fig)
            + table(["image size", "prompt tokens", "TTFT TP1, 1 user s",
                     "TTFT TP4, 1 user s"], rows)
            + fig_html(fig2))


def section_mnbt():
    rows = []
    for wl, c in [("omnidoc", 8), ("omnidoc", 32), ("omnidoc", 128), ("cord", 32),
                  ("cord", 128), ("syn1024_o128", 16), ("syn1024_o128", 128)]:
        r = [wl, c]
        for k in ["TP1", "TP1_MNBT4K", "TP1_MNBT8K", "TP1_MNBT16K"]:
            v = single(k, wl, c)
            r.append(f"{v['sus']:.3f} / {v['ttft']:.1f}" if v else "–")
        rows.append(r)
    return ("<p>Raising <code>--max-num-batched-tokens</code> from the default 2048 "
            "(a full page is about 2,270 prompt tokens, so its prefill takes two "
            "steps) changes nothing beyond noise.</p>"
            + table(["workload", "users", "2048 (default)", "4096", "8192", "16384"],
                    rows, cls="")
            + "<p class='note'>Cells: sustained pages/s / mean TTFT in seconds, TP1.</p>")


def section_replicas():
    series = [("TP1", "TP1"), ("TP2", "TP2"), ("TP4", "TP4"),
              ("TP1x6", "TP1 ×6 replicas"), ("TP1x12", "TP1 ×12 replicas")]
    out = []
    for wl, label in [("omnidoc", "OmniDocBench"), ("cord", "CORD-v2 receipts")]:
        for metric, yl, t, fmt in [("sus", "pages / s", "sustained pages/s", "{:.2f}"),
                                   ("lat", "seconds", "mean latency per page",
                                    "{:.0f}"),
                                   ("pu", "tok/s per user", "decode speed per user",
                                    "{:.0f}")]:
            fig, ax = plt.subplots(figsize=(10, 3.4))
            for i, (k, lbl) in enumerate(series):
                xs = [c for c in RCONCS if deploy(k, wl, c)]
                ax.plot(xs, [deploy(k, wl, c)[metric] for c in xs], "o-", label=lbl,
                        color=G.PALETTE[i], lw=2 if "x" in k else 1.2)
            ax.set_xscale("log", base=2)
            ax.set_xticks(RCONCS, [str(c) for c in RCONCS])
            if metric == "lat":
                ax.set_yscale("log")
            ax.set_xlabel("concurrent users")
            ax.set_ylabel(yl)
            ax.set_title(f"{label}: {t} (whole machine vs single instances)")
            ax.legend(fontsize=8, ncol=5)
            out.append(fig_html(fig))
    rows = []
    for layout in ["TP1x6", "TP1x12"]:
        for wl in ["omnidoc", "cord"]:
            for c in RCONCS:
                a = rep(layout, "round_robin", wl, c)
                b = rep(layout, "power_of_two", wl, c)
                if a and b:
                    rows.append([layout, wl, c, f"{a['sus']:.3f}", f"{b['sus']:.3f}",
                                 f1(a["p90"]), f1(b["p90"]),
                                 f"{100 * (b['sus'] / a['sus'] - 1):+.0f}%"])
    out.append("<h3>Routing policy: round robin vs power of two choices</h3>"
               "<p>Power of two picks two replicas at random and sends the page to "
               "the one with fewer requests in flight. With page costs varying "
               "from ~170 to 4,096 output tokens it should help tail latency, but in "
               "practice it is within noise of round robin (and slightly worse on "
               "average), so round robin is shown in the charts above.</p>")
    out.append(table(["layout", "dataset", "users", "round robin pages/s",
                      "power of two pages/s", "round robin p90 s",
                      "power of two p90 s", "difference"], rows))
    return "\n".join(out)


def section_capacity():
    rows = []
    cands = [("TP1", "TP1 ×1 (1 node)"), ("TP2", "TP2 ×1 (2 nodes)"),
             ("TP4", "TP4 ×1 (4 nodes)"), ("TP1x6", "TP1 ×6 (6 nodes)"),
             ("TP1x12", "TP1 ×12 (6 nodes)")]
    for wl, slos in [("omnidoc", [60, 120]), ("cord", [15, 30])]:
        for slo in slos:
            for k, name in cands:
                concs = RCONCS if "x" in k else CONCS
                best = None
                for c in concs:
                    v = deploy(k, wl, c)
                    if v and v["p90"] <= slo and (best is None or v["sus"] > best[1]["sus"]):
                        best = (c, v)
                if best:
                    c, v = best
                    rows.append([wl, f"≤ {slo} s", name, c, f"{v['sus']:.2f}",
                                 f"{v['sus'] / NODES[k]:.3f}", f"{v['sus'] * 3600:,.0f}",
                                 f1(v["lat"]), f1(v["p90"])])
                else:
                    rows.append([wl, f"≤ {slo} s", name, "–", "–", "–", "–", "–",
                                 "–"])
    return ("<p>Best sustained throughput whose p90 latency per page stays within a "
            "target. Single instances are scaled to the whole machine in the "
            "pages/s-per-node column only.</p>"
            + table(["dataset", "p90 latency target", "deployment", "users",
                     "pages/s", "pages/s per node", "pages/hour", "mean latency s",
                     "p90 latency s"], rows))


def findings():
    r6 = lambda wl, c: deploy("TP1x6", wl, c)  # noqa: E731
    r12 = lambda wl, c: deploy("TP1x12", wl, c)  # noqa: E731
    t4 = lambda wl, c: single("TP4", wl, c)  # noqa: E731
    gain12 = [r12(w, c)["sus"] / r6(w, c)["sus"] - 1
              for w in ("omnidoc", "cord") for c in (64, 128, 256)]
    return f"""
<div class="box">
<ul>
<li><b>For throughput, run independent TP1 replicas, not one big instance.</b>
Six TP1 replicas (one per NUMA node) behind vllm-router sustain
{r6('omnidoc', 128)['sus']:.2f} pages/s on OmniDocBench at 128 users and
{r6('omnidoc', 256)['sus']:.2f} at 256, against {t4('omnidoc', 128)['sus']:.2f} for the
best single instance (TP4) at 128. At 128 users each page takes
{r6('omnidoc', 128)['lat']:.0f} s instead of {t4('omnidoc', 128)['lat']:.0f} s. On
receipts (CORD) the replicas sustain {r6('cord', 256)['sus']:.1f} pages/s against
{t4('cord', 128)['sus']:.1f}. A 1B model fits easily in one node, so tensor
parallelism only adds all-reduce cost.</li>
<li><b>Two replicas per node (TP1 ×12) gains little over one (TP1 ×6)</b>
({min(gain12):+.0%} to {max(gain12):+.0%} at 64–256 users). Decode is bound by each
node's memory bandwidth, so halving the cores per replica roughly halves its
speed.</li>
<li><b>For lowest latency on a single page, TP4.</b> At one user TP4 decodes at about
{t4('omnidoc', 1)['pu']:.0f} tok/s vs {single('TP1', 'omnidoc', 1)['pu']:.0f} for TP1,
and takes {t4('omnidoc', 1)['lat']:.0f} s per full page with card sampling
({t4('omnidoc_greedy', 1)['lat']:.1f} s greedy).</li>
<li><b>Use greedy decoding.</b> The model card's top_p 0.9 costs ~8.5 ms per token in
the CPU sampler. Greedy is up to 2.7× faster per page at low load and 1.1–1.4× at 128
users, with identical OmniDocBench edit distance (0.168 vs 0.168 on TP1).</li>
<li><b>Accuracy is the same in every configuration</b> (mean edit distance
0.163–0.168, median 0.058), which also validates the local Pixtral patch needed to
run the model on this vLLM build.</li>
<li><b>Knobs that do not matter here:</b> <code>--mm-encoder-tp-mode data</code>,
<code>--max-num-batched-tokens</code> 2048–16384. Power-of-two routing is no better
than round robin (5% lower on average, −30% to +20% per point, mostly noise).</li>
<li><b>Output length drives cost.</b> Full pages average ~1,250 output tokens and
~4% loop until the 4,096-token cap; those runaway pages dominate tail latency.
Lowering <code>max_tokens</code> or adding a repetition penalty is the next lever to
evaluate (it would need an accuracy check).</li>
</ul>
</div>
<h3>Recommended deployments</h3>
""" + table(
        ["goal", "deployment", "measured (card sampling unless noted)"],
        [["Bulk document throughput",
          "TP1 ×6 replicas + vllm-router (round robin), greedy",
          f"{r6('omnidoc', 256)['sus']:.2f} full pages/s at 256 users; "
          f"{r6('cord', 256)['sus']:.1f} receipts/s"],
         ["Interactive, one document at a time",
          "TP4 (or TP2 if only 2 nodes are free), greedy",
          f"{t4('omnidoc_greedy', 1)['lat']:.1f} s per full page (greedy, 1 user)"],
         ["Mixed", "TP1 ×6; per-user speed stays above 25 tok/s up to 32 users",
          f"{r6('omnidoc', 32)['sus']:.2f} pages/s at 32 users, "
          f"{r6('omnidoc', 32)['lat']:.0f} s mean per page"]])

ISSUES = """
<ul>
<li><b>Model fails to load on vLLM 0.30.0 with transformers 5.17</b>
(<code>ImportError: PixtralRotaryEmbedding</code>). Fixed locally by backporting the
<code>pixtral.py</code> change from upstream PR #56108
(<code>patches/pr56108_pixtral.diff</code>).</li>
<li><b>A request with <code>seed</code> crashes the CPU engine</b>:
<code>RuntimeError: CPU Generator does not use offset</code> in
<code>gpu_model_runner._bookkeeping_sync</code> (it rewinds the generator with
<code>set_offset</code>, which CPU generators lack). The harness sends no seed, so
sampled runs are not bit-reproducible.</li>
<li><b>top-p sampling is expensive on CPU</b> (~8.5 ms/token for a 151k vocabulary);
an optimization candidate for the CPU sampler.</li>
<li><b><code>vllm bench serve</code> cannot drive this model</b>: the ShareGPT/custom
loaders drop prompts with fewer than 4 text tokens (OCR prompts are image-only) and cap
prompt + output at 2,048 tokens. A small async harness (<code>ocr/bench_ocr.py</code>)
was used instead.</li>
<li><b>Measured pages/s is noisy at low page counts</b> because one runaway page can
set the run's wall time; charts use sustained pages/s = users ÷ mean latency, and the
tables show both.</li>
</ul>
"""


def build():
    css = G.CSS
    body = f"""
<h1>LightOnOCR-2-1B on CPU with vLLM: serving benchmark report</h1>
<p class="note">Model <code>lightonai/LightOnOCR-2-1B</code> (Pixtral vision encoder +
Qwen3 decoder, bf16). 2× Xeon 6972P (192 cores, SNC3 → 6 NUMA nodes of 32 cores,
1.5 TB), vLLM 0.30.0 CPU wheel. Generated by <code>ocr/report_ocr.py</code>.</p>

<h2>Key findings and deployment guidance</h2>
{findings()}

<h2>1. Method</h2>
<ul>
<li><b>Datasets</b>: OmniDocBench (200 English pages, stratified over 8 document
types), olmOCR-bench (200 PDF pages over 7 categories, rendered with pypdfium2),
CORD-v2 test split (100 receipts), and random-noise images at 768², 1024² and
1190×1540. All images are resized so the longest side is ≤ 1540 px (model card).</li>
<li><b>Requests</b>: one image per request, no text, via
<code>/v1/chat/completions</code>, streamed. Real documents stop at EOS with
<code>max_tokens</code> 4096 and the card's sampling (temperature 0.2, top_p 0.9)
unless marked greedy; synthetic images force a fixed output length.</li>
<li><b>Load</b>: closed loop, N concurrent users, 2 pages per user (at least 8).
Pages repeat above 200 requests; prefix caching and the multimodal processor cache
are off, so repeats are not cached.</li>
<li><b>Metrics</b>: sustained pages/s = users ÷ mean latency per page, the
steady-state rate with every user busy (Little's law). Measured pages/s = pages ÷
wall time is lower, by up to ~1.6× at high load, because each run ends with a drain
in which the last long pages run nearly alone; both are in the tables. Latency per
page = request start to last token; TTFT; tok/s per user = 1 / mean time per output
token.</li>
<li><b>Server</b>: <code>--max-model-len 8192 --limit-mm-per-prompt '{{"image":1}}'
--mm-processor-cache-gb 0 --no-enable-prefix-caching</code>, defaults otherwise
(<code>--max-num-batched-tokens 2048 --max-num-seqs 128</code>), 40 GB KV cache,
OpenMP threads bound per NUMA node. TP1 on node 0, TP2 on nodes 3–4, TP4 on
nodes 0–3. Replicas: one server per node (TP1 ×6) or two per node on 15 cores each
(TP1 ×12) behind <code>vllm-router</code>.</li>
</ul>

<h2>2. Single instance: TP1 vs TP2 vs TP4</h2>
{section_single()}

<h3>Vision encoder: tensor-parallel vs data-parallel (TP2)</h3>
{section_encoder()}

<h2>3. Sampling: model-card settings vs greedy, and accuracy</h2>
{section_sampling()}

<h2>4. Image size and output length (synthetic)</h2>
{section_synthetic()}

<h2>5. Max batched tokens</h2>
{section_mnbt()}

<h2>6. Multiple replicas behind vllm-router</h2>
{section_replicas()}

<h2>7. Capacity at a latency target</h2>
{section_capacity()}

<h2>8. Issues found</h2>
{ISSUES}
"""
    return ("<!doctype html><html><head><meta charset='utf-8'><title>LightOnOCR-2-1B "
            f"CPU serving report</title><style>{css}</style></head><body>{body}"
            "</body></html>")


if __name__ == "__main__":
    out = R / "report.html"
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
