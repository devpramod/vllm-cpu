"""Closed-loop OCR load generator for an OpenAI-compatible vLLM server.

Each request is one page (image-only user message), streamed so TTFT and
per-token latency can be measured. Writes a summary JSON and, optionally,
per-request records including the generated text.
"""

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import aiohttp

SETS = Path(__file__).parent / "data" / "sets"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def load_pages(name):
    pages = [json.loads(l) for l in open(SETS / name / "manifest.jsonl")]
    for p in pages:
        b64 = base64.b64encode(Path(p["image"]).read_bytes()).decode()
        p["url"] = f"data:image/png;base64,{b64}"
    return pages


async def one(session, url, model, page, args, idx):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": page["url"]}}]}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    rec = {"id": page["id"], "category": page["category"], "idx": idx}
    t0 = time.perf_counter()
    ttft, text, usage, finish = None, [], None, None
    try:
        async with session.post(url, json=body) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                return rec
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    delta = ch.get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text.append(delta)
                    finish = ch.get("finish_reason") or finish
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        return rec
    e2e = time.perf_counter() - t0
    if usage is None:
        rec["error"] = "stream ended without usage"
        return rec
    out = usage["completion_tokens"]
    rec.update(start=t0, e2e=e2e, ttft=ttft, out_tokens=out, finish=finish,
               prompt_tokens=usage["prompt_tokens"])
    if ttft is not None and out and out > 1:
        rec["tpot"] = (e2e - ttft) / (out - 1)
    if args.save_text:
        rec["text"] = "".join(text)
    return rec


async def run(args):
    pages = load_pages(args.set)
    n = args.num_pages or max(args.min_pages, args.concurrency * args.pages_per_user)
    work = [(i, pages[i % len(pages)]) for i in range(n)]
    url = f"http://localhost:{args.port}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        async with s.get(f"http://localhost:{args.port}/v1/models") as r:
            model = (await r.json())["data"][0]["id"]
        q = asyncio.Queue()
        for w in work:
            q.put_nowait(w)
        recs = []

        async def worker():
            while not q.empty():
                i, p = q.get_nowait()
                recs.append(await one(s, url, model, p, args, i))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        dur = time.perf_counter() - t0

    ok = [r for r in recs if "error" not in r]
    lat = [r["e2e"] for r in ok]
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpot = [r["tpot"] for r in ok if "tpot" in r]
    out = sum(r["out_tokens"] or 0 for r in ok)
    summary = {
        "set": args.set, "concurrency": args.concurrency, "num_pages": n,
        "completed": len(ok), "failed": len(recs) - len(ok), "duration_s": dur,
        "pages_per_s": len(ok) / dur,
        "latency_mean_s": statistics.mean(lat) if lat else None,
        "latency_p50_s": pct(lat, 50), "latency_p90_s": pct(lat, 90),
        "latency_p99_s": pct(lat, 99),
        "ttft_mean_s": statistics.mean(ttft) if ttft else None,
        "ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99),
        "tpot_mean_ms": 1000 * statistics.mean(tpot) if tpot else None,
        "tok_s_user": 1 / statistics.mean(tpot) if tpot else None,
        "output_tok_s": out / dur,
        "out_tokens_mean": out / len(ok) if ok else None,
        "prompt_tokens_mean": (statistics.mean(pt) if (pt := [
            r["prompt_tokens"] for r in ok if r["prompt_tokens"]]) else None),
        "finish_length": sum(r["finish"] == "length" for r in ok),
        "errors": sorted({r["error"] for r in recs if "error" in r})[:5],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    if args.records:
        with open(args.records, "w") as f:
            for r in sorted(recs, key=lambda r: r["idx"]):
                f.write(json.dumps(r) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in summary.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--set", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-pages", type=int)
    ap.add_argument("--pages-per-user", type=int, default=4)
    ap.add_argument("--min-pages", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
