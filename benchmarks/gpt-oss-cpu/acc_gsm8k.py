#!/usr/bin/env python
"""GSM8K exact-match accuracy against a running vLLM server (greedy)."""

import argparse
import asyncio
import json
import re
import time

from datasets import load_dataset
from openai import AsyncOpenAI

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
SUFFIX = "\nGive the final answer as a plain number after '####'."


def to_num(s: str | None) -> float | None:
    if not s:
        return None
    m = NUM_RE.findall(s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract(text: str) -> float | None:
    if "####" in text:
        return to_num(text.rsplit("####", 1)[1].split("\n")[0])
    return to_num(text)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--num", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.num))
    client = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="x",
                         timeout=1800)
    model = (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(a.concurrency)

    async def one(item):
        async with sem:
            r = await client.chat.completions.create(
                model=model, temperature=0, max_tokens=2048,
                messages=[{"role": "user", "content": item["question"] + SUFFIX}])
            text = r.choices[0].message.content or ""
            gold = to_num(item["answer"].rsplit("####", 1)[1])
            pred = extract(text)
            return {"correct": pred is not None and abs(pred - gold) < 1e-6,
                    "pred": pred, "gold": gold,
                    "finish": r.choices[0].finish_reason,
                    "completion_tokens": r.usage.completion_tokens}

    t0 = time.time()
    res = await asyncio.gather(*(one(x) for x in ds))
    acc = sum(r["correct"] for r in res) / len(res)
    summary = {"num": len(res), "accuracy": acc,
               "truncated": sum(r["finish"] == "length" for r in res),
               "mean_completion_tokens":
                   sum(r["completion_tokens"] for r in res) / len(res),
               "wall_s": time.time() - t0, "results": res}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    asyncio.run(main())
