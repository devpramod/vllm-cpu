#!/usr/bin/env python
"""Write ShareGPT first-turn prompts whose token length is within +/-20% of
each target input length, as JSONL for `vllm bench serve --dataset-name custom`.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
TARGETS = [128, 1024, 2048]
TOL = 0.2

tok = AutoTokenizer.from_pretrained(ROOT / "models/gpt-oss-20b")
convs = json.load(open(ROOT / "data/ShareGPT_V3_unfiltered_cleaned_split.json"))
prompts = list(dict.fromkeys(
    c["conversations"][0]["value"] for c in convs
    if len(c["conversations"]) >= 2 and c["conversations"][0]["from"] == "human"
))
lens = [len(ids) for ids in tok(prompts)["input_ids"]]

for t in TARGETS:
    lo, hi = int(t * (1 - TOL)), int(t * (1 + TOL))
    sel = [(p, n) for p, n in zip(prompts, lens) if lo <= n <= hi]
    out = ROOT / f"data/sharegpt_in{t}.jsonl"
    with open(out, "w") as f:
        for p, _ in sel:
            f.write(json.dumps({"prompt": p}) + "\n")
    mean = sum(n for _, n in sel) / len(sel)
    print(f"{out.name}: {len(sel)} prompts, {lo}-{hi} tok, mean {mean:.0f}")
