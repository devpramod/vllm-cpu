#!/usr/bin/env python
"""Post-process the converted multi-turn ShareGPT file: end every
conversation on an assistant turn, keep those that fit the 4096-token context,
and sample 1000 of them."""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
MAX_TOKENS = 3500

tok = AutoTokenizer.from_pretrained(ROOT / "models/gpt-oss-20b")
convs = json.load(open(ROOT / "data/sharegpt_multiturn_raw.json"))
kept = []
for c in convs:
    msgs = c["messages"][: len(c["messages"]) // 2 * 2]
    if len(msgs) < 4:
        continue
    n = sum(len(tok(m["content"])["input_ids"]) for m in msgs)
    if n <= MAX_TOKENS:
        kept.append({"id": c["id"], "messages": msgs})
random.seed(0)
kept = random.sample(kept, min(1000, len(kept)))
json.dump(kept, open(ROOT / "data/sharegpt_multiturn.json", "w"), indent=1)
tot = [sum(len(tok(m["content"])["input_ids"]) for m in c["messages"]) for c in kept]
print(f"{len(kept)} conversations, turns/conv "
      f"{sum(len(c['messages']) // 2 for c in kept) / len(kept):.2f}, "
      f"tokens/conv mean {sum(tot) / len(tot):.0f} max {max(tot)}")
