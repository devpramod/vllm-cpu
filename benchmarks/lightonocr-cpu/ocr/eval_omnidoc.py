"""Page-level text edit distance on the OmniDocBench subset.

Approximates OmniDocBench's text metric: tables, display equations, figures,
headers/footers and page numbers are dropped from both sides, then the
remaining text is compared as one string (whitespace and markdown removed).
Lower is better.
"""

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

SETS = Path(__file__).parent / "data" / "sets"
DROP = [
    r"<table.*?</table>", r"\$\$.*?\$\$", r"\\\[.*?\\\]", r"!\[[^\]]*\]\([^)]*\)",
]
MD = re.compile(r"[#*_>|`~\-\s]+")


def norm(s: str) -> str:
    for pat in DROP:
        s = re.sub(pat, " ", s, flags=re.S)
    s = re.sub(r"^\s*\|.*\|\s*$", " ", s, flags=re.M)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return MD.sub("", s)


def score(rec_path: Path) -> dict:
    refs = {p["id"]: p for p in map(json.loads, open(SETS / "omnidoc/manifest.jsonl"))}
    seen, per_cat = {}, defaultdict(list)
    for r in map(json.loads, open(rec_path)):
        if r["id"] in seen or "text" not in r:
            continue
        ref, pred = norm(refs[r["id"]]["ref_text"]), norm(r["text"])
        if not ref:
            continue
        seen[r["id"]] = Levenshtein.normalized_distance(pred, ref)
        per_cat[r["category"]].append(seen[r["id"]])
    vals = sorted(seen.values())
    return {
        "pages": len(vals),
        "ned_mean": sum(vals) / len(vals),
        "ned_median": vals[len(vals) // 2],
        "by_category": {k: sum(v) / len(v) for k, v in sorted(per_cat.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {p: score(Path(p)) for p in a.records}
    for p, s in res.items():
        print(f"{p}: pages={s['pages']} ned_mean={s['ned_mean']:.3f} "
              f"median={s['ned_median']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
