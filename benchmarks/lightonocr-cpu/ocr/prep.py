"""Build OCR benchmark page sets: ocr/data/sets/<name>/{images/,manifest.jsonl}.

Every image is resized so its longest side is <= 1540 px (LightOnOCR model card).
"""

import io
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pypdfium2 as pdfium
from PIL import Image

ROOT = Path(__file__).parent / "data"
SETS = ROOT / "sets"
MAX_SIDE = 1540
N_PAGES = 200
TEXT_CATS = {
    "text_block", "title", "figure_caption", "table_caption", "equation_caption",
    "reference", "table_footnote", "figure_footnote", "page_footnote",
    "list_group", "code_txt", "code_txt_caption",
}
rng = random.Random(0)


def fit(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = MAX_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def stratified(groups: dict[str, list], n: int) -> list:
    for v in groups.values():
        rng.shuffle(v)
    out, keys = [], sorted(groups)
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def write_set(name: str, items: list[tuple[str, str, Image.Image, dict]]):
    d = SETS / name
    (d / "images").mkdir(parents=True, exist_ok=True)
    with open(d / "manifest.jsonl", "w") as f:
        for pid, cat, img, extra in items:
            p = d / "images" / f"{pid}.png"
            img = fit(img)
            img.save(p)
            f.write(json.dumps({"id": pid, "category": cat, "image": str(p),
                                "width": img.width, "height": img.height, **extra})
                    + "\n")
    print(name, len(items))


def omnidoc():
    src = ROOT / "OmniDocBench"
    groups = defaultdict(list)
    for e in json.load(open(src / "OmniDocBench.json")):
        pi = e["page_info"]
        if pi["page_attribute"]["language"] == "english":
            groups[pi["page_attribute"]["data_source"]].append(e)
    items = []
    for e in stratified(groups, N_PAGES):
        dets = sorted((a for a in e["layout_dets"]
                       if a["category_type"] in TEXT_CATS and not a.get("ignore")
                       and a.get("text")), key=lambda a: a.get("order") or 0)
        ref = "\n".join(a["text"] for a in dets)
        name = e["page_info"]["image_path"]
        img = Image.open(src / "images" / name)
        items.append((Path(name).stem, e["page_info"]["page_attribute"]["data_source"],
                      img, {"ref_text": ref}))
    write_set("omnidoc", items)


def olmocr():
    src = ROOT / "olmOCR-bench" / "bench_data"
    groups = defaultdict(set)
    for f in src.glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            groups[f.stem].add((r["pdf"], r.get("page", 1)))
    groups = {k: sorted(v) for k, v in groups.items()}
    items = []
    for pdf, page in stratified(groups, N_PAGES):
        doc = pdfium.PdfDocument(src / "pdfs" / pdf)
        pg = doc[page - 1]
        scale = MAX_SIDE / max(pg.get_size())
        img = pg.render(scale=scale).to_pil()
        pid = pdf.replace("/", "__").removesuffix(".pdf") + f"_p{page}"
        items.append((pid, pdf.split("/")[0], img, {}))
        doc.close()
    write_set("olmocr", items)


def cord():
    f = next((ROOT / "cord-v2" / "data").glob("test-*.parquet"))
    t = pq.read_table(f).to_pylist()
    items = [(f"cord_{i:03d}", "receipt", Image.open(io.BytesIO(r["image"]["bytes"])),
              {}) for i, r in enumerate(t)]
    write_set("cord", items)


def synthetic():
    for w, h in [(768, 768), (1024, 1024), (1190, 1540)]:
        items = []
        for i in range(8):
            px = rng.randbytes(w * h * 3)
            items.append((f"noise_{w}x{h}_{i}", f"{w}x{h}",
                          Image.frombytes("RGB", (w, h), px), {}))
        write_set(f"synth_{w}x{h}", items)


if __name__ == "__main__":
    omnidoc()
    olmocr()
    cord()
    synthetic()
