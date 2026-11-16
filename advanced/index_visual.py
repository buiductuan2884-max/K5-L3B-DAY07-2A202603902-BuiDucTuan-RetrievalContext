"""python -m advanced.index_visual [--pages 517,736]  ->  advanced/data/chunks_visual.json
Renders candidate pages (images / vector drawings / Table|Figure captions), transcribes them with a vision model.
Raw responses are cached per page in advanced/data/cache/vision/p{page}.json, so reruns resume."""
import argparse
import base64
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf

from src.day7 import config
from src.day7.config import dump_json, load_json
from .index_text import chapter_map

CACHE = config.ADV_DATA / "cache" / "vision"
CAPTION = re.compile(r"^(Table|Figure) \d+\.\d+", re.M)
PROMPT = """You transcribe the visual content of one textbook page so that questions can be answered from your text alone.
The page image is attached, followed by the text the PDF extracted from the page (use it to confirm labels, not as a substitute for looking).
Return strict JSON: {"items":[{"region": "Figure 30.8" or "Table 21.4" or "unlabelled image", "kind": "table"|"chart"|"diagram"|"photo"|"other", "title": "...", "content": "..."}]}
One item per table, chart, diagram or image on the page. Use the printed caption label as region when there is one.
Rules for content:
- table: full markdown transcription of every row and column, with the header and the units.
- chart: chart type, axis titles, units, ranges, every series name, and the approximate values read at labelled ticks, peaks, troughs, and the start and end of each series (say "about" for read-off values).
- diagram: every label and what each arrow or line connects, with positions and ordering.
- photo, decorative images, QR codes, logos: kind photo or other with one short line.
Never invent values that are not visible; write "unreadable" when unsure. If the page has no such content return {"items":[]}."""
lock = threading.Lock()


def candidates(doc):
    out = []
    for i, page in enumerate(doc, 1):
        if page.get_images() or len(page.get_drawings()) >= 15 or CAPTION.search(page.get_text()):
            out.append(i)
    return out


def transcribe(doc, client, page_no):
    path = CACHE / f"p{page_no}.json"
    if path.exists():
        return None
    with lock:  # pymupdf is not thread-safe
        page = doc[page_no - 1]
        png, text = page.get_pixmap(dpi=150).tobytes("png"), page.get_text()
    err = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=config.VISION_MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": f"{PROMPT}\n\nExtracted page text:\n{text}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode(), "detail": "high"}}]}])
            raw = r.choices[0].message.content
            json.loads(raw)
            dump_json(path, {"page": page_no, "raw": raw, "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens})
            return None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** (attempt + 1))
    return {"page": page_no, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", help="comma-separated 1-based pages (default: all candidates)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    doc = pymupdf.open(config.ADV_PDF)
    cands = candidates(doc)
    print("candidate pages:", len(cands))
    pages = [int(p) for p in a.pages.split(",")] if a.pages else cands
    CACHE.mkdir(parents=True, exist_ok=True)
    client = config.openai_client()
    with ThreadPoolExecutor(8) as ex:
        failures = [f for f in ex.map(lambda p: transcribe(doc, client, p), pages) if f]
    dump_json(config.ADV_DATA / "visual_failures.json", failures)

    chap = chapter_map(doc.get_toc(), len(doc))
    chunks, kinds, tin, tout, pages_with_items = [], {}, 0, 0, 0
    for p in pages:
        path = CACHE / f"p{p}.json"
        if not path.exists():
            continue
        c = load_json(path)
        tin, tout = tin + c["input_tokens"], tout + c["output_tokens"]
        items = [it for it in json.loads(c["raw"]).get("items", []) if isinstance(it, dict)]
        n0 = len(chunks)
        for idx, it in enumerate(items):
            kind, region = it.get("kind", "other"), str(it.get("region") or "unlabelled image")
            content, title = str(it.get("content") or ""), str(it.get("title") or "")
            if kind not in ("table", "chart", "diagram", "photo", "other"):
                kind = "other"
            if kind in ("photo", "other") and len(content) < 40:
                continue
            hits = doc[p - 1].search_for(region)
            text = f"[{kind} {region}: {title}] (page {p})\n{content}"
            chunks.append({"id": hashlib.sha1(f"{p}|v{idx}|{text}".encode()).hexdigest()[:16], "text": text,
                           "source_id": config.ADV_SOURCE_ID, "source": "Economics2e.pdf", "page": p,
                           "metadata": {"chapter": chap[p][0], "chapter_title": chap[p][1], "kind": kind, "region": region,
                                        "caption_bbox": [round(v, 1) for v in hits[0]] if hits else None}})
            kinds[kind] = kinds.get(kind, 0) + 1
        pages_with_items += len(chunks) > n0
    pr = config.PRICES[config.VISION_MODEL]
    stats = {"candidate_pages": len(cands), "pages_processed": len(pages), "pages_with_items": pages_with_items,
             "items_by_kind": kinds, "failures": len(failures), "input_tokens": tin, "output_tokens": tout,
             "cost_usd": (tin * pr["input"] + tout * pr["output"]) / 1e6, "elapsed_s": time.perf_counter() - t0}
    dump_json(config.ADV_DATA / "chunks_visual.json", chunks)
    dump_json(config.ADV_DATA / "visual_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
