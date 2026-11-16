"""python -m advanced.index_text  ->  advanced/data/chunks_text.json (per-page text chunks, never across pages)"""
import hashlib
import re
import time

import pymupdf

from src.day7 import config
from src.day7.config import dump_json

FOOTER = "This OpenStax book is available for free at http://cnx.org/content/col12122/1.4"
HEADER = re.compile(r"^(Chapter \d+ \| .+|\d+\.\d+ \| .+)$")
TARGET, OVERLAP = 1200, 150


def clean(text):
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l and l != FOOTER]
    drop = {i for i, l in enumerate(lines) if HEADER.match(l)}
    drop |= {j for i in drop for j in (i - 1, i + 1) if 0 <= j < len(lines) and lines[j].isdigit()}  # printed page number
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


def chapter_map(toc, n_pages):
    """Per 1-based page: (chapter int or 0, title) from level-1 TOC entries."""
    starts = sorted((p, t) for lvl, t, p in toc if lvl == 1 and p >= 1)
    out, cur = {}, (0, "")
    for page in range(1, n_pages + 1):
        while starts and starts[0][0] <= page:
            t = starts.pop(0)[1].replace("\xa0", " ")
            m = re.match(r"Chapter (\d+)\. (.*)", t)
            cur = (int(m[1]), m[2]) if m else (0, t)
        out[page] = cur
    return out


def chunk_page(text):
    units = [u for u in re.split(r"(?<=[.!?])\s+|\n+", text) if u]
    pieces, cur = [], ""
    for u in units:
        while len(u) > TARGET:  # shortcut: hard split of a unit with no sentence break
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(u[:TARGET])
            u = u[TARGET:]
        if cur and len(cur) + 1 + len(u) > TARGET:
            pieces.append(cur)
            tail = cur[-OVERLAP:]
            cur = tail[tail.find(" ") + 1:] if " " in tail else tail  # start overlap on a word
        cur = f"{cur} {u}" if cur else u
    if cur and (not pieces or len(cur) > OVERLAP):
        pieces.append(cur)
    return pieces


def main():
    t0 = time.perf_counter()
    doc = pymupdf.open(config.ADV_PDF)
    chap = chapter_map(doc.get_toc(), len(doc))
    chunks, with_text = [], 0
    for i, page in enumerate(doc, 1):
        text = clean(page.get_text())
        if not text.strip():
            continue
        with_text += 1
        for idx, piece in enumerate(chunk_page(text)):
            cid = hashlib.sha1(f"{i}|{idx}|{piece}".encode()).hexdigest()[:16]
            chunks.append({"id": cid, "text": piece, "source_id": config.ADV_SOURCE_ID, "source": "Economics2e.pdf", "page": i,
                           "metadata": {"chapter": chap[i][0], "chapter_title": chap[i][1], "kind": "text"}})
    dump_json(config.ADV_DATA / "chunks_text.json", chunks)
    dump_json(config.ADV_DATA / "text_stats.json", {"pages_total": len(doc), "pages_with_text": with_text,
                                                    "chunks": len(chunks), "elapsed_s": time.perf_counter() - t0})
    print("pages", len(doc), "with text", with_text, "chunks", len(chunks))


if __name__ == "__main__":
    main()
