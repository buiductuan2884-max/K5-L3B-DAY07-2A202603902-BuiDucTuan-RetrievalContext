"""python -m advanced.build_index  ->  advanced/data/chunks.json, advanced/chroma_db, advanced/data/index_stats.json"""
import json
import time
from collections import Counter

import chromadb

from src.day7 import config
from src.day7.config import dump_json, load_json

BATCH = 100


def main():
    t0 = time.perf_counter()
    d = config.ADV_DATA
    chunks = load_json(d / "chunks_text.json") + load_json(d / "chunks_visual.json")
    dump_json(d / "chunks.json", chunks)
    client = config.openai_client()
    col = chromadb.PersistentClient(path=str(config.ADV_CHROMA_DIR)).get_or_create_collection(
        config.ADV_COLLECTION, metadata={"hnsw:space": "cosine"})
    tokens = 0
    for i in range(0, len(chunks), BATCH):
        b = chunks[i:i + BATCH]
        r = client.embeddings.create(model=config.EMBED_MODEL, input=[c["text"] for c in b])
        tokens += r.usage.total_tokens
        meta = [{"document_id": c["source_id"], "source": c["source"], "page": c["page"],
                 **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c["metadata"].items() if v is not None}} for c in b]
        col.upsert(ids=[c["id"] for c in b], embeddings=[e.embedding for e in r.data],
                   documents=[c["text"] for c in b], metadatas=meta)
    cost = tokens * config.PRICES[config.EMBED_MODEL]["input"] / 1e6
    print("count", col.count(), "tokens", tokens, "cost_usd", round(cost, 4))
    ts, vs = load_json(d / "text_stats.json"), load_json(d / "visual_stats.json")
    dump_json(d / "index_stats.json", {
        "pages_total": ts["pages_total"], "pages_with_text": ts["pages_with_text"],
        "candidate_visual_pages": vs["candidate_pages"], "pages_with_visual_items": vs["pages_with_items"],
        "visual_items_by_kind": vs["items_by_kind"], "visual_failures": load_json(d / "visual_failures.json"),
        "chunks_by_kind": dict(Counter(c["metadata"]["kind"] for c in chunks)), "chunks_total": len(chunks),
        "embedding_tokens": tokens, "embedding_cost_usd": cost,
        "vision_input_tokens": vs["input_tokens"], "vision_output_tokens": vs["output_tokens"], "vision_cost_usd": vs["cost_usd"],
        "elapsed_s": {"text": ts["elapsed_s"], "vision_last_run": vs["elapsed_s"], "embed": time.perf_counter() - t0}})


if __name__ == "__main__":
    main()
