"""python -m src.day7.run_retrieval --method dense|bm25|hybrid --split dev --name NAME"""
import argparse
import time
from pathlib import Path

from . import config
from .config import dump_json, load_json
from .metrics import evidence_hit, question_hit_rank, retrieval_summary
from .retrievers import make_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=("dense", "bm25", "hybrid"))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--use-filter", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--name", required=True)
    ap.add_argument("--questions", type=Path, default=config.QUESTIONS_PATH)
    ap.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    ap.add_argument("--chroma", type=Path)
    ap.add_argument("--collection")
    ap.add_argument("--out-root", type=Path, default=config.RESULTS)
    a = ap.parse_args()

    questions = [q for q in load_json(a.questions) if q["split"] == a.split]
    retriever = make_retriever(a.method, a.chroma, a.collection, a.chunks)
    t0 = time.perf_counter()
    traces, ranks, group_ranks = [], [], {}
    for q in questions:
        flt = q.get("filter") if a.use_filter else None
        t = time.perf_counter()
        res = retriever.search(q["question"], filters=flt, top_k=a.top_k)  # original question text only
        latency = time.perf_counter() - t
        rank = question_hit_rank(res, q) if q["answerable"] else None
        if q["answerable"]:
            ranks.append(rank)
            group_ranks.setdefault(q.get("group"), []).append(rank)
        golds = q.get("gold", [])
        traces.append({"id": q["id"], "query": q["question"], "filter": flt, "answerable": q["answerable"],
                       "top": [{"id": r["id"], "page": r["page"], "score": r["score"], "rank": r["rank"],
                                "hit": any(evidence_hit(r, g) for g in golds)} for r in res],
                       "hit_rank": rank, "latency_s": latency})
    tokens = retriever.embed_tokens
    summary = {"method": a.method, "split": a.split, "use_filter": a.use_filter, "top_k": a.top_k,
               **retrieval_summary(ranks),
               "Hit@10_by_group": {g: retrieval_summary(r)["Hit@10"] for g, r in group_ranks.items() if g}, "total_time_s": time.perf_counter() - t0,
               "embedding_tokens": tokens, "cost_usd": tokens * config.PRICES[config.EMBED_MODEL]["input"] / 1e6,
               "index": {"collection": (a.collection or config.COLLECTION) if a.method != "bm25" else None,
                         "bm25_corpus": a.chunks.name if a.method != "dense" else None,
                         "n_chunks": len(load_json(a.chunks))}}
    out = a.out_root / "retrieval" / f"{a.name}-{a.split}"
    dump_json(out / "traces.json", traces)
    dump_json(out / "summary.json", summary)
    for k in ("Hit@1", "Hit@3", "Hit@10", "MRR@10", "Hit@10_by_group"):
        print(k, summary[k])
    print("cost_usd", summary["cost_usd"], "time_s", round(summary["total_time_s"], 2))


if __name__ == "__main__":
    main()
