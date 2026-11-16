"""Export Day 6 chunks and questions to Day 7 schema: python -m src.day7.export_day6"""
from collections import Counter

import chromadb

from . import config
from .config import dump_json, load_json
from .metrics import evidence_hit


def export_chunks():
    col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION)
    r = col.get(include=["documents", "metadatas"])
    chunks = []
    for id_, doc, m in sorted(zip(r["ids"], r["documents"], r["metadatas"])):
        rest = {k: v for k, v in m.items() if k not in ("source", "page", "document_id")}
        chunks.append({"id": id_, "text": doc, "source_id": m["document_id"], "source": m["source"],
                       "page": m["page"], "metadata": rest})
    return chunks


def convert_question(q):
    return {"id": q["id"], "split": "dev", "day6_split": q["split"], "type": q["type"], "question": q["question"],
            "expected_answer": q["expected_answer"], "answerable": True,
            "gold": [{"source_id": q["gold_source_id"], "page": p, "evidence": q["evidence_quote"]} for p in q["gold_pages"]],
            "filter": q.get("filter")}


def validate(questions):
    ids = [q["id"] for q in questions]
    assert len(set(ids)) == len(ids), "duplicate question ids"
    for q in questions:
        assert q.get("split") in ("dev", "test"), f"{q['id']}: bad split"
        for k in ("id", "question", "answerable"):
            assert k in q, f"{q.get('id')}: missing {k}"
        if q["answerable"]:
            assert q.get("expected_answer") and q.get("gold"), f"{q['id']}: missing expected_answer/gold"
            for g in q["gold"]:
                assert all(g.get(k) not in (None, "") for k in ("source_id", "page", "evidence")), f"{q['id']}: bad gold"
        else:
            assert q.get("expected_behavior") == "search_and_abstain" and q.get("absence_reason"), f"{q['id']}: bad unanswerable"


def main():
    chunks = export_chunks()
    dump_json(config.CHUNKS_PATH, chunks)
    questions = [convert_question(q) for q in load_json(config.DAY6_QUESTIONS)]
    if config.NEW_QUESTIONS_PATH.exists():
        questions += load_json(config.NEW_QUESTIONS_PATH)
    validate(questions)
    dump_json(config.QUESTIONS_PATH, questions)
    print(f"chunks: {len(chunks)}")
    for (split, ans), n in sorted(Counter((q["split"], q["answerable"]) for q in questions).items()):
        print(f"{split:5} answerable={ans!s:5} {n}")
    answerable = [q for q in questions if q["answerable"]]
    full = [q["id"] for q in answerable if any(evidence_hit(c, g) for c in chunks for g in q["gold"])]
    print(f"answerable with evidence fully inside a chunk : {len(full)}/{len(answerable)}")
    missing = [q["id"] for q in answerable if q["id"] not in full]
    if missing:
        print("not fully inside any chunk:", ", ".join(missing))


if __name__ == "__main__":
    main()
