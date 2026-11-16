"""python -m src.day7.run_agent_bench --config configs/NAME.json --split dev [--limit N] [--ids q01,q02]"""
import argparse
from collections import Counter
from pathlib import Path

from . import config
from .agent import run
from .config import dump_json, load_json
from .metrics import context_evidence_rate, context_has_evidence
from .retrievers import make_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--questions", type=Path, default=config.QUESTIONS_PATH)
    ap.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    ap.add_argument("--chroma", type=Path)
    ap.add_argument("--collection")
    ap.add_argument("--out-root", type=Path, default=config.RESULTS)
    a = ap.parse_args()

    cfg = load_json(a.config)
    name = a.config.stem
    questions = [q for q in load_json(a.questions) if q["split"] == a.split]
    if a.ids:
        questions = [q for q in questions if q["id"] in a.ids.split(",")]
    if a.limit:
        questions = questions[:a.limit]
    chunks = {c["id"]: c for c in load_json(a.chunks)}
    client, retriever = config.openai_client(), make_retriever(cfg["method"], a.chroma, a.collection, a.chunks)

    traces = []
    for q in questions:
        t = run(q, cfg, client=client, retriever=retriever)
        if q["answerable"]:
            t["context_evidence"] = context_has_evidence([chunks[i] for i in t["context_ids"]], q)
        traces.append({"query_id": q["id"], "split": q["split"], "answerable": q["answerable"], **t})
        print(q["id"], t["status"], [e["event"] for e in t["events"]], round(t["latency_s"], 1), "s", t["error"] or "")

    ok = [t for t in traces if t["status"] != "error"]
    by_ans = {k: Counter(t["status"] for t in traces if t["answerable"] == (k == "answerable") and t["status"] != "error")
              for k in ("answerable", "unanswerable")}
    n_search = Counter(sum(e["event"] == "search" for e in t["events"]) for t in ok)
    summary = {
        "config": cfg, "name": name, "split": a.split, "model": cfg.get("model", config.CHAT_MODEL), "n_questions": len(traces),
        "status_counts": dict(Counter(t["status"] for t in traces)),
        "answered_abstain_by_answerable": {k: dict(v) for k, v in by_ans.items()},
        "tool_use": {"n": sum(c for n, c in n_search.items() if n > 0), "N": len(ok)},
        "searches_distribution": {str(k): v for k, v in sorted(n_search.items())},
        "context_evidence_rate": context_evidence_rate([t["context_evidence"] for t in traces if t["answerable"]]),
        "total_latency_s": sum(t["latency_s"] for t in traces),
        "mean_latency_s": sum(t["latency_s"] for t in traces) / len(traces) if traces else None,
        "prompt_tokens": sum(t["prompt_tokens"] for t in traces), "completion_tokens": sum(t["completion_tokens"] for t in traces),
        "embedding_tokens": sum(t["embedding_tokens"] for t in traces), "cost_usd": sum(t["cost_usd"] for t in traces)}
    summary["tool_use"]["rate"] = summary["tool_use"]["n"] / len(ok) if ok else None
    out = a.out_root / "agent" / f"{name}-{a.split}"
    dump_json(out / "traces.json", traces)
    dump_json(out / "summary.json", summary)
    dump_json(out / "cases.json", [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions])
    print({k: v for k, v in summary.items() if k != "config"})


if __name__ == "__main__":
    main()
