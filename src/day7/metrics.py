import re
import unicodedata


def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub("[‘’‚‛′ʼ“”„‟�`\"]", "'", t)
    t = re.sub("[‐‑‒–—―−]", "-", t)
    t = t.replace("­", "")
    return re.sub(r"\s+", " ", t).strip()


def evidence_hit(chunk, gold_item):
    """gold_item: {source_id, page, evidence[, region, evidence_type]}. Evidence must lie fully inside the chunk.
    Advanced gold also hits via a visual chunk of the same region, or any chunk on the page for visual evidence."""
    meta = chunk.get("metadata", {})
    last = meta.get("page_end") or chunk["page"]
    if not (chunk["source_id"] == gold_item["source_id"] and chunk["page"] <= gold_item["page"] <= last):
        return False
    if gold_item.get("evidence_type") == "visual":
        return True
    if meta.get("kind") not in (None, "text") and gold_item.get("region") and meta.get("region") == gold_item["region"]:
        return True
    return normalize(gold_item["evidence"]) in normalize(chunk["text"])


def question_hit_rank(results, question):
    """Rank at which the question counts as hit: first gold hit (rule any) or the rank covering every gold (rule all)."""
    ranks = [next((i for i, c in enumerate(results, 1) if evidence_hit(c, g)), None) for g in question["gold"]]
    if question.get("gold_rule") == "all":
        return None if None in ranks else max(ranks)
    ranks = [r for r in ranks if r]
    return min(ranks) if ranks else None


def context_has_evidence(chunks, question):
    return question_hit_rank(chunks, question) is not None  # honours gold_rule "all" for multi-hop


def _rate(n, N):
    return {"n": n, "N": N, "rate": n / N if N else None}


def retrieval_summary(ranks):
    """ranks: hit rank (or None) per answerable question."""
    N = len(ranks)
    mrr = sum(1 / r for r in ranks if r and r <= 10)
    return {"Hit@1": _rate(sum(r == 1 for r in ranks), N),
            "Hit@3": _rate(sum(bool(r) and r <= 3 for r in ranks), N),
            "Hit@10": _rate(sum(bool(r) and r <= 10 for r in ranks), N),
            "MRR@10": {"n": round(mrr, 4), "N": N, "rate": mrr / N if N else None}}


def context_evidence_rate(flags):
    return _rate(sum(flags), len(flags))
