"""Offline checks for the Day 7 code; no network."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7 import agent, metrics, retrievers


def chunk(id_, text, page=1, **meta):
    return {"id": id_, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page, "metadata": meta}


class BM25Test(unittest.TestCase):
    def test_exact_term_first(self):
        r = retrievers.BM25Retriever.__new__(retrievers.BM25Retriever)
        r.k1, r.b = 1.5, 0.75
        r.chunks = [chunk("a", "the market sets prices"), chunk("b", "elasticity of demand measures response"),
                    chunk("c", "supply shifts right", chapter=4)]
        r.tf = [retrievers.Counter(retrievers.tokenize(c["text"])) for c in r.chunks]
        r.len = [sum(t.values()) for t in r.tf]
        r.avg = sum(r.len) / 3
        df = retrievers.Counter(w for t in r.tf for w in t)
        r.idf = {w: retrievers.math.log(1 + (3 - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.assertEqual(r.search("elasticity", top_k=2)[0]["id"], "b")
        self.assertEqual([x["id"] for x in r.search("supply market", filters={"chapter": 4})], ["c"])


class RRFTest(unittest.TestCase):
    def test_order(self):
        out = retrievers.rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
        self.assertEqual([i for i, _ in out], ["b", "c", "a", "d"])
        self.assertAlmostEqual(out[0][1], 1 / 62 + 1 / 61)


class MetricsTest(unittest.TestCase):
    def test_normalize_and_hit(self):
        self.assertEqual(metrics.normalize("It’s  a­b – c"), "it's ab - c")
        c = chunk("a", "Firms’ profits rise.", page=3, page_end=4)
        gold = {"source_id": "doc", "page": 4, "evidence": "firms' profits"}
        self.assertTrue(metrics.evidence_hit(c, gold))
        self.assertFalse(metrics.evidence_hit(c, {**gold, "page": 5}))
        self.assertFalse(metrics.evidence_hit(c, {**gold, "source_id": "x"}))
        self.assertEqual(metrics.question_hit_rank([chunk("z", "no"), c], {"gold": [gold]}), 2)
        s = metrics.retrieval_summary([1, 3, None])
        self.assertEqual((s["Hit@1"]["n"], s["Hit@3"]["n"], s["Hit@3"]["N"]), (1, 2, 3))

    def test_advanced_hit_and_all_rule(self):
        vis = chunk("v", "[table Table 1.1: x] (page 5)", page=5, kind="table", region="Table 1.1")
        g1 = {"source_id": "doc", "page": 5, "evidence": "nope", "region": "Table 1.1", "evidence_type": "text"}
        g2 = {"source_id": "doc", "page": 7, "evidence": "nope", "region": "Figure 2.1", "evidence_type": "visual"}
        self.assertTrue(metrics.evidence_hit(vis, g1))
        self.assertTrue(metrics.evidence_hit(chunk("t", "x", page=7), g2))
        res = [vis, chunk("a", "x", page=1), chunk("t", "x", page=7)]
        self.assertEqual(metrics.question_hit_rank(res, {"gold": [g1, g2], "gold_rule": "all"}), 3)
        self.assertEqual(metrics.question_hit_rank(res[:2], {"gold": [g1, g2], "gold_rule": "all"}), None)
        self.assertEqual(metrics.question_hit_rank(res[:2], {"gold": [g1, g2]}), 1)


class ContextEvidenceTest(unittest.TestCase):
    def test_all_rule_needs_every_gold_in_context(self):
        g1 = {"source_id": "doc", "page": 1, "evidence": "alpha"}
        g2 = {"source_id": "doc", "page": 2, "evidence": "beta"}
        q = {"gold": [g1, g2], "gold_rule": "all"}
        self.assertFalse(metrics.context_has_evidence([chunk("a", "alpha", page=1)], q))
        self.assertTrue(metrics.context_has_evidence([chunk("a", "alpha", page=1), chunk("b", "beta", page=2)], q))
        self.assertTrue(metrics.context_has_evidence([chunk("a", "alpha", page=1)], {"gold": [g1, g2]}))


class FakeClient:
    """Replays scripted tool calls: [(name, args), ...]; records the tools offered per call."""
    def __init__(self, script):
        self.script, self.offered = iter(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.offered.append([t["function"]["name"] for t in kw["tools"]])
        name, args = next(self.script)
        tc = NS(id="t", function=NS(name=name, arguments=json.dumps(args)))
        return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], usage=NS(prompt_tokens=10, completion_tokens=5))


class FakeRetriever:
    embed_tokens = 0

    def search(self, query, *, filters=None, top_k=3):
        return [{**chunk("c1", "Demand falls when price rises."), "score": 0.9, "rank": 1, "method": "fake"}]


CFG = {"method": "fake", "top_k": 3, "max_searches": 2, "budget_chars": 2400, "use_filter": False, "context_policy": "first"}
ITEM = {"id": "q", "question": "What happens to demand?"}


def go(script, **cfg):
    client = FakeClient(script)
    return agent.run(ITEM, {**CFG, **cfg}, client=client, retriever=FakeRetriever()), client


class AgentTest(unittest.TestCase):
    def test_answered(self):
        r, _ = go([("search", {"query": "demand"}), ("answer", {"answer": "It falls [c1].", "citations": ["c1"]})])
        self.assertEqual((r["status"], r["citations"], r["context_ids"]), ("answered", ["c1"], ["c1"]))
        self.assertEqual([e["event"] for e in r["events"]], ["search", "answer"])
        self.assertEqual(r["n_model_calls"], 2)

    def test_citation_not_in_context(self):
        bad = ("answer", {"answer": "It falls [zz].", "citations": ["zz"]})
        r, _ = go([("search", {"query": "demand"}), bad, bad])  # one correction turn, then abstain
        self.assertEqual(r["status"], "abstain")
        self.assertEqual([e["event"] for e in r["events"]][-2:], ["citation_retry", "invalid_citations"])
        self.assertIn("raw_answer", r)

    def test_citation_missing_from_text(self):
        bad = ("answer", {"answer": "It falls.", "citations": ["c1"]})
        r, _ = go([("search", {"query": "demand"}), bad, bad])
        self.assertEqual(r["status"], "abstain")

    def test_verifier_reject_and_accept(self):
        ans = ("answer", {"answer": "It falls [c1].", "citations": ["c1"]})
        # reject -> one retry turn -> reject again -> abstain (5 calls); accept -> answered (3 calls)
        for supported, status, n in ((False, "abstain", 5), (True, "answered", 3)):
            client = FakeClient([("search", {"query": "demand"}), ans, ans])
            tool_create = client.create
            def create(**kw):  # the verify call has no tools and returns JSON content
                if "tools" in kw:
                    return tool_create(**kw)
                return NS(choices=[NS(message=NS(content=json.dumps({"supported": supported, "reason": "r"})))],
                          usage=NS(prompt_tokens=7, completion_tokens=3))
            client.chat = NS(completions=NS(create=create))
            r = agent.run(ITEM, {**CFG, "verify": True}, client=client, retriever=FakeRetriever())
            self.assertEqual(r["status"], status)
            self.assertEqual(r["n_model_calls"], n)

    def test_region_siblings(self):
        t1 = {**chunk("a", "rows 1-5", page=144), "metadata": {"region": "Table 6.1"}, "score": 0.5, "rank": 2}
        t2 = {**chunk("b", "rows 6-9", page=145), "metadata": {"region": "Table 6.1"}}
        far = {**chunk("c", "other", page=300), "metadata": {"region": "Table 6.1"}}
        txt = {**chunk("d", "text", page=145), "metadata": {"region": "text"}}
        sib = agent.region_siblings(NS(chunks=[t1, t2, far, txt]), [t1])
        self.assertEqual([(s["id"], s["score"], s["rank"]) for s in sib], [("b", 0.5, 2)])

    def test_citation_retry_then_valid(self):
        r, _ = go([("search", {"query": "demand"}), ("answer", {"answer": "It falls.", "citations": ["c1"]}),
                   ("answer", {"answer": "It falls [c1].", "citations": ["c1"]})])
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["events"][1]["missing_inline"], ["c1"])

    def test_abstain_tool(self):
        r, _ = go([("search", {"query": "x"}), ("abstain", {"reason": "nothing"})])
        self.assertEqual((r["status"], r["events"][-1]["event"]), ("abstain", "abstain"))

    def test_max_searches_forces_final_call(self):
        r, c = go([("search", {"query": "a"}), ("answer", {"answer": "Falls [c1].", "citations": ["c1"]})], max_searches=1)
        self.assertEqual(c.offered, [["search", "answer", "abstain"], ["answer", "abstain"]])
        self.assertEqual([e["event"] for e in r["events"]], ["search", "search_limit", "answer"])
        self.assertEqual(r["status"], "answered")

    def test_error_is_reported(self):
        r, _ = go([])  # exhausted script raises StopIteration -> error status
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["error"])


if __name__ == "__main__":
    unittest.main()
