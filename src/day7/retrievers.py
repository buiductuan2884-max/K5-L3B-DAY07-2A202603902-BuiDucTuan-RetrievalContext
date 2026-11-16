import math
import re
from collections import Counter

from . import config
from .config import dump_json, load_json

STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were which with".split())


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


def _result(chunk, score, rank, method):
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "metadata": chunk["metadata"], "score": score, "rank": rank, "method": method}


class DenseRetriever:
    method = "dense"

    def __init__(self, client=None, chroma=None, collection=None):
        import chromadb
        self.client = client or config.openai_client()
        self.col = chromadb.PersistentClient(path=str(chroma or config.CHROMA_DIR)).get_collection(collection or config.COLLECTION)
        self.embed_tokens = 0
        self.cache_path = config.CACHE / "query_vectors.json"
        self.cache = load_json(self.cache_path) if self.cache_path.exists() else {}

    def _embed(self, query):
        key = f"{config.EMBED_MODEL}|{query}"
        if key not in self.cache:
            r = self.client.embeddings.create(model=config.EMBED_MODEL, input=[query])
            self.embed_tokens += r.usage.total_tokens  # cached queries cost nothing
            self.cache[key] = r.data[0].embedding
            dump_json(self.cache_path, self.cache)
        return self.cache[key]

    def search(self, query, *, filters=None, top_k=3):
        r = self.col.query(query_embeddings=[self._embed(query)], n_results=top_k,
                           where={"chapter": filters["chapter"]} if filters else None)
        out = []
        for i, (id_, doc, meta, dist) in enumerate(zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]), 1):
            rest = {k: v for k, v in meta.items() if k not in ("source", "page", "document_id")}
            chunk = {"id": id_, "text": doc, "source_id": meta["document_id"], "source": meta["source"],
                     "page": meta["page"], "metadata": rest}
            out.append(_result(chunk, 1 - dist, i, self.method))
        return out


class BM25Retriever:
    method = "bm25"
    embed_tokens = 0

    def __init__(self, k1=1.5, b=0.75, chunks=None):
        self.k1, self.b = k1, b
        self.chunks = load_json(chunks or config.CHUNKS_PATH)
        self.tf = [Counter(tokenize(c["text"])) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query, *, filters=None, top_k=3):
        terms = set(tokenize(query))
        scored = []
        for c, tf, ln in zip(self.chunks, self.tf, self.len):
            if filters and c["metadata"].get("chapter") != filters["chapter"]:
                continue
            s = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    for w in terms if w in tf)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [_result(c, s, i, self.method) for i, (s, c) in enumerate(scored[:top_k], 1)]


def rrf(rank_lists, k=60):
    """rank_lists: lists of ids, best first. Returns [(id, score)] best first."""
    score = Counter()
    for ids in rank_lists:
        for r, id_ in enumerate(ids, 1):
            score[id_] += 1 / (k + r)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    method = "hybrid"

    def __init__(self, dense=None, bm25=None, n=20, k=60):
        self.dense, self.bm25, self.n, self.k = dense or DenseRetriever(), bm25 or BM25Retriever(), n, k

    @property
    def embed_tokens(self):
        return self.dense.embed_tokens

    def search(self, query, *, filters=None, top_k=3):
        d = self.dense.search(query, filters=filters, top_k=self.n)
        b = self.bm25.search(query, filters=filters, top_k=self.n)
        by_id = {r["id"]: r for r in b + d}
        dr = {r["id"]: r["rank"] for r in d}
        br = {r["id"]: r["rank"] for r in b}
        out = []
        for i, (id_, s) in enumerate(rrf([[r["id"] for r in d], [r["id"] for r in b]], self.k)[:top_k], 1):
            meta = {**by_id[id_]["metadata"], "dense_rank": dr.get(id_), "bm25_rank": br.get(id_)}
            out.append({**by_id[id_], "metadata": meta, "score": s, "rank": i, "method": self.method})
        return out


def make_retriever(method, chroma=None, collection=None, chunks=None):
    if method == "dense":
        return DenseRetriever(chroma=chroma, collection=collection)
    if method == "bm25":
        return BM25Retriever(chunks=chunks)
    return HybridRetriever(DenseRetriever(chroma=chroma, collection=collection), BM25Retriever(chunks=chunks))
