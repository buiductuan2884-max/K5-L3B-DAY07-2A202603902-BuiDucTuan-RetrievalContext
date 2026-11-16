"""Demo Agentic RAG (Day 7): streamlit run app.py. Đọc kết quả đã lưu; chỉ tab "Hỏi thử" gọi OpenAI thật."""
import json
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.day7 import agent, config  # noqa: E402
from src.day7.retrievers import make_retriever  # noqa: E402

R = HERE / "results"
st.set_page_config(page_title="Demo Agentic RAG", layout="wide")


@st.cache_data
def load_json(p: str):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def maybe(p: Path):
    return load_json(str(p)) if p.exists() else None


def table(rows: list[dict]):
    """Bảng markdown: st.dataframe/table cần pyarrow, DLL bị Application Control chặn trên máy này."""
    if not rows:
        return st.caption("(trống)")
    cols = list(rows[0])
    st.markdown("\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                          ["| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |" for r in rows]))


def rate(m):  # {"n","N","rate"} -> "n/N (xx%)"
    return f"{m['n']:g}/{m['N']} ({m['rate']:.0%})" if m else "-"


def runs(root: Path):
    """Các thư mục kết quả <tên>-<split> có summary.json."""
    return sorted(d for d in root.glob("*-*") if (d / "summary.json").exists())


@st.cache_data
def chunk_map(p: str):
    return {c["id"]: c for c in load_json(p)}


BASIC_CHUNKS, ADV_CHUNKS = str(config.CHUNKS_PATH), str(config.ADV_DATA / "chunks.json")
ADV_R = R / "advanced"


@st.cache_resource
def get_retriever(corpus: str, method: str):
    if corpus == "basic":
        return make_retriever(method)
    return make_retriever(method, chroma=str(config.ADV_CHROMA_DIR), collection=config.ADV_COLLECTION, chunks=ADV_CHUNKS)


def _phrases(text):
    return [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) >= 25]


PDF_URL = "https://assets.openstax.org/oscms-prodcms/media/documents/Economics2e-OP_s2jF42u.pdf"


@st.cache_data(max_entries=64)
def page_png(page: int, chunk_text: str, evidence: str = "", region: str = "") -> bytes:
    """Ảnh trang sách gốc: tô vàng phần chunk (chunk hình/bảng: tô chú thích region), tô xanh evidence gold."""
    import fitz
    with fitz.open(config.ADV_PDF) as doc:
        p = doc[page - 1]
        yellow = _phrases(chunk_text) + ([region] if region else [])
        green = [s.strip() for s in evidence.replace("\n", " ").split(". ") if len(s.strip()) >= 6]  # giữ cả nhãn ngắn như "Table 6.1"
        for phrases, color in ((yellow, (1, 0.85, 0)), (green, (0.3, 0.9, 0.4))):
            for ph in phrases:
                for r in p.search_for(ph) or p.search_for(ph[:45]):
                    a = p.add_highlight_annot(r)
                    a.set_colors(stroke=color)
                    a.update()
        return p.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).tobytes("png")


def gold_evidence(q, first, last):
    """Evidence gold nằm trong khoảng trang; gold dạng hình là mô tả, nên tô chú thích region (vd "Figure 3.4")."""
    return ". ".join(g.get("region", "") if g.get("evidence_type") == "visual" else g["evidence"]
                     for g in (q or {}).get("gold", []) if first <= g["page"] <= last)


def book_page(c, key, q=None):
    first, last = c["page"], c.get("metadata", {}).get("page_end") or c["page"]
    if not st.toggle(f"Xem trang sách gốc (tr. {first}{'' if last == first else f'-{last}'})", key=key):
        return
    st.markdown(f"[Mở PDF gốc ở trang {first}]({PDF_URL}#page={first}) · 🟨 chunk · 🟩 bằng chứng gold")
    if not Path(config.ADV_PDF).exists():
        return st.info(f"Chưa có file PDF ({config.ADV_PDF}); dùng link ở trên để mở trang gốc.")
    reg = c.get("metadata", {}).get("region", "")
    reg = "" if reg in ("", "text", "unlabelled image") else reg
    for pg in range(first, last + 1):
        st.image(page_png(pg, c["text"], gold_evidence(q, pg, pg), reg), caption=f"Trang PDF {pg}")


def chunk_card(c, head, key, pdf=True, limit=300, q=None):
    with st.container(border=True):
        st.markdown(head)
        t = c["text"]
        st.text(t[:limit] + ("…" if len(t) > limit else ""))
        if len(t) > limit:
            with st.expander("Toàn văn"):
                st.text(t)
        if pdf:
            book_page(c, key, q)


def gold_view(q, key):
    """Bằng chứng gold của câu benchmark, kèm trang PDF tô xanh."""
    if not (q or {}).get("gold"):
        return
    st.markdown("**Bằng chứng gold**")
    for i, g in enumerate(q["gold"]):
        with st.container(border=True):
            st.markdown(f"tr. {g['page']}" + (f" · {g['region']}" if g.get("region") else "")
                        + (" · (mô tả hình)" if g.get("evidence_type") == "visual" else ""))
            st.text(g["evidence"])
            book_page({"page": g["page"], "text": "", "metadata": {}}, f"{key}-gold{i}", q)


def trace_view(t, chunks, key, review=None, label=None, q=None):
    """Hiện một lượt chạy agent: các lần search, verify, câu trả lời + trích dẫn, trạng thái, chi phí."""
    st.markdown(f"**Trạng thái:** {t['status']} · chi phí ${t.get('cost_usd', 0):.5f} · "
                f"độ trễ {t.get('latency_s', 0):.1f}s · {t.get('n_model_calls', '?')} lần gọi model")
    if t.get("error"):
        st.error(f"Lỗi agent: {t['error']}")
    st.markdown("**Các bước của agent**")
    for e in t.get("events", []):
        k = e["event"]
        if k == "search":
            flt = f" · lọc {e['filter']}" if e.get("filter") else ""
            res = ", ".join(f"{i[:6]} (tr.{p})" for i, p, *_ in e["results"])
            st.markdown(f"- 🔎 search `{e['query']}`{flt} → {res}")
            if e.get("region_siblings"):
                st.caption("  + chunk cùng vùng: " + ", ".join(f"{i[:6]} (tr.{p})" for i, p in e["region_siblings"]))
        elif k == "verify":
            st.markdown(f"- {'✅' if e['supported'] else '❌'} verify: {e['reason']}")
        elif k == "citation_retry":
            st.markdown(f"- ↻ citation_retry: ngoài context {e['not_in_context']}, thiếu inline {e['missing_inline']}")
        else:
            st.markdown(f"- {k}")
    st.markdown("**Câu trả lời**")
    st.write(t["answer"] if t["answer"] else "(abstain: không trả lời)")
    if t["citations"]:
        st.markdown("**Trích dẫn**")
    for i, cid in enumerate(t["citations"]):
        c = chunks.get(cid)
        if c:
            chunk_card(c, f"`{cid[:8]}` · tr. {c['page']}", f"{key}-cite{i}", q=q)
        else:
            st.warning(f"Không tìm thấy chunk {cid} trong kho đã lưu.")
    gold_view(q, key)
    if label:
        st.markdown(f"**Nhãn:** {'✅' if label['label'] == 'pass' else '❌'} {label['label']} · {label['reason']}")
    if review:
        st.markdown(f"**Chấm tay:** câu trả lời {review['answer_grade']}/2 · context {review['context_grade']}/2 · "
                    f"trích dẫn đúng {review['citation_support']} · {review['reason']}")


tabs = st.tabs(["Tổng quan", "Retrieval", "Agent – benchmark", "Hỏi thử (agent thật)", "Nâng cao"])

# ---------------------------------------------------------------- 1
with tabs[0]:
    bq, aq = load_json(str(config.QUESTIONS_PATH)), load_json(str(config.ADV_QUESTIONS_PATH))
    st.markdown("""**Pipeline**
1. **Kho Day 6**: chunk sách *Principles of Economics 2e* đã ingest (basic: chương 3-5; advanced: cả sách + bảng/biểu đồ/hình do vision mô tả).
2. **3 retriever**: dense (embedding), BM25, hybrid RRF (gộp hạng của dense + BM25).
3. **Agent gọi công cụ** `search` / `answer` / `abstain`: tự đặt truy vấn, kiểm tra trích dẫn inline ngay khi trả lời (sai thì thử lại), tuỳ chọn *verifier* kiểm câu trả lời có được bằng chứng ủng hộ.
4. **Benchmark** dev/test: đo retrieval (Hit@k, MRR) và agent (đúng, trích dẫn, abstain, chi phí).""")
    table([{"bộ": n, "dev": sum(q["split"] == "dev" for q in qs), "test": sum(q["split"] == "test" for q in qs),
            "trả lời được": sum(q["answerable"] for q in qs), "không có đáp án": sum(not q["answerable"] for q in qs),
            "chunk": len(load_json(cp))}
           for n, qs, cp in (("Basic (Day 6)", bq, BASIC_CHUNKS), ("Advanced (cả sách)", aq, ADV_CHUNKS))])
    for name in ("agent_hybrid_score_filter", "advanced_gpt41"):
        st.markdown(f"**Cấu hình chọn: `configs/{name}.json`**")
        cfg = load_json(str(HERE / "configs" / f"{name}.json"))
        table([{"tham số": k, "giá trị": v} for k, v in cfg.items()])

# ---------------------------------------------------------------- 2
with tabs[1]:
    def retr_rows(root):
        rows = []
        for d in runs(root):
            s = load_json(str(d / "summary.json"))
            rows.append({"phương pháp": d.name, "split": s["split"], "Hit@1": rate(s["Hit@1"]), "Hit@3": rate(s["Hit@3"]),
                         "Hit@10": rate(s["Hit@10"]), "MRR@10": f"{s['MRR@10']['rate']:.3f}", "n/N (Hit@10)": f"{s['Hit@10']['n']}/{s['Hit@10']['N']}"})
        return rows
    allr = retr_rows(R / "retrieval")
    for sp in ("dev", "test"):
        st.markdown(f"**Split {sp}**")
        table([r for r in allr if r["split"] == sp])
    st.divider()
    sp = st.radio("Split", ["dev", "test"], horizontal=True, key="rsp")
    dirs = [d for d in runs(R / "retrieval") if d.name.endswith(f"-{sp}")]
    tr = {d.name: {t["id"]: t for t in load_json(str(d / "traces.json"))} for d in dirs}
    qs = {q["id"]: q for q in load_json(str(config.QUESTIONS_PATH)) if q["split"] == sp}
    qid = st.selectbox("Câu hỏi", [i for i in qs if all(i in t for t in tr.values())], key=f"rq-{sp}",
                       format_func=lambda i: f"{i}: {qs[i]['question']}")
    q, ch = qs[qid], chunk_map(BASIC_CHUNKS)
    st.write(f"**Đáp án:** {q['expected_answer']} · trang gold: {sorted({g['page'] for g in q['gold']})} · lọc: {q['filter']}")
    for col, (name, t) in zip(st.columns(len(tr)), tr.items()):
        with col:
            st.subheader(name)
            for h in t[qid]["top"][:5]:
                c = ch[h["id"]]
                chunk_card(c, f"**#{h['rank']} {'✓' if h['hit'] else '✗'}** · {h['score']:.3f} · tr. {h['page']}", f"r-{name}-{h['id']}", limit=200, q=q)

# ---------------------------------------------------------------- 3
with tabs[2]:
    rows = []
    for d in runs(R / "agent"):
        s = load_json(str(d / "summary.json"))
        sc = s["status_counts"]
        rows.append({"cấu hình": s["name"], "split": s["split"], "CER": rate(s["context_evidence_rate"]), "tool use": rate(s["tool_use"]),
                     "answered": sc.get("answered", 0), "abstain": sc.get("abstain", 0), "error": sc.get("error", 0),
                     "chi phí $": f"{s['cost_usd']:.4f}", "latency TB (s)": f"{s['mean_latency_s']:.2f}"})
    st.markdown("**Tất cả cấu hình** (CER = tỉ lệ câu trả lời mà context chứa bằng chứng)")
    table(rows)
    st.markdown("**agent_metrics (2 lượt test)**")
    ms = {n: maybe(R / "agent" / f"{n}-test" / "agent_metrics.json") for n in ("agent_dense_baseline", "agent_hybrid_score_filter")}
    ms = {n: m for n, m in ms.items() if m}
    if ms:
        keys = list(next(iter(ms.values())))
        table([{"chỉ số": k, **{n: json.dumps(m.get(k), ensure_ascii=False) for n, m in ms.items()}} for k in keys])
    st.divider()
    c1, c2 = st.columns(2)
    sp = c1.radio("Split", ["dev", "test"], horizontal=True, key="asp")
    dirs = [d for d in runs(R / "agent") if d.name.endswith(f"-{sp}")]
    d = c2.selectbox("Cấu hình", dirs, format_func=lambda d: d.name, key=f"acfg-{sp}")
    if d:
        traces = {t["query_id"]: t for t in load_json(str(d / "traces.json"))}
        qm = {q["id"]: q for q in load_json(str(config.QUESTIONS_PATH))}
        rev = {r["query_id"]: r for r in maybe(d / "reviews.json") or []}
        qid = st.selectbox("Câu hỏi", list(traces), key=f"aq-{d.name}", format_func=lambda i: f"{i}: {traces[i]['question']}")
        st.write(f"**Câu hỏi:** {traces[qid]['question']}")
        st.write(f"**Đáp án mong đợi:** {qm[qid]['expected_answer'] if qm[qid]['answerable'] else '(không có trong sách, phải abstain)'}")
        trace_view(traces[qid], chunk_map(BASIC_CHUNKS), f"a-{d.name}-{qid}", review=rev.get(qid), q=qm[qid])

# ---------------------------------------------------------------- 4
with tabs[3]:
    corpus = st.radio("Kho", ["basic", "advanced"], horizontal=True, key="corp",
                      format_func=lambda c: "Basic (kho Day 6)" if c == "basic" else "Advanced (cả sách)")
    cfgs = sorted(p.stem for p in (HERE / "configs").glob("*.json") if p.stem.startswith("advanced_") == (corpus == "advanced"))
    cname = st.selectbox("Cấu hình", cfgs, key=f"cfg-{corpus}")
    cfg = load_json(str(HERE / "configs" / f"{cname}.json"))
    st.caption(json.dumps(cfg))
    question = st.text_area("Câu hỏi", key="live_q")
    chap = st.selectbox("Lọc chương (chỉ kho basic, cần cấu hình use_filter)", ["Không lọc", 3, 4, 5], key="live_ch") if corpus == "basic" else "Không lọc"
    st.caption("⚠ Nút dưới gọi OpenAI thật và tốn tiền (vài phần nghìn USD với gpt-4o-mini, nhiều hơn với gpt-4.1).")
    if st.button("Chạy agent"):
        if not question.strip():
            st.error("Nhập câu hỏi trước.")
        else:
            try:
                item = {"question": question, "filter": None if chap == "Không lọc" else {"chapter": chap}}
                with st.spinner("Agent đang chạy..."):
                    res = agent.run(item, cfg, client=config.openai_client(), retriever=get_retriever(corpus, cfg["method"]))
                st.session_state.live = {"res": res, "corpus": corpus}
            except Exception as e:  # thiếu key / lỗi Chroma: báo lỗi, không giả kết quả
                st.session_state.pop("live", None)
                st.error(f"Lỗi: {type(e).__name__}: {e}")
    live = st.session_state.get("live")
    if live:
        trace_view(live["res"], chunk_map(BASIC_CHUNKS if live["corpus"] == "basic" else ADV_CHUNKS), "live")

# ---------------------------------------------------------------- 5
with tabs[4]:
    stt = load_json(str(config.ADV_DATA / "index_stats.json"))
    st.markdown("**Chỉ mục cả sách**")
    kinds = ", ".join(f"{k}: {v}" for k, v in stt["chunks_by_kind"].items())
    table([{"trang": stt["pages_total"], "trang có chữ": stt["pages_with_text"], "trang có hình/bảng": stt["pages_with_visual_items"],
            "chunk": stt["chunks_total"], "theo loại": kinds, "chi phí embed $": f"{stt['embedding_cost_usd']:.3f}",
            "chi phí vision $": f"{stt['vision_cost_usd']:.2f}"}])
    test_l = load_json(str(ADV_R / "test_labels.json"))
    groups = sorted({x["group"] for x in test_l})
    st.markdown("**Test: pass theo nhóm** (advanced_gpt41)")
    table([{"nhóm": g, "pass/tổng": f"{sum(x['label'] == 'pass' for x in test_l if x['group'] == g)}/{sum(x['group'] == g for x in test_l)}"} for g in groups]
          + [{"nhóm": "**Tổng**", "pass/tổng": f"{sum(x['label'] == 'pass' for x in test_l)}/{len(test_l)}"}])
    dev_l = load_json(str(ADV_R / "dev_labels.json"))
    st.markdown("**Dev: pass theo cấu hình**")
    table([{"cấu hình": k, "pass/N": f"{v['pass']}/{v['N']}", "file": v["config"]} for k, v in dev_l["runs"].items()])
    st.markdown("**Retrieval nâng cao**")
    table([{"phương pháp": d.name, **{k: rate(s[k]) for k in ("Hit@1", "Hit@3", "Hit@10")}, "MRR@10": f"{s['MRR@10']['rate']:.3f}"}
           for d in runs(ADV_R / "retrieval") for s in [load_json(str(d / "summary.json"))]])
    st.divider()
    c1, c2 = st.columns(2)
    sp = c1.radio("Split", ["test", "dev"], horizontal=True, key="nsp")
    dirs = [d for d in runs(ADV_R / "agent") if d.name.endswith(f"-{sp}")]
    d = c2.selectbox("Cấu hình", dirs, format_func=lambda d: d.name, key=f"ncfg-{sp}")
    if d:
        traces = {t["query_id"]: t for t in load_json(str(d / "traces.json"))}
        if sp == "test":
            labels = {x["query_id"]: x for x in test_l}
        else:
            labels = {x["query_id"]: x for x in dev_l["runs"].get(d.name[:-4], {}).get("labels", [])}
        aqm = {q["id"]: q for q in load_json(str(config.ADV_QUESTIONS_PATH))}
        qid = st.selectbox("Câu hỏi", list(traces), key=f"nq-{d.name}",
                           format_func=lambda i: f"{i} [{aqm[i].get('group', '')}] {labels.get(i, {}).get('label', '?')}: {traces[i]['question']}")
        q = aqm[qid]
        st.write(f"**Câu hỏi:** {q['question']} · nhóm: {q.get('group')}")
        st.write(f"**Đáp án mong đợi:** {q['expected_answer'] if q['answerable'] else '(không có trong sách: ' + q.get('absence_reason', '') + ')'}")
        if q.get("gold"):
            st.caption("Gold: " + "; ".join(f"tr.{g['page']} ({g.get('region')}, {g.get('evidence_type')})" for g in q["gold"]))
        if qid not in labels:
            st.info("Chưa có nhãn chấm cho câu này.")
        trace_view(traces[qid], chunk_map(ADV_CHUNKS), f"n-{d.name}-{qid}", label=labels.get(qid), q=q)
