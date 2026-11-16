# Day 7 - Agentic RAG (cách chạy)

Chạy từ thư mục gốc repo, bằng Python của Day 6 (có openai, chromadb, python-dotenv).
Khóa OpenAI được đọc từ `<DAY6_DIR>/.env`; biến môi trường `DAY6_DIR` mặc định là `../K5-L3-Day06-DataIngestionEngineering/starter`.

```bash
export PYTHONIOENCODING=utf-8
PY=E:/K5-L3-Day06-DataIngestionEngineering/starter/.venv/Scripts/python

# 1. Xuất chunks + câu hỏi sang data/day7/
$PY -m src.day7.export_day6

# 2. Retrieval (kết quả: results/retrieval/NAME-SPLIT/)
$PY -m src.day7.run_retrieval --method bm25   --split dev --name bm25
$PY -m src.day7.run_retrieval --method dense  --split dev --name dense
$PY -m src.day7.run_retrieval --method hybrid --split dev --name hybrid [--use-filter]

# 3. Agent (kết quả: results/agent/NAME-SPLIT/)
$PY -m src.day7.run_agent_bench --config configs/agent_dense_baseline.json --split dev [--limit 5] [--ids q01,q21]

# 4. Test (offline, không gọi mạng)
$PY -m unittest discover -s tests -v
```

Cấu hình agent nằm trong `configs/*.json`: `method`, `top_k`, `max_searches`, `budget_chars`, `use_filter`, `context_policy` (`first` | `score`).
`cases.json` trong thư mục kết quả agent dùng được với `starter/agent_metrics.py`.
