# Code agent + retrieval Day 7 (Vo Sy Thai Son)

## Chạy gì, chạy thế nào

Yêu cầu: Python 3.11+, venv có `chromadb`, `sentence-transformers` (+ `pillow`,
`torchvision` để tải được model đa phương tiện `embeddinggemma-2`), file `.env` ở
thư mục gốc repo chứa `OPENROUTER_API_KEY` (không nộp, đã gitignore).

Chạy từ thư mục gốc repo:

```bash
# 1. baseline lexical (offline, không tốn API)
python starter/baseline.py --chunks mydata/chunks.json --questions mydata/questions_eval.json \
    --split dev --output starter/results/dev-tfidf
python starter/baseline.py --chunks mydata/chunks.json --questions mydata/questions_eval.json \
    --split test --output starter/results/test-tfidf

# 2. agent — dense top-5 (cấu hình đã chọn)
python mydata/run_agent.py --split dev  --method dense --top-k 5 --out starter/results/agent-dev-dense-k5
python mydata/run_agent.py --split test --method dense --top-k 5 --out starter/results/agent-test-dense-k5

# 3. agent — hybrid RRF (phương pháp #3, context theo score)
python mydata/run_agent.py --split dev --method hybrid --context-order score --out starter/results/agent-dev-hybrid-score

# 4. chấm (reviews.json do người làm bài tự chấm, kèm note từng câu)
python starter/agent_metrics.py --cases starter/results/agent-dev-dense-k5/cases.json \
    --traces starter/results/agent-dev-dense-k5/traces.json \
    --reviews starter/results/agent-dev-dense-k5/reviews.json \
    --split dev --output starter/results/agent-dev-dense-k5/summary.json
# (tương tự cho agent-test-dense-k5, --split test)

# 5. test unit (không cần API, mock controller/retriever)
python -m unittest discover -s tests
```

## 3 phương pháp

| Phương pháp | File | Cách hoạt động |
|---|---|---|
| Lexical TF-IDF | `starter/baseline.py` (sử dụng) | cosine TF-IDF trên `mydata/chunks.json` |
| Dense | `dense_retriever.py` | query embedding bằng `embeddinggemma-2` (local, normalize) → Chroma `day6_B` |
| Hybrid RRF | `hybrid_retriever.py` | dense(top-12) + TF-IDF(top-12), Reciprocal Rank Fusion k=60 |

Controller LLM (OpenRouter gpt-4o-mini, tool `search`): `openrouter_controller.py`,
dự phòng Ollama local: `controller.py`. Vòng lặp agent: `agent_loop.py`.

## Dữ liệu

- `chunks.json` (534 chunks), `questions.json` (25 câu: 18 dev / 7 test, 4 câu
  thiếu evidence): tái tạo được từ Day 6 bằng
  `K5-L3A-Day06-2A202603623-DataIngestionEngineering/starter/export_day7_chunks.py`,
  rồi `python mydata/convert_son.py --chunks ../K5-L3A-Day06-2A202603623-DataIngestionEngineering/starter/day7_raw_chunks.json --questions ../K5-L3A-Day06-2A202603623-DataIngestionEngineering/benchmark/questions.json`,
  cuối cùng `python mydata/fix_evidence.py --chunks mydata/chunks.json --questions mydata/questions.json`
  để evidence khớp verbatim trong chunk. `questions_eval.json` = bỏ 4 câu na*,
  dùng riêng cho baseline (validate() của baseline yêu cầu mọi câu có gold).
- Vector store: Chroma `day6_B` trong repo Day 6 (đường dẫn hard-code trong
  `dense_retriever.py`, đổi được bằng env `DAY6_CHROMA_PATH`).
