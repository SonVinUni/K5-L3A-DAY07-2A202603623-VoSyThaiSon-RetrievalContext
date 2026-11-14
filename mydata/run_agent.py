"""Chạy agent Day 7: LLM (OpenRouter) + retrieval method (dense/hybrid) + Chroma Day 6.

Chạy từ thư mục gốc Day 7 (venv có chromadb + sentence-transformers, vd venv Day 6):
    python mydata/run_agent.py --split dev --method dense --out starter/results/agent-dev
    python mydata/run_agent.py --split dev --method hybrid --top-k 5 --out starter/results/agent-dev-hybrid5

- backend=openrouter: cần OPENROUTER_API_KEY trong .env (gitignore).
  Model: DAY7_AGENT_MODEL trong .env (mặc định openai/gpt-4o-mini).
- method=dense: Chroma day6_B (embeddinggemma-2 local).
- method=hybrid: RRF dúp dense + TF-IDF (phương pháp #3).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "mydata"))

from agent_loop import run_agent_v2
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever
from openrouter_controller import OpenRouterController
from controller import OllamaController


def run_with_retry(question, controller, retriever, filters, top_k, budget,
                   max_searches, context_order="new", retries=2):
    """Retry khi lỗi timeout/connect (RULES: ghi lỗi thật, không bịa kết quả)."""
    last = None
    for attempt in range(retries + 1):
        try:
            return run_agent_v2(question, controller, retriever, filters=filters,
                                 max_searches=max_searches, top_k=top_k, budget=budget,
                                 context_order=context_order), None
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(5)
    return None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="chỉ chạy các question id, cách dấu phẩy")
    ap.add_argument("--max-searches", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--method", default="dense", choices=["dense", "hybrid"])
    ap.add_argument("--context-order", default="new", choices=["new", "score"],
                    help='"new": kết quả mới đứng trước (v2 hiện tại); '
                         '"score": hợp nhất theo score giảm dần rồi cắt theo budget')
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "ollama"])
    args = ap.parse_args()
    out_dir = Path(args.out or f"starter/results/agent-{args.split}-{args.method}")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads((ROOT / "mydata" / "questions.json").read_text(encoding="utf-8"))
    questions = [q for q in questions if q["split"] == args.split]
    if args.only:
        keep = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in keep]
    if not questions:
        raise SystemExit(f"Không có câu split={args.split}")

    controller = OpenRouterController() if args.backend == "openrouter" else OllamaController()
    if args.method == "hybrid":
        chunks = json.loads((ROOT / "mydata" / "chunks.json").read_text(encoding="utf-8"))
        retriever = HybridRetriever(chunks)
    else:
        retriever = DenseRetriever()
    print(f"method={args.method} top_k={args.top_k} budget={args.budget} -> {out_dir}")

    traces, cases = [], []
    for q in questions:
        qid, question = q["id"], q["question"]
        filters = q.get("filter")
        cases.append({"id": qid, "split": args.split,
                      "answerable": q.get("answerable", True),
                      "expected_behavior": q.get("expected_behavior",
                                                 "search_and_answer" if q.get("answerable", True)
                                                 else "search_and_abstain")})
        res, err = run_with_retry(question, controller, retriever, filters,
                                  args.top_k, args.budget, args.max_searches,
                                  context_order=args.context_order)
        if res is None:
            res = {"status": "error", "answer": "", "citations": [],
                   "events": [{"event": "error", "detail": f"{type(err).__name__}: {err}"}]}
        traces.append({"query_id": qid, "status": res["status"],
                       "answer": res.get("answer", ""),
                       "citations": res.get("citations", []),
                       "events": res.get("events", [])})
        print(f"{qid}: {res['status']} | searches={sum(1 for e in res.get('events', []) if e.get('event') == 'search')} | {res.get('answer', '')[:60]!r}")

    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nLưu {out_dir}/traces.json + cases.json ({len(traces)} câu)")


if __name__ == "__main__":
    main()
