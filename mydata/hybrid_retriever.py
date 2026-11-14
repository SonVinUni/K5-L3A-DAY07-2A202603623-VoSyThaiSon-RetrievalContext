"""Phương pháp retrieval #3: Hybrid - gộp ranking dense (Chroma Day 6) + lexical (TF-IDF)
bằng Reciprocal Rank Fusion (RRF, k=60). Khác dense thuần ở cách tìm/xếp hạng,
vậy được tính là phương pháp riêng theo GUIDE.

Dùng chung Retriever protocol của starter/agentic.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from baseline import TfidfIndex
from dense_retriever import DenseRetriever  # mydata/

RRF_K = 60


class HybridRetriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.dense = DenseRetriever()
        self.lex = TfidfIndex(chunks)
        print(f"HybridRetriever: RRF(k={RRF_K}) = dense(Chroma Day 6) + TF-IDF({len(chunks)} chunks)")

    def search(self, query: str, *, filters: dict | None, top_k: int) -> list[dict]:
        d = self.dense.search(query, filters=filters, top_k=max(top_k * 4, 12))
        l = [{"id": c["id"], "text": c["text"], "source_id": c["source_id"],
              "source": c["source"], "page": c["page"], "metadata": c["metadata"], "score": c["score"]}
             for c in self.lex.search(query, top_k=max(top_k * 4, 12))]
        # Note: TF-IDF không hỗ trợ filter (theo starter README) - filter chỉ áp cho nhánh dense.
        rrf: dict[str, float] = {}
        for rank, r in enumerate(d, 1):
            rrf.setdefault(r["id"], {"chunk": r})
            rrf[r["id"]]["score"] = rrf.get(r["id"], {}).get("score", 0) + 1.0 / (RRF_K + rank)
        for rank, r in enumerate(l, 1):
            rrf.setdefault(r["id"], {"chunk": r})
            rrf[r["id"]]["score"] = rrf.get(r["id"], {}).get("score", 0) + 1.0 / (RRF_K + rank)
        top = sorted(rrf.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]
        return [{**v["chunk"], "score": round(v["score"], 5)} for _, v in top]
