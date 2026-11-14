"""Dense retrieval trên vector store Day 6 (Chroma day6_B, embeddinggemma-2 local).

Query embedding dùng cùng model + normalize_embeddings=True với index Day 6
(ks: starter Day 6 src/embeddings.py, src/store.py) để đảm bảo tương thích theo GUIDE.
"""
import os
from pathlib import Path

DAY6_CHROMA = Path(os.getenv(
    "DAY6_CHROMA_PATH",
    r"C:\Users\chise\Desktop\K5-L3A-Day06-2A202603623-DataIngestionEngineering\starter\chroma_data",
))
EMBED_MODEL = os.getenv("DAY7_EMBED_MODEL", "google/embeddinggemma-2")
COLLECTION = os.getenv("DAY7_COLLECTION", "day6_B")


class DenseRetriever:
    """Thực hiện Retriever protocol của starter/agentic.py (lười tải model)."""

    def __init__(self) -> None:
        from chromadb import PersistentClient
        from sentence_transformers import SentenceTransformer
        self._client = PersistentClient(path=str(DAY6_CHROMA))
        self._coll = self._client.get_collection(COLLECTION)
        self._model = SentenceTransformer(EMBED_MODEL)  # lấy từ cache, không tải lại
        print(f"DenseRetriever: {COLLECTION} ({self._coll.count()} docs) + {EMBED_MODEL} local")

    def search(self, query: str, *, filters: dict | None, top_k: int) -> list[dict]:
        vec = self._model.encode([query], normalize_embeddings=True).tolist()[0]
        where = None
        if filters:
            where = {k: v for k, v in filters.items()
                     if k in ("chapter", "section", "printed_page", "chapter_title", "section_title")}
        res = self._coll.query(query_embeddings=[vec], where=where, n_results=min(top_k, self._coll.count()))
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            m = meta or {}
            out.append({"id": cid, "text": doc, "source_id": m.get("document_id"),
                        "source": m.get("source"), "page": m.get("page"),
                        "metadata": {k: v for k, v in m.items() if k not in ("document_id", "source", "page")},
                        "score": round(1 - float(dist), 4)})
        return out
