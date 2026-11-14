"""Adapter: JSON export Day 6 (của Son) -> schema Day 7 (mydata/chunks.json, mydata/questions.json).

Ngày 6 của mình dùng: chunk_id/document_id/gold_pages (list), convert_day6.py chuẩn
không xử lý được gold_pages list nên tự viết adapter. Chạy từ thư mục gốc Day 7:
    python mydata/convert_son.py --chunks <day7_raw_chunks.json> --questions <day7_raw_questions.json>
"""
import argparse
import json
from pathlib import Path


def convert_chunk(c):
    return {"id": c.get("id", c.get("chunk_id")),
            "source_id": c.get("source_id", c.get("document_id")),
            "source": c.get("source"), "page": c.get("page"),
            "text": c.get("text"), "metadata": c.get("metadata", {})}


def convert_question(q):
    pages = q.get("gold_pages", [q.get("gold_page")])
    if isinstance(pages, int) or pages is None:
        pages = [pages]
    evidence = q.get("evidence_quote")
    gold = []
    if q.get("answerable", True):
        gold = [{"source_id": q.get("gold_source_id", q.get("document_id")),
                 "page": p, "evidence": evidence} for p in pages]
    result = {"id": q["id"], "split": q["split"], "question": q["question"],
              "gold": gold, "expected_answer": q.get("expected_answer", "")}
    if "filter" in q:
        result["filter"] = q["filter"]
    if q.get("answerable") is False:
        result.update({"answerable": False,
                       "expected_behavior": q.get("expected_behavior", "search_and_abstain"),
                       "absence_reason": q.get("absence_reason", "")})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--output", default=Path(__file__).parent, type=Path)
    args = ap.parse_args()
    chunks = [convert_chunk(c) for c in json.loads(args.chunks.read_text(encoding="utf-8"))]
    questions = [convert_question(q) for q in json.loads(args.questions.read_text(encoding="utf-8"))]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("chunks.json", chunks), ("questions.json", questions)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_ans = sum(1 for q in questions if q.get("answerable") is not False)
    n_no = len(questions) - n_ans
    print(f"OK: {len(chunks)} chunks, {len(questions)} questions ({n_ans} có đáp án, {n_no} thiếu evidence)")


if __name__ == "__main__":
    main()
