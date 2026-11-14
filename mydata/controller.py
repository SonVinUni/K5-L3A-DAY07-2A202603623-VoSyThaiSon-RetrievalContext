"""Controller cho agent Day 7: gọi LLM local qua Ollama (OpenAI-compatible API).

Model thật chạy local (mặc định qwen2.5:7b) - không tốn API, không mock.
Ollama phải chạy: `ollama serve` (mặc định http://localhost:11434).
Corpus tiếng Anh + câu hỏi tiếng Việt -> model dịch/paraphrase query ra tiếng Anh khi search.
"""
import json
import os
import re
import urllib.request

OLLAMA_BASE = os.getenv("DAY7_OLLAMA_BASE", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("DAY7_LLM_MODEL", "qwen2.5:7b")

SYSTEM = """Bạn là trợ lý RAG tra cứu tài liệu "Introduction to Computer Science" (tiếng Anh).
Dùng tool search để tìm bằng chứng trong tài liệu, sau đó trả lời có citation.
Quy tắc:
- Query nên viết bằng TIẾNG ANH (tài liệu tiếng Anh), ngắn, chứa từ khóa chính.
- Chỉ trả lời khi context có bằng chứng. Trả lời bằng tiếng Việt.
- Câu trả lời phải chứa [chunk-id] đúng với các chunk hỗ trợ, và liệt kê vào citations.
- Nếu tìm nhiều lượt mà vẫn không có bằng chứng: trả lời "Không đủ bằng chứng để trả lời." với citations rỗng.
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {"name": "search",
    "description": "Tìm các đoạn (chunk) trong tài liệu bằng embedding. Query nên là tiếng Anh, chứa từ khóa.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query tiếng Anh ngắn"},
            "filters": {"type": "object", "description": "Tùy chọn: chapter/section (string) lọc theo metadata", "additionalProperties": True},
            "top_k": {"type": "integer", "description": "Số chunk lấy về, mặc định 3", "default": 3},
        },
            "required": ["query"],
        },
    },
}


def _chat(messages: list[dict]) -> dict:
    body = json.dumps({"model": LLM_MODEL, "messages": messages,
                       "tools": [SEARCH_TOOL], "temperature": 0.1}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_BASE + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]


class OllamaController:
    """Thực hiện Controller protocol của starter/agentic.py."""

    model_name = LLM_MODEL

    def decide(self, question: str, history: list[dict]) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM + f"\nCâu hỏi người dùng: {question}"},
        ]
        for h in history:
            messages.append({"role": "assistant", "content": None,
                            "tool_calls": [{"function": {"name": "search",
                                                          "arguments": json.dumps({"query": h["queries"][-1] if h["queries"] else ""})}}]})
            messages.append({"role": "tool", "content": h["context"][:4000]})
        out = _chat(messages)
        if out.get("tool_calls"):
            args = json.loads(out["tool_calls"][0]["function"]["arguments"] or "{}")
            q = (args.get("query") or "").strip()
            if not q:
                raise ValueError("model trả search với query rỗng")
            return {"action": "search", "query": q}
        ans = (out.get("content") or "").strip()
        if not ans:
            raise ValueError("model không trả nội dung cuối")
        # trích [chunk-id] từ câu trả lời (ID dạng openstax-ics-web:p250:0)
        citations = sorted(set(re.findall(r"\[([\w\-]+:p\d+:\d+)\]", ans)))
        return {"action": "answer", "answer": ans, "citations": citations}
