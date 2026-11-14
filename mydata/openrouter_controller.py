"""Controller OpenRouter (OpenAI-compatible) cho agent Day 7.

Yêu cầu:
- File .env ở thư mục gốc Day 7 với OPENROUTER_API_KEY (đã có trong .gitignore).
- Model: env DAY7_AGENT_MODEL, mặc định openai/gpt-4o-mini.
  Đổi model bất kỳ của OpenRouter được; ghi rõ model + chi phí vào báo cáo.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from controller import SYSTEM, SEARCH_TOOL  # dùng chung prompt + schema tool


def _load_env_file() -> None:
    """Đưa các cặp KEY=VALUE trong .env (thư mục gốc repo) vào os.environ (không ghi đè key đã có)."""
    env = Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_env_file()


def _load_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY trống: tạo tại openrouter.ai/keys và đặt vào .env (đã gitignore)")
    return key


class OpenRouterController:
    """Thực hiện Controller protocol của starter/agentic.py."""

    model = os.getenv("DAY7_AGENT_MODEL", "openai/gpt-4o-mini")

    def __init__(self) -> None:
        self._key = _load_key()
        print(f"OpenRouterController: model={self.model}")

    def _chat(self, messages: list[dict]) -> dict:
        body = json.dumps({"model": self.model, "messages": messages,
                           "tools": [SEARCH_TOOL], "temperature": 0.1}).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # in response body để chẩn đoán lỗi (RULES: ghi lỗi, không giấu)
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from e
        msg = data["choices"][0]["message"]
        if msg.get("tool_calls"):
            args = json.loads(msg["tool_calls"][0]["function"]["arguments"] or "{}")
            q = (args.get("query") or "").strip()
            if not q:
                raise ValueError("model trả search với query rỗng")
            return {"action": "search", "query": q}
        ans = (msg.get("content") or "").strip()
        if not ans:
            raise ValueError("model không trả nội dung cuối")
        citations = sorted(set(re.findall(r"\[([\w\-]+:p\d+:\d+)\]", ans)))
        return {"action": "answer", "answer": ans, "citations": citations}

    def decide(self, question: str, history: list[dict]) -> dict:
        """history: mỗi mục = 1 lượt search, có keys queries/context/context_ids (từ agentic.py)."""
        messages = [{"role": "system", "content": SYSTEM + f"\nCâu hỏi người dùng: {question}"}]
        for i, h in enumerate(history):
            q = h["queries"][-1] if h["queries"] else ""
            tc_id = f"call_{i}"
            messages.append({
                "role": "assistant",
                "content": "",  # một số backend (Gemini/Anthropic) chặn content: null
                "tool_calls": [{"id": tc_id, "type": "function",
                                 "function": {"name": "search",
                                              "arguments": json.dumps({"query": q})}}],
            })
            messages.append({"role": "tool", "tool_call_id": tc_id,
                             "content": h["context"][:4000] or "(không có kết quả)"})
        return self._chat(messages)
