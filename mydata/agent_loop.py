"""Vòng lặp agent thay thế run_agent của starter, giữ nguyên giao diện/trace.

Khác biệt (ghi vào báo cáo, GUIDE cho phép 'dùng, sửa hoặc thay khung agent'):
- Chọn context 'mới-ưu-tiên': kết quả search MỚI đứng trước, chunk cũ sau.
  run_agent starter gộp candidates theo thứ tự cũ-trước nên khi chunk đầu
  chiếm gần hết budget, các lượt search sau không thêm được evidence mới
  (đã quan sát trong trace dev: context_ids giữ nguyên qua 4 lượt).
- Dedup chunk theo ID giữa các lượt (giữ lại chunk cũ làm phương án bổ sung).
Dùng lại pack_context + logic citation của starter (không đổi định nghĩa).
"""
from agentic import pack_context


def run_agent_v2(question: str, controller, retriever, *,
                 filters=None, max_searches: int = 4, top_k: int = 3,
                 budget: int = 800, context_order: str = "new") -> dict:
    """context_order: "new" (mới-ưu-tiên, default) hoặc "score" (hợp nhất các lượt,
    sắp theo score giảm dần rồi cắt theo budget). "score" tránh evict chunk có
    evidence cũ khi lượt mới nhiều (quan sát trace v2: q01)."""
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict] = []
    seen: dict[str, dict] = {}
    chunks: dict[str, dict] = {}  # khởi tạo để model trả lời trước khi search không gây UnboundLocalError
    events: list[dict] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain',
                        'answer': 'Không đủ bằng chứng để trả lời.',
                        'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            new = [r for r in results if r['id'] not in seen]
            for r in results:
                if r['id'] in seen:
                    # giữ score cao nhất giữa các lượt để xếp hạng ổn định
                    seen[r['id']]['score'] = max(float(seen[r['id']].get('score', 0)),
                                                 float(r.get('score', 0)))
                else:
                    seen[r['id']] = r
            prev = [c for cid, c in seen.items() if c['id'] not in {n['id'] for n in new}]
            if context_order == "score":
                pool = sorted(seen.values(),
                               key=lambda c: (-float(c.get('score', 0)), c['id']))
            else:
                pool = new + prev
            context, ids = pack_context(pool, budget=budget)
            chunks = {c['id']: c for c in pool if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters,
                           'results': results, 'context': context, 'context_ids': ids})
            history = [{'tool': 'search',
                        'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and
                         all(isinstance(i, str) and i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations,
                        'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain',
                    'answer': 'Không đủ bằng chứng để trả lời.',
                    'citations': [],
                    'events': events + [{'event': 'invalid_or_missing_citations'}]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.',
            'citations': [], 'events': events}
