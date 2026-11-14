"""Làm evidence khớp verbatim trong chunk (cùng source + trang).

Day 6 trích evidence_quote từ PDF nên chứa \n, dấu ', và OCR artifact
('pie ces', 'o f', hyphen rớt). baseline.py yêu cầu evidence in chunk['text']
verbatim -> script này thay evidence bằng đoạn TRÙNG GIỮNG (regex cho phép
thoái với whitespace/hyphen giữa các từ) lấy THẲNG TỪ text chunk, nên
luôn nằm trọn trong đúng chunk đúng nguồn/trang.

Chạy (từ thư mục gốc Day 7, idempotent):
    python mydata/fix_evidence.py --chunks mydata/chunks.json --questions mydata/questions.json
"""
import argparse
import json
import re
from pathlib import Path


def tolerant_regex(evidence: str) -> re.Pattern:
    words = [w for w in evidence.split() if w]
    if len(words) < 6:
        return None
    # anchor: 6 từ đầu để không match nhầm chỗ khác trong trang
    prefix = r'[\s\-–—]*'.join(re.escape(w) for w in words[:6])
    return re.compile(prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', required=True, type=Path)
    ap.add_argument('--questions', required=True, type=Path)
    args = ap.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))

    fixed = already = broken = 0
    for q in questions:
        if q.get('answerable') is False:
            continue
        for g in q.get('gold', []):
            ev = g['evidence']
            raw_ok = any(g['evidence'] in c['text'] and c['source_id'] == g['source_id']
                         and c['page'] == g['page'] for c in chunks)
            if raw_ok:
                already += 1
                continue
            rx = tolerant_regex(re.sub(r'\s+', ' ', ev))
            if rx is None:
                g['note'] = 'EVIDENCE_NOT_FOUND'
                broken += 1
                continue
            for c in chunks:
                if c['source_id'] != g['source_id'] or c['page'] != g['page']:
                    continue
                m = rx.search(c['text'])
                if not m:
                    continue
                # lấy cửa sổ từ vị trí match đến dấu câu kết thúc (<= 200 ký tự)
                start = m.start()
                tail = c['text'][start:]
                end = len(tail)
                for i in range(40, min(len(tail), 200)):
                    if tail[i] in '.!?' and i >= 40:
                        end = i + 1
                        break
                new_ev = c['text'][start:start + end].strip()
                assert new_ev in c['text'], f"regression: {q['id']}"
                g['evidence'] = new_ev
                g['note'] = 'verbatim-window-from-chunk'
                fixed += 1
                break
            else:
                g['note'] = 'EVIDENCE_NOT_FOUND'
                broken += 1
    args.questions.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'verbatim sẵn: {already}; đã thay bằng cửa sổ verbatim: {fixed}; lỗi: {broken}')
    for q in questions:
        for g in q.get('gold', []):
            if g.get('note') == 'EVIDENCE_NOT_FOUND':
                print(f"  {q['id']}: {g['evidence'][:60]!r}")


if __name__ == '__main__':
    main()
