# Báo cáo Day 7 — Vo Sy Thai Son

## Dữ liệu và benchmark

- **Nguồn**: OpenStax *Introduction to Computer Science* (phiên bản WEB), giấy phép
  CC BY NC-SA 4.0, dùng chương 5–6 (memory, CPU, OS) ~90 trang in. Chia thành
  534 chunk (~500 ký tự/chunk), ID dạng `openstax-ics-web:p214:3`.
- **Vector store / embedding**: Chroma (PersistentClient) collection `day6_B` tạo ở
  Day 6, model `google/embeddinggemma-2` chạy local qua sentence-transformers,
  normalize vector ở cả hai phía index lẫn query (768 chiều) — query embedding
  tương thích index nên không tạo index mới.
- **Agent model**: `openai/gpt-4o-mini` qua OpenRouter (key nằm trong `.env`,
  không nộp). Benchmark: 25 câu tự xây từ tài liệu, 18 dev / 7 test, trong đó 4 câu
  (na1–na4) chủ ý không có bằng chứng để kiểm tra việc từ chối. Gold/evidence chỉ
  dùng để chấm, không đưa vào agent. Chấm theo BENCHMARK.md: đọc từng citation
  đối chiếu với chunk, không tin ID.
- **Code**: `mydata/` (controller OpenRouter/Ollama, dense + hybrid retriever,
  vòng lặp agent), chạy từ repo gốc bằng lệnh trong mục dưới.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Hit@3 | Answered (16 câu dev) | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, top-3, budget 800 | sàn so sánh | dev | 3/16 | 3/16 | (baseline không trả lời) | ~0,3 ms/câu, 0$ |
| #2 Dense k3 | Chroma day6_B, top-3, context mới-ưu-tiên, budget 1200 | dense thắng lexical | dev | 7/16 | 8/16 | 13/16 | API gpt-4o-mini |
| #3 Hybrid RRF | dense(top-12) + TF-IDF(top-12), RRF k=60, top-3 | lexical vá chỗ dense yếu | dev | 6/16 | 10/16 | 11/16 | như trên |
| Cải thiện 1 (k5) | Dense top-5, context mới-ưu-tiên, budget 1200 | top-5 giúp evidence lọt top dễ hơn | dev | 7/16 | 8/16 | 14/16 | như trên |
| **Đã chọn** | Dense top-5, context mới-ưu-tiên, budget 1200, max 4 lượt search | - | test | 3/5 | 5/5 | 4/5 | 7 câu test ≈ $0,004 (toàn bộ chạy dev+test tuần này $0,071) |
| Baseline | TF-IDF top-3, budget 800 | - | test | 1/5 | 2/5 | - | 0,5 ms/câu, 0$ |

So sánh dev: TF-IDF hit@3 chỉ 3/16 (lexical thuần, câu hỏi tiếng Việt trong khi
corpus tiếng Anh — đúng như GUIDE gợi ý thử nghiệm). Hybrid hit@3 cao nhất (10/16)
nhưng answered thấp nhất (11/16): RRF trộn nhánh lexical kéo các trang "cùng từ
khóa" (p238, p232 và các trang lân cận) lên trên, chunk evidence đúng bị đẩy ra ngoài top-3, model
không cite được → từ chối. Dense k5 kém nhất về hit@1/3 nhưng answered cao nhất
(14/16) vì top-5 rộng nên có dư chunk để model chọn. **Chọn dense k5** vì tiêu chí
công bố trước của mình là: (1) quality of answer trên dev, (2) độ tin cậy abstain
câu thiếu evidence, (3) chi phí API — k5 cân bằng tốt nhất; hybrid hit@3 cao hơn
mà answer ít hơn thì không đáng (trả lời đúng mới là đích, retrieval hay mà
trả lời sai vẫn là 0 điểm).

**Hai lỗi dev đã đọc (yêu cầu GUIDE §4)**:

1. **q04 (SRAM làm cache, evidence p229)** — cả 4 lượt search, top-5 chỉ quay
   quanh p228/p233/p235, p229 chưa bao giờ vào. Nguyên nhân: các trang p228–235
   đều nói về memory nên embedding gần nhau, query "cache memory technology"
   không phân biệt được. Đã thử thêm "chapter 5" vào query (lượt 2) — không thay
   đổi gì. Lỗi này lặp lại ở test (q18, evidence p232).
2. **q16 (OS phổ biến, evidence p250:3)** — model chỉ xoay 2 query
   "popular operating systems" / "mobile operating systems" qua 4 lượt, kết quả
   y nguyên, p250:3 không vào top-5 → hết lượt, từ chối câu có đáp án. Hướng
   thử: prompt buộc query mỗi lượt phải khác + cho dùng filter metadata
   (chưa kịp đo, ghi là hạn chế).

## Agent và đánh giá

- **Tool**: 1 tool `search` (query + filter metadata tùy chọn), controller là
  LLM, max 4 lượt, dừng khi model trả lời hoặc hết lượt. Query do model viết
  (tiếng Anh, vì corpus tiếng Anh — tiếng Việt khớp kém với cả dense lẫn TF-IDF,
  đã thấy ở baseline). Context: budget 1200 ký tự, kết quả mới đứng trước, dedup
  theo ID. Lỗi API được retry 3 lần, lỗi thật ghi vào trace với `status: error`.

| Cấu hình / split | Tool use | Số lượt/trung bình | Grounded (0/1/2) | Citation hỗ trợ | Câu có citation | Abstain đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense k5 / dev | 18/18 | 1,5 | 9×2, 3×1, 4×0 (tr.bình 1,31) | 18/19 | 14/14 | 1/2 | 3/16 |
| **Dense k5 / test** | **7/7** | **1,6** | **4×2, 1×0 (tr.bình 1,6)** | **5/5** | **4/4** | **2/2** | **1/5** |

Trên dev, câu điểm 0 gồm: q02 (trả "policy" thay vì "scheduling" — citation hợp lệ
nhưng sai đáp án), q03, q06 (evidence trượt top-k), q16 (lỗi 2 trên đây). Trên test
chỉ còn q18 (lỗi 1). Điểm 0/2 của dev (na1) được trả lời bừa ở cấu hình k5 dev —
đáng lẽ phải từ chối; lỗi này không lặp lại trên test (na3, na4 đều từ chối đúng).

| Query ID | Loại | Tool call | Trả lời + citation | Nhãn, lý do |
|---|---|---|---|---|
| q20 (test) | Trả lời có bằng chứng | 1 search "pipeline instruction processing steps" | "commit" + p237:2, p238:0 | 2/2 — cả 2 chunk đọc thấy hỗ trợ |
| na4 (test) | Thiếu bằng chứng | 4 search, hết lượt | Từ chối, không citation | Abstain đúng — tài liệu không nói về garbage collection của Python |
| q18 (test) | Lỗi | 4 search | Từ chối dù p232 (vùng có evidence) đã vào top-5 cả 4 lượt | False abstention — p232 có mặt trong results nhưng budget 1200 chỉ giữ 2 chunk (p233, p232:4), chunk p232:3 chứa evidence "programming" trượt khỏi context nên model không cite được; cùng loại lỗi retrieval với q04 dev |

- **Trace, nhãn, lệnh chạy**: `starter/results/agent-dev-dense-k5/` (dev),
  `starter/results/agent-test-dense-k5/` (test), mỗi thư mục có
  `traces.json`, `cases.json`, `reviews.json` (chấm tay, kèm note từng câu),
  `summary.json` (từ `agent_metrics.py`). Chạy:
  `python mydata/run_agent.py --split dev --method dense --top-k 5 --out starter/results/agent-dev-dense-k5`
  (và `--split test --out starter/results/agent-test-dense-k5`),
  rồi chấm:
  `python starter/agent_metrics.py --cases starter/results/agent-dev-dense-k5/cases.json --traces starter/results/agent-dev-dense-k5/traces.json --reviews starter/results/agent-dev-dense-k5/reviews.json --split dev --output starter/results/agent-dev-dense-k5/summary.json`
  (tương tự cho thư mục test, đổi `--split test`).
  Chi phí API (gpt-4o-mini qua OpenRouter, cả dev + test): ~$0,071, xem
  openrouter.ai/activity. Thời gian thực thi từng câu chưa đo, chỉ ghi số lượt
  LLM (dev trung bình 1,5 lượt/câu, test 1,6 lượt/câu).
- **Đánh đổi**: top-5 + budget 1200 làm context rộng hơn, dễ giữ evidence, nhưng
  model thỉnh thoảng chọn citation mảnh OCR dở (p253:8) hoặc "đúng nội dung sai
  câu hỏi" (q02). Từ chối nhầm 1/5 test (q18) là cái giá của việc chưa tách được
  cluster trang memory.

## Thử thách nâng cao

Không thực hiện.
