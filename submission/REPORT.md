# Lab 21 — Evaluation Report

**Họ tên**: <điền>  **MSSV**: <điền>  **Ngày**: <điền>
**Tier**: `<CPU|LAPTOP|T4|BIGGPU>`  **Base model**: `<model id>`  **GPU thực tế**: `<T4 16GB / ...>`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | `<tên + số mẫu>` (mặc định: 250 ticket CSKH → JSON triage) |
| Train / val | `<n>` / `<n>` (seed 42) |
| `max_length` | `<n>` — p95 đo được là `<n>` *(results/token_stats.json)* |
| `MASK_MODE` | `<assistant-only | ...>` |
| Epochs / max_steps | `<n>` |

**Template có giữ khối `<think>` không?** `<có/không>` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào?

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `<0.xx>` |
| Câu trả lời nằm trong loss | `<true>` |
| Câu hỏi KHÔNG nằm trong loss | `<true>` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
<paste>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | | | | |
| (b) base + optimized prompt | | | | |
| (c) LoRA fine-tune | | | | |

**(b) có thật sự mạnh hơn (a) không?** `<có/không>` — nếu không, bạn đã cải thiện (b) thế nào?
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | | | | | | |
| `attn_only` | q,v | *(matched)* | | | | | | |
| `wrong_lr` | text-linear | 16 | | | | | | |
| `qlora` | text-linear | 16 | | | | | | |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `<PASSED | FAILED>`
`target Δ = <+0.xxx>` · `regression Δ = <+0.xxx>` · `valid_trace_rate = <0.xx>`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | | | | | ✅ FT thắng |
| 2 | | | | | ✅ FT thắng |
| 3 | | | | | ❌ **FT thua** |
| 4 | | | | | ❌ **FT thua** |
| 5 | | | | | |

Có mẫu chung nào ở các ca FT thua không?

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

**Ba điều tôi học được** (cụ thể, không generic):
1.
2.
3.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
