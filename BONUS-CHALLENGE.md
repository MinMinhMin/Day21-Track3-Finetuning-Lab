# Bonus Challenges — Lab 21

> Làm sau khi `make verify` đã xanh. Mỗi thử thách nối thẳng với một mục **mới** của
> deck 2026 (Part A: pretrain → mid-train → post-train → optimizer → kiến trúc).
> Ghi kết quả vào phần Phụ lục của `submission/REPORT.md`.

---

## B1 — Merge & phục vụ nhiều adapter (+3) · deck §23

Chạy `make nb6`. Yêu cầu:
* `results/merge_check.json` cho thấy điểm **sau merge** không tụt quá 0.01
* Hot-swap ≥2 adapter trên **cùng một** base đang nạp

**Câu hỏi:** merge cho overhead suy luận bằng 0, nhưng bạn mất gì? Khi nào *nên* giữ
adapter riêng dù chậm hơn một chút?

---

## B2 — Dataset miền của bạn (+3) · deck §17

≥200 mẫu chất lượng cao, miền bạn thật sự quan tâm. Bắt buộc có `data/CUSTOM_DATASET.md`:
nguồn · cách thu thập · cách khử nhiễm (eval không được xuất hiện trong train) · vì sao
dữ liệu này **mới về phân phối** so với thứ base model đã thấy (deck §3.3).

**Cảnh báo:** 200 mẫu tự làm cẩn thận thường thắng 2.000 mẫu cào tự động. Deck §3.3 giải
thích vì sao: base 2026 đã bão hoà dữ liệu web phổ thông.

---

## B3 — Reasoning-trace collapse (+4) · deck §17.5 ⭐ khó nhất

Phát hiện mới nhất của deck: fine-tune một model biết suy luận bằng dữ liệu hỏi-đáp
thông thường **phá huỷ năng lực suy luận trong khi accuracy vẫn tăng** — không chỉ số
quen thuộc nào báo động.

Tái lập nó:

```bash
MASK_MODE=assistant-only make nb3 && make nb5   # ghi lại valid_trace_rate
MASK_MODE=response-only  make nb3 && make nb5   # ghi lại valid_trace_rate
```

Báo cáo bảng:

| MASK_MODE | target | **valid_trace_rate** | regression |
|---|---|---|---|
| assistant-only | | | |
| response-only | | | |

**Câu hỏi:** `target` có tăng trong khi `valid_trace_rate` giảm không? Nếu chỉ nhìn
`target`, bạn có phát hiện ra vấn đề không? Đây chính là lý do deck §21 nói perplexity —
và cả accuracy — một mình không phải bằng chứng.

> Chiều tác động **phụ thuộc model**: nghiên cứu gốc thấy "khối `<think>` rỗng" huỷ hoại
> Qwen3-8B nhưng lại bảo vệ Llama-R1-8B. Đừng khái quát từ một model.

---

## B4 — Quét rank CÓ kiểm soát (+3) · deck §11

Thí nghiệm trung tâm của lab cũ, làm cho đúng: **cố định** `target_modules="text-linear"`,
chỉ quét `r ∈ {8, 16, 64}`, giữ nguyên LR và số step.

**Câu hỏi:** rank có phải đòn bẩy không? So biên độ thay đổi khi đổi *rank* với biên độ
khi đổi *vị trí* (run `attn_only` ở NB4) và khi đổi *LR* (run `wrong_lr`). Xếp hạng ba
nút vặn đó theo mức ảnh hưởng — kèm số.

Deck §11 nói rank là **năng lực so với lượng thông tin trong dữ liệu**, không phải nút
chỉnh chất lượng. Dữ liệu 250 mẫu của bạn có đủ thông tin để r=64 dùng hết không?

---

## B5 — HuggingFace Hub (+2)

```python
model.push_to_hub("<user>/lab21-qwen35-triage-vi")
```
Link trong report. Adapter công khai = kiểm chứng được, và đẹp trên CV.

---

## B6 — Không tính điểm: optimizer mismatch · deck §7.3

Cho ai muốn chạm vào Part A của deck. Deck §7.3: chuyển sang **Muon** để fine-tune một
model **đã pre-train bằng Adam** làm *giảm* chất lượng — gọi là *lệch optimizer* — và mức
hư hại **tỷ lệ với độ lớn bước cập nhật**, nên **LoRA làm nó sống sót được**.

Thử: chạy lại `correct` với một optimizer họ Muon (`optim=` trong SFTConfig, cần thư viện
ngoài). Đừng bê learning rate từ Adam sang — nó không chuyển được.

**Dự đoán trước khi chạy**, rồi so với kết quả. Sai dự đoán mà giải thích được vì sao
thì giá trị hơn đoán đúng.

---

## B7 — Không tính điểm: MoE route-aware LoRA · deck §7.5

Nếu bạn có GPU đủ lớn cho một base **MoE** (vd. Qwen3.5-35B-A3B): định tuyến expert lệch
nặng, nên gắn LoRA vào *mọi* expert là lãng phí. Đếm số lần định tuyến trên một tập hiệu
chuẩn nhỏ, chỉ gắn adapter vào **top 25% expert được gọi nhiều nhất mỗi lớp**, và so với
LoRA đầy đủ.

Kết quả đã công bố: chênh lệch trong ±1 điểm %, giảm 70–73% tham số huấn luyện. Chọn
expert **ngẫu nhiên** cùng ngân sách thì kém hơn ~2,5 điểm % — tức *tín hiệu định tuyến*
mới là thứ tạo ra kết quả.

**Và đừng huấn luyện lớp router** — nhà cung cấp tắt mặc định vì lý do ổn định.
