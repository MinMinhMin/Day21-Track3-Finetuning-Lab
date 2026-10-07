# Lab 21 — Rubric & Submission Format

> **Module**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn
> **Thời lượng**: ~2 giờ thực hành (NB1–NB5) + 30 phút viết report
> **Cấu phần**: CP3 · **PLO**: K1, K4

---

## 🎯 Mục tiêu học tập

Sau lab này, học viên có thể:

1. **Chứng minh** loss mask của mình đúng — bằng cách giải mã ngược, không bằng niềm tin
2. **Đóng băng** một mốc đánh giá và đo **ba baseline trước khi** huấn luyện
3. **Cấu hình** LoRA trong vùng không hối tiếc: đủ lớp, đúng thang LR, batch hiệu dụng < 32
4. **Thiết kế một phép so sánh công bằng** — cùng ngân sách tham số, cùng số step, một biến
5. **Phán quyết** bằng cổng hồi quy bốn nhóm, và bảo vệ kết luận kể cả khi nó là "không nên fine-tune"

---

## 📋 Tổng quan

| Bước | Notebook | Thời gian | Output bắt buộc |
|---|---|---|---|
| 1 | `01_data_and_mask` | ~1 ph | `results/mask_proof.json`, `template_check.json`, `token_stats.json` |
| 2 | `02_baselines` | 17–23 ph | `results/baselines_frozen.json` |
| 3 | `03_train_correct` | 15–25 ph | `adapters/correct/`, dòng `correct` trong `results/runs.csv` |
| 4 | `04_misconfig_autopsy` | **45–60 ph** | 3 dòng đối chứng trong `runs.csv` |
| 5 | `05_evaluate_and_verdict` | ~21 ph | `results/verdict.json`, `autopsy.json`, `qualitative.json` |
| 6 | `06_merge_and_serve` *(tuỳ chọn)* | 10 ph | `results/merge_check.json` |
| 7 | Viết report | 30 ph | `submission/REPORT.md` |

> Đo thật trên T4 free 2026-08-20 (`docs/MEASURED-T4-2026-08-20.md`). Là **khoảng**, không
> phải một con số: cùng cấu hình 30 step chạy 1456 s rồi 1021 s trên đúng mã đó. Core
> NB1–NB5 ≈ **100–130 ph**. Hết giờ thì `EPOCHS=1` (giảm nửa NB3 **và** NB4) hoặc
> `EVAL_LIMIT=8`; nộp bài thì để mặc định.

---

## 🏆 Thang điểm (100 + tối đa 15 thưởng)

| Tiêu chí | Điểm |
|---|---|
| **1. Tính đúng đắn của pipeline** | **30** |
| **2. Thiết kế thí nghiệm công bằng** | **25** |
| **3. Chất lượng đánh giá & phán quyết** | **25** |
| **4. Chất lượng report** | **20** |
| 🎁 Thưởng | **+15** |

### 1. Tính đúng đắn của pipeline — 30 điểm

| | Điểm | Yêu cầu |
|---|---|---|
| 1.1 | 10 | `mask_proof.json` có **cả hai** assert xanh: câu trả lời nằm trong loss, câu hỏi **không** nằm trong loss |
| 1.2 | 5 | `template_check.json` tồn tại và bạn **nêu được** template có giữ khối `<think>` hay không |
| 1.3 | 5 | `max_length` đặt theo p95 đo được (không phải số đoán); nếu lệch tier thì có giải thích |
| 1.4 | 10 | `adapters/correct/` train xong và lưu được; `runs.csv` có dòng `correct` với loss + VRAM |

> Mất trắng 1.1 nếu `supervised_fraction ≥ 0.95` — nghĩa là bạn đang tính loss cả trên prompt.

### 2. Thiết kế thí nghiệm công bằng — 25 điểm

| | Điểm | Yêu cầu |
|---|---|---|
| 2.1 | 10 | Run `attn_only` dùng **rank đã khớp ngân sách tham số** (sai lệch < 5% so với `correct`). `make verify` kiểm tra tự động. |
| 2.2 | 5 | Cả **bốn** run (kể cả `correct`) dùng **cùng `max_steps`** — `make verify` đọc `runs.csv` và kiểm tra tự động |
| 2.3 | 5 | Mỗi run chỉ đổi **một** biến; nêu rõ biến đó trong report |
| 2.4 | 5 | Có phân tích *vị trí vs rank*: điều gì thực sự là đòn bẩy, và bằng chứng nào |
| 2.5 | 0 | *(không tính điểm riêng, nhưng chấm sai ở đây kéo tụt 2.4 và mục 3)* Xếp hạng bốn run bằng **điểm target ở NB5 §4**, không bằng `final_loss` của NB4 |

> **Đây là mục dễ mất điểm nhất.** So `q,v @ r=16` với `all-linear @ r=16` là so *ngân
> sách*, không phải so *vị trí* — và không chứng minh được gì. Dùng `matched_rank()`.

### 3. Chất lượng đánh giá & phán quyết — 25 điểm

| | Điểm | Yêu cầu |
|---|---|---|
| 3.1 | 5 | Baseline **(b) đo trước khi train**, và `(b) > (a)` — prompt "tối ưu" phải thật sự tốt hơn |
| 3.2 | 10 | Đủ **bốn nhóm**: target · regression · format · latency |
| 3.3 | 5 | `verdict.json` có phán quyết, và report **diễn giải** nó (PASS hay FAIL đều được điểm) |
| 3.4 | 5 | ≥5 ví dụ định tính, trong đó **≥2 ca fine-tune THUA**. Chỉ chọn ca thắng = cherry-pick, trừ hết mục này |

> Perplexity **được phép** báo cáo thêm. Nó **không được** là bằng chứng duy nhất.

### 4. Chất lượng report — 20 điểm

| | Điểm | Yêu cầu |
|---|---|---|
| 4.1 | 5 | Report **do bạn tự cấu trúc** (mẫu `submission/REPORT.md` chỉ là gợi ý), nhưng phải có: model + dataset đã chọn và **lý do**, bằng chứng mask, mốc đã đóng băng, kết quả so sánh, phán quyết, điều học được |
| 4.2 | 5 | Kết luận ≥150 từ, có lập luận nhân quả — không chỉ liệt kê số |
| 4.3 | 5 | Mọi con số trong report **khớp** với file trong `results/` |
| 4.4 | 5 | "Điều tôi học được" — phản tư cá nhân, cụ thể, không generic |

### 🎁 Thưởng (tối đa +15)

| | Điểm | |
|---|---|---|
| B1 | +3 | **NB6**: merge + assert điểm không tụt + hot-swap ≥2 adapter |
| B2 | +3 | **Dataset miền riêng** ≥200 mẫu chất lượng + `data/CUSTOM_DATASET.md` có mô tả khử nhiễm |
| B3 | +4 | **Reasoning-trace collapse** (deck §17.5): train hai lần với `MASK_MODE=assistant-only` và `response-only` trên base có chế độ thinking; báo cáo `valid_trace_rate` của cả hai |
| B4 | +3 | **Quét rank có kiểm soát**: cố định vị trí = `text-linear`, quét r ∈ {8,16,64}, trả lời *khi nào* rank mới là đòn bẩy |
| B5 | +2 | Push adapter lên **HuggingFace Hub** công khai + link trong report |

---

## 📦 Ba lựa chọn định dạng nộp

### 🥉 Option A — ZIP gọn (mặc định · ~5–15 MB)

```
lab21_<MSSV>/
├── submission/REPORT.md
├── results/              ← TẤT CẢ file json + runs.csv (grader verify bằng cái này)
├── adapters/correct/     ← chỉ adapter chính: adapter_model.safetensors + adapter_config.json
└── notebooks/            ← .py hoặc .ipynb đã clear output
```

### 🥈 Option B — GitHub + HuggingFace Hub (+2 điểm B5)

```
lab21_<MSSV>/
├── submission/REPORT.md  ← có link HF Hub
├── results/
└── LINKS.md              ← URL repo + URL adapter
```

### 🥇 Option C — Code-only (~500 KB)

```
lab21_<MSSV>/
├── submission/REPORT.md  ← đầy đủ số liệu + 5 ví dụ định tính
├── results/
└── requirements.txt      ← pin để tái lập
```

Cả ba option **đều bắt buộc** có `results/` đầy đủ — đó là thứ grader dùng để kiểm tra
chéo các con số trong report.

---

## ✅ Trước khi nộp

```bash
make verify
```

Gatekeeper kiểm tra artefact **và** tính liêm chính của phép so sánh:

| Kiểm tra | Vì sao |
|---|---|
| mask proof xanh | pipeline sai từ gốc thì mọi số sau đều vô nghĩa |
| `attn_only` khớp ngân sách <5% | phép đối chứng phải công bằng |
| checksum tập eval | sửa eval sau khi thấy kết quả = phép so sánh hỏng |
| SHA của prompt (b) | **làm yếu prompt (b) để fine-tune trông thắng là lỗi liêm chính** |
| `(b) > (a)` | prompt "tối ưu" phải thật sự tối ưu |

Muốn đổi corpus hay cải thiện prompt (b)? Được — nhưng **khai báo** trong report.
Làm mạnh (b) lên: hoan nghênh. Làm yếu đi: đó là gian lận.

---

## 💡 Sai lầm thường gặp

| Triệu chứng | Nguyên nhân | Sửa |
|---|---|---|
| Model viết lại câu hỏi | `MASK_MODE=everything` | quay về `assistant-only` (NB1) |
| Loss phẳng từ step 0 | LR thang full-FT | ×10 (xem run `wrong_lr` ở NB4) |
| `format` ~0 nhưng target khá | template/EOS lệch | in chuỗi **sau** `apply_chat_template` |
| `regression` tụt mạnh | quên thảm hoạ | trộn 1–5% dữ liệu phổ thông (deck §6.3) |
| Suy luận biến mất, accuracy vẫn tăng | reasoning-trace collapse | `MASK_MODE=response-only` (deck §17.5) |
| OOM ở run thứ hai | không giải phóng bộ nhớ | `generate.free_memory()` giữa các run |
| `all-linear` ra adapter to bất thường | gắn cả vào vision tower | dùng `resolve_target_modules()` |

---

## 📚 Tham chiếu

* Deck Ngày 21 — §11 (LoRA Without Regret), §17 (dữ liệu & mask), §21 (đánh giá), §23 (merge/serve)
* `RESEARCH-day21-v2-training-stack-2026.md` — nguồn cho mọi con số 2026 trong lab
* LoRA (Hu et al. 2021) · QLoRA (Dettmers et al. 2023) · *LoRA Without Regret* (Thinking Machines 2025)
* TRL docs — `lora_without_regret`
