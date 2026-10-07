# %% [markdown]
# # NB4 — Giải phẫu cấu hình sai (phần quan trọng nhất của lab)
#
# Lab Day 21 **phiên bản cũ** lấy "quét rank r=8/16/64" làm thí nghiệm trung tâm, gắn
# LoRA vào `q_proj, v_proj`, và chấm bằng perplexity. Deck hiện tại gọi đúng ba thứ đó
# là **Lỗi #1, #2, #3** (§11.2–§11.4).
#
# Notebook này không xoá thí nghiệm cũ — nó **chạy lại thí nghiệm cũ như một đối chứng**,
# để bạn tự tay thấy danh tiếng *"LoRA học kém hơn full fine-tune"* xuất hiện rồi biến mất.
#
# Ba run đối chứng, **cùng số step**, chỉ đổi một biến mỗi lần:
#
# | Run | Đổi gì | Kỳ vọng |
# |---|---|---|
# | `attn_only` | chỉ q,v — **rank nâng lên cho bằng số tham số** | thua `correct` |
# | `wrong_lr` | LR thang full-FT (÷10) | loss gần như phẳng |
# | `qlora` | 4-bit thay bf16 | nhẹ hơn, chất lượng ? |
#
# > **Vì sao phải "bằng số tham số".** So `q,v @ r=16` với `all-linear @ r=16` là so
# > *ngân sách*, không phải so *vị trí* — và không chứng minh được gì. `matched_rank()`
# > giải ra rank đưa attention-only về đúng ngân sách của `correct`.

# %%
import json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import data, generate, modeling, report, train
from labkit.config import CONTRAST_KEYS, SPECS, get_tier, training_epochs

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))

from datasets import Dataset

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
# Same pre-tokenized, NB1-verified mask as NB3 — the contrasts must differ only in the
# variable under test, and that includes using an identical loss mask.
_tok_for_data = None      # filled on first model load below
train_ds = None

# %% [markdown]
# ## 1. Bảng vị trí × rank × số tham số
#
# Đọc bảng này **trước** khi chạy. Nó cho thấy vì sao so cùng-rank là không công bằng.

# %%
model, tok = generate.load_base(TIER)
placement = modeling.describe_placement(model, SPECS["correct"].r)
print(report.markdown_table(placement))
del model
generate.free_memory()

# %% [markdown]
# ## 2. Ba run đối chứng — cùng ngân sách step

# %%
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def run_contrast(key: str) -> dict:
    spec = SPECS[key]
    global train_ds
    model, tok = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
    if train_ds is None:
        train_ds = Dataset.from_list(
            data.to_training_dataset(tok, train_rows, max_length=TIER.max_length,
                                     mask_mode=os.environ.get("MASK_MODE", "assistant-only")))
        print("  train_ds:", train_ds)
    targets = modeling.resolve_target_modules(model, spec.target)

    if spec.r is None:                       # attn_only: solve for the matched rank
        base_targets = modeling.resolve_target_modules(model, "text-linear")
        r = modeling.matched_rank(model, base_targets, SPECS["correct"].r, targets)
        spec = spec.resolved(r)
        print(f"  matched rank for {key}: r={r} (alpha={spec.alpha})")

    trainable = modeling.count_lora_params(model, targets, spec.r)
    # Same step budget as NB3's `correct`, derived from the same recipe rather than
    # hardcoded -- one variable per contrast means the step count is NOT a variable.
    max_steps = train.planned_steps(len(train_ds), TIER, training_epochs())
    want = train.sft_config_kwargs(TIER, spec, str(ROOT / "adapters" / key),
                                   max_steps=max_steps)
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[{key}]")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]")

    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=train_ds, processing_class=tok,
                         peft_config=LoraConfig(**lora_kwargs))
    # Without this the `qlora` run dies at step 0: TRL hands back bf16 LoRA weights and
    # fp16's GradScaler has no BFloat16 kernel. See F-23 / scripts/probe_precision.py.
    fix = train.align_trainable_precision(trainer.model)
    if fix.get("recast"):
        print(f"  precision fix: recast {fix['recast']}/{fix['trainable_tensors']} "
              f"trainable tensors bf16 -> fp32 for the fp16 GradScaler")

    t0 = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - t0

    out = ROOT / "adapters" / key
    trainer.model.save_pretrained(out)

    row = train.summarize_run(spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
    row["final_loss"] = round(res.training_loss, 4)
    row["max_steps"] = max_steps
    row["teaches"] = spec.teaches
    report.append_row(row, results_dir=ROOT / "results")

    del trainer, model
    generate.free_memory()
    return row


# Resumable on purpose. This notebook is the long one -- three full training runs --
# and Colab disconnects, hits its idle timeout, and drops runtimes routinely. Losing 55
# minutes of finished work because the third run crashed is not a lesson about LoRA.
#
# An adapter directory that already exists is treated as done. Set FORCE_RETRAIN=1 to
# retrain everything, and delete `adapters/<key>/` to redo just one.
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "").lower() in {"1", "true", "yes"}
ONLY = [k for k in os.environ.get("ONLY", "").split(",") if k.strip()]

rows = []
for key in (ONLY or CONTRAST_KEYS):
    if key not in SPECS:
        raise SystemExit(f"unknown run {key!r}; expected some of {CONTRAST_KEYS}")
    done = (ROOT / "adapters" / key / "adapter_model.safetensors").exists()
    if done and not FORCE_RETRAIN:
        print(f"skip {key}: adapters/{key}/ already trained "
              f"(FORCE_RETRAIN=1 to redo, or delete the directory)")
        continue
    print("=" * 70)
    print(f"RUN {key}: {SPECS[key].label}")
    print(f"     {SPECS[key].teaches}")
    rows.append(run_contrast(key))

# The table below reads runs.csv, not just this session's rows, so a resumed run still
# prints all four. Rows are appended, so the last one per key is the current one.
if not rows:
    print("nothing retrained this session — reading runs.csv for the table")
_seen = {}
for r in report.read_rows("runs.csv", results_dir=ROOT / "results"):
    if r.get("run"):
        _seen[r["run"]] = r
rows = [_seen[k] for k in ["correct", *CONTRAST_KEYS] if k in _seen] or rows

# %% [markdown]
# ## 3. Bảng đối chứng
#
# Cả bốn run — `correct` ở NB3 và ba run ở đây — chạy **cùng một số optimizer step**
# (`train.planned_steps(...)`, xem `labkit.config.training_epochs`). Nên loss so được
# trực tiếp: khác biệt duy nhất giữa mỗi contrast và `correct` là đúng một biến.
#
# > Trước đây NB4 cố định `max_steps=60` trong khi NB3 chạy 30 step, và phần này bảo bạn
# > tự chạy lại `correct` cho công bằng. Đó là bug: contrast được huấn luyện gấp đôi
# > baseline mà nó bị đem ra so.

# %%
cols = ["run", "label", "r", "trainable_params", "learning_rate", "final_loss",
        "train_seconds", "peak_vram_gb"]
print(report.markdown_table(rows, cols))

# %% [markdown]
# ### ⚠ `final_loss` ở đây là LOSS HUẤN LUYỆN — đừng xếp hạng bằng nó
#
# Cột này rẻ (có sẵn từ lúc train) và **không phải** thang đo dùng để kết luận. Chính
# lab này gọi "chấm bằng chỉ số thay thế thay vì bằng năng lực trên tác vụ" là **Lỗi
# #3** — nếu bạn xếp hạng bốn run bằng `final_loss`, bạn đang mắc đúng lỗi đó.
#
# Với 225 mẫu và 30 step, một adapter r=283 có thể ép loss huấn luyện **thấp hơn**
# `correct` mà vẫn tệ hơn trên tập target. Loss thấp có thể chỉ là ghi nhớ.
#
# > **NB5 §4 chấm cả ba adapter này trên tập target** — cùng thang đo đã dùng cho
# > `correct`. Đó mới là bảng để trả lời ba câu dưới đây. Nếu thứ tự của hai bảng khác
# > nhau, hãy nói thẳng điều đó trong REPORT.md: bạn vừa đo được lý do lab cũ kết luận sai.
#
# **Về `grad_norm: nan` ở dòng log đầu tiên:** đó là `GradScaler` của fp16 đang dò thang
# — vài step đầu tràn số và bị bỏ qua, đúng theo thiết kế. Bình thường. Cái *không*
# bình thường là `nan` kéo dài suốt run: khi đó run đã chết và loss cuối vô nghĩa.

# %% [markdown]
# ## 4. Câu hỏi phải trả lời trong REPORT.md
#
# Trả lời bằng bảng **NB5 §4** (điểm target), rồi đối chiếu với `final_loss` ở trên.
#
# 1. `attn_only` có **cùng số tham số huấn luyện** với `correct`. Trên tập target nó
#    thắng, thua, hay hoà? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?
# 2. `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau bao nhiêu? Nếu chỉ nhìn
#    loss mà không biết LR, bạn sẽ kết luận gì — và kết luận đó có đúng không?
# 3. `qlora` tiết kiệm bao nhiêu VRAM, và **trả giá bằng gì**? Nhà cung cấp khuyến nghị
#    *không* dùng QLoRA cho dòng model này (deck §13) — số đo của bạn có ủng hộ điều đó không?
