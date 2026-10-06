# %% [markdown]
# # NB5 — Đánh giá bốn nhóm & PHÁN QUYẾT
#
# Đây là notebook cho điểm. Câu hỏi được chấm **không phải** "perplexity giảm bao nhiêu"
# mà là câu của deck §21:
#
# > **Bạn có chứng minh được bản fine-tune thắng baseline (b) — và bạn có phát hiện được
# > nếu nó KHÔNG thắng?**
#
# Bốn nhóm: **target · regression · format · latency**. Một run chỉ "đạt" khi vượt (b) ở
# target **và** không tụt general capability quá ngưỡng (deck §6.3).

# %%
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import evaluate as ev, generate, report
from labkit.config import get_tier

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")

# Must match NB2's slice, or the comparison against the frozen baselines is invalid.
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
if EVAL_LIMIT:
    target, regression = target[:EVAL_LIMIT], regression[:EVAL_LIMIT]

frozen = json.loads((ROOT / "results" / "baselines_frozen.json").read_text(encoding="utf-8"))
base_b = ev.GroupScores(**{k: v for k, v in frozen["baseline_b"].items() if k != "extra"})
base_a = ev.GroupScores(**{k: v for k, v in frozen["baseline_a"].items() if k != "extra"})

# Guard: comparing a 50-item fine-tune score against a 10-item baseline is meaningless.
if frozen.get("n_target") != len(target):
    raise SystemExit(
        f"eval slice mismatch: baselines were frozen on {frozen.get('n_target')} target items, "
        f"this run has {len(target)}. Set EVAL_LIMIT to the same value as NB2 (or unset both)."
    )
print("baseline (b) target =", round(base_b.target, 3), "— đây là mốc phải vượt")

# %% [markdown]
# ## 1. Chấm một adapter

# %%
from peft import PeftModel


def score_adapter(adapter_dir: pathlib.Path, system_prompt: str | None, *,
                  load_in_4bit: bool = False, with_regression: bool = True,
                  label: str = "ft") -> tuple:
    """Score one adapter on the target group (+ format + latency), regression optional.

    `load_in_4bit` must match how the adapter was TRAINED. The `qlora` contrast learned
    against a 4-bit base; scoring it on the fp16 base measures a base/adapter mismatch
    and calls the result "what QLoRA costs you". Take the flag from the run's own spec,
    never from the default.
    """
    model, tok = generate.load_base(TIER, load_in_4bit=load_in_4bit)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    preds, lat = generate.generate_batch(
        model, tok, [r["input"] for r in target], system=system_prompt, label=f"{label}/target")
    tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target)) / len(target)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)

    # The contrasts are scored on target/format only: the graded 4-group verdict is about
    # `correct`, and the regression pass is a second full generation sweep per adapter.
    rpreds, reg = [], 0.0
    if with_regression:
        rpreds, _ = generate.generate_batch(
            model, tok, [r["instruction"] for r in regression], system=None, max_new_tokens=96,
            label=f"{label}/regression")
        reg = sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rpreds, regression)) / len(regression)

    # Deck §17.5 — reasoning-trace collapse. Only meaningful if the base has a thinking
    # mode AND you trained on traces; scored anyway so the number is on the record.
    trace = sum(ev.valid_reasoning_trace(p) for p in preds) / len(preds)

    del model
    generate.free_memory()
    s = ev.GroupScores(target=tgt, regression=reg, format=fmt, latency_ms=lat,
                       n=len(target), extra={"valid_trace_rate": round(trace, 4)})
    return s, preds, rpreds


# The fine-tune is evaluated WITHOUT the long optimized prompt — that is the point of
# fine-tuning: the behaviour moved into the weights, so the prompt can shrink.
scores_ft, preds_ft, rpreds_ft = score_adapter(ROOT / "adapters" / "correct",
                                               generate.NAIVE_PROMPT)
print("fine-tune:", scores_ft.as_dict())

# %% [markdown]
# ## 2. Bảng so sánh ba baseline

# %%
table = ev.comparison_table({
    "(a) base + naive prompt": base_a,
    "(b) base + optimized prompt": base_b,
    "(c) LoRA fine-tune": scores_ft,
})
print(report.markdown_table(table))

# %% [markdown]
# ## 3. Cổng hồi quy — phán quyết

# %%
verdict = ev.regression_gate(scores_ft, base_b)
print("PASSED" if verdict.passed else "FAILED")
for r in verdict.reasons:
    print(" -", r)

report.write_json(
    {"comparison": table, "verdict": verdict.as_dict(),
     "valid_trace_rate": scores_ft.extra.get("valid_trace_rate")},
    "verdict.json", results_dir=ROOT / "results")

# %% [markdown]
# ### Nếu FAILED — đừng sửa eval
#
# Một phán quyết FAILED **được chấm điểm đầy đủ** nếu bạn phân tích đúng. Deck §1: đôi
# khi kết luận đúng là *"bài toán này không cần fine-tune"*. Cái bị trừ điểm là:
# nới ngưỡng, làm yếu prompt (b), hay đổi tập eval sau khi thấy kết quả.
#
# Thứ tự chẩn đoán:
# 1. `format` thấp → template/mask (NB1), không phải LoRA
# 2. `regression` tụt → quên thảm hoạ → thêm 1–5% replay (deck §6.3)
# 3. `target` không nhúc nhích → xem lại LR (NB4 `wrong_lr`) trước khi đụng tới rank
# 4. Cả ba đều ổn nhưng vẫn thua (b) → prompt engineering đã thắng. Đó là một kết quả.

# %% [markdown]
# ## 4. Giải phẫu NB4 — chấm ba cấu hình sai trên CÙNG thang đo
#
# NB4 in ra `final_loss`. Đó là **loss huấn luyện** — và toàn bộ lab này lấy việc "chấm
# bằng chỉ số thay thế thay vì bằng năng lực trên tác vụ" làm **Lỗi #3**. Một adapter
# hoàn toàn có thể ép loss huấn luyện xuống thấp hơn `correct` mà vẫn **tệ hơn** trên
# tập target: 225 mẫu, 30 step, LoRA r=283 — loss thấp có thể chỉ là ghi nhớ.
#
# Nên câu "cấu hình sai có thua không?" phải được trả lời ở đây, bằng **cùng thang đo
# đã dùng cho `correct`**, chứ không phải bằng cột `final_loss` của NB4.
#
# > **Đừng đoán trước kết quả.** `attn_only` có *cùng ngân sách tham số*; trên một tác
# > vụ hẹp như triage JSON, nó có thể hoà — hoặc thắng. Đó vẫn là một kết quả đúng và
# > được chấm điểm đầy đủ. Điều bị trừ điểm là báo cáo một thứ tự mà số đo không ủng hộ.

# %%
from labkit.config import CONTRAST_KEYS, SPECS

autopsy = [{"run": "correct", "target": round(scores_ft.target, 4),
            "format": round(scores_ft.format, 4),
            "latency_ms": round(scores_ft.latency_ms, 1), "n": scores_ft.n}]

for key in CONTRAST_KEYS:
    adir = ROOT / "adapters" / key
    if not adir.exists():
        print(f"skip {key}: {adir} chưa có — chạy NB4 trước")
        continue
    s_k, _, _ = score_adapter(adir, generate.NAIVE_PROMPT,
                              load_in_4bit=SPECS[key].load_in_4bit,
                              with_regression=False, label=key)
    autopsy.append({"run": key, "target": round(s_k.target, 4),
                    "format": round(s_k.format, 4),
                    "latency_ms": round(s_k.latency_ms, 1), "n": s_k.n})
    print(f"{key}: target={s_k.target:.3f}  format={s_k.format:.3f}")

print()
print(report.markdown_table(autopsy, ["run", "target", "format", "latency_ms", "n"]))
report.write_json(autopsy, "autopsy.json", results_dir=ROOT / "results")

# %% [markdown]
# ### Bảng này mới là câu trả lời cho ba câu hỏi ở cuối NB4
#
# Đặt nó cạnh cột `final_loss` của NB4. Nếu thứ tự hai bảng **khác nhau**, bạn vừa tự
# tay đo được lý do lab cũ kết luận sai: nó dừng lại ở chỉ số thay thế.

# %% [markdown]
# ## 5. Định tính — bắt buộc có cả ca THUA
#
# Chọn 5 ví dụ: ≥2 ca fine-tune thắng, **≥2 ca fine-tune thua**. Chỉ chọn ca thắng là
# cherry-pick và bị trừ điểm ở mục Evaluation Quality.

# %%
rows = []
for i, (p, r) in enumerate(zip(preds_ft, target)):
    s_ft = ev.triage_field_accuracy(p, r["label"])
    rows.append({"i": i, "ticket": r["input"][:70], "ft_score": round(s_ft, 2),
                 "ft_pred": p.replace("\n", " ")[:90]})
rows.sort(key=lambda x: x["ft_score"])
print("--- 3 ca TỆ NHẤT (bắt buộc đưa vào report) ---")
print(report.markdown_table(rows[:3], ["i", "ticket", "ft_score", "ft_pred"]))
print("\n--- 3 ca TỐT NHẤT ---")
print(report.markdown_table(rows[-3:], ["i", "ticket", "ft_score", "ft_pred"]))
report.write_json(rows, "qualitative.json", results_dir=ROOT / "results")

# %% [markdown]
# ## ✅ Checkpoint NB5
# - [ ] `results/verdict.json` — có phán quyết pass/fail
# - [ ] `results/autopsy.json` — ba cấu hình sai đã được chấm trên thang đo tác vụ
# - [ ] Bảng ba baseline đã đủ
# - [ ] `results/qualitative.json` — có cả ca thắng lẫn ca thua
