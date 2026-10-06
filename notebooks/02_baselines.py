# %% [markdown]
# # NB2 — Đóng băng eval & đo BA baseline (trước khi train)
#
# > Deck §21: *điểm không nằm ở việc perplexity giảm bao nhiêu, mà ở việc bạn có chứng
# > minh được bản fine-tune thắng baseline (b) hay không.*
#
# **Thứ tự quan trọng.** Đo baseline **trước** khi train, không phải sau. Nếu đo sau,
# bạn sẽ (một cách vô thức) chỉnh prompt baseline cho tới khi fine-tune của mình thắng.
# Đó là lý do notebook này chạy trước NB3.
#
# Ba baseline:
# | | Là gì | Vì sao có mặt |
# |---|---|---|
# | **(a)** | base + prompt ngây thơ | mốc sàn |
# | **(b)** | base + prompt **đã tối ưu** | **mốc thật sự phải vượt** |
# | (c) | bản fine-tune | đo ở NB5 |

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

# EVAL_LIMIT truncates BOTH eval sets — a smoke mode for slow hardware. It is recorded
# in results/ so the grader can see the run was abbreviated; a submitted run must use
# the full sets (leave EVAL_LIMIT unset).
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
if EVAL_LIMIT:
    target, regression = target[:EVAL_LIMIT], regression[:EVAL_LIMIT]
    print(f"⚠ EVAL_LIMIT={EVAL_LIMIT} — SMOKE MODE, not a submittable run")
print(f"target={len(target)}  regression={len(regression)}  tier={TIER.name}")

# %% [markdown]
# ## 1. Nạp base model (chưa fine-tune)

# %%
model, tok = generate.load_base(TIER)
generate.free_memory()

# %% [markdown]
# ## 2. Chấm một baseline trên cả bốn nhóm
#
# Bốn nhóm: **target · regression · format · latency**. Cùng một hàm cho mọi run —
# baseline hay fine-tune — nên các con số so sánh được với nhau.

# %%
def score_run(model, tok, system_prompt, label):
    prompts = [r["input"] for r in target]
    preds, lat = generate.generate_batch(model, tok, prompts, system=system_prompt,
                                         label=f"{label}/target")

    tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target)) / len(target)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)

    rprompts = [r["instruction"] for r in regression]
    rpreds, _ = generate.generate_batch(model, tok, rprompts, system=None, max_new_tokens=96,
                                        label=f"{label}/regression")
    reg = sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rpreds, regression)) / len(regression)

    scores = ev.GroupScores(target=tgt, regression=reg, format=fmt, latency_ms=lat, n=len(target))
    print(f"{label:28s} target={tgt:.3f}  regression={reg:.3f}  format={fmt:.3f}  {lat:.0f}ms")
    return scores, preds, rpreds


scores_a, preds_a, _ = score_run(model, tok, generate.NAIVE_PROMPT, "(a) base + naive prompt")
scores_b, preds_b, rpreds_b = score_run(model, tok, generate.OPTIMIZED_PROMPT, "(b) base + optimized prompt")

# %% [markdown]
# ## 3. Đóng băng
#
# Từ đây **không được sửa** `eval_target.jsonl`, `eval_regression.jsonl`, hay
# `OPTIMIZED_PROMPT` nữa. Sửa bất kỳ thứ nào sau khi biết kết quả fine-tune = tự lừa mình.

# %%
frozen = {
    "tier": TIER.name,
    "model": TIER.model_id,
    "baseline_a": scores_a.as_dict(),
    "baseline_b": scores_b.as_dict(),
    "optimized_prompt_sha": __import__("hashlib").sha256(
        generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16],
    "n_target": len(target),
    "n_regression": len(regression),
    "eval_limit": EVAL_LIMIT or None,
    "smoke_mode": bool(EVAL_LIMIT),
}
report.write_json(frozen, "baselines_frozen.json", results_dir=ROOT / "results")
print(json.dumps(frozen, ensure_ascii=False, indent=2))

# %% [markdown]
# ### Đọc kết quả trước khi đi tiếp
#
# * **(b) đã cao sẵn?** Tốt — bài toán của bạn có thể *không cần* fine-tune. Đó là một
#   kết luận hợp lệ và được chấm điểm đầy đủ (deck §1).
# * **(b) ≈ (a)?** Prompt "tối ưu" của bạn chưa đủ tốt. Cải thiện nó **bây giờ**, trước
#   khi train — nếu không, phần thắng ở NB5 sẽ là ảo.
#
# %% [markdown]
# ## ✅ Checkpoint NB2
# - [ ] `results/baselines_frozen.json` có cả (a) và (b)
# - [ ] Bạn đã đọc và chấp nhận con số (b) — **trước** khi thấy bất kỳ kết quả train nào
