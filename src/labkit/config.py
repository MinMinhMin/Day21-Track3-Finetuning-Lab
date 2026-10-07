"""Tier + model resolution for Lab 21.

One source of truth for "which model on which GPU". Everything else imports from here
so a student changes `COMPUTE_TIER` in `.env` and the whole lab follows.

Design note (deck §13): Unsloth's own Qwen3.5 guidance is *do not use QLoRA on this
generation* — quantization error is higher than normal. So the default path here is
**bf16 LoRA**, and 4-bit is something the student *measures* in NB4 rather than
assumes. That is the opposite of the 2024-era lab this replaces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Tier:
    name: str
    model_id: str
    vram_gb_bf16_lora: float
    max_length: int
    per_device_batch: int
    grad_accum: int
    notes: str

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum


# VRAM figures are the vendor's published bf16-LoRA minimums for the Qwen3.5 family
# (0.8B→3GB, 2B→5GB, 4B→10GB, 9B→22GB, 27B→56GB). Keep them honest: if you change the
# model, change the number, and re-measure with `torch.cuda.max_memory_allocated()`.
TIERS: dict[str, Tier] = {
    "CPU": Tier(
        name="CPU",
        model_id="Qwen/Qwen3.5-0.8B",
        vram_gb_bf16_lora=0.0,
        max_length=512,
        per_device_batch=1,
        grad_accum=8,
        notes="No GPU. NB1 (data+mask) and the eval harness run here; training does not.",
    ),
    "LAPTOP": Tier(
        name="LAPTOP",
        model_id="Qwen/Qwen3.5-2B",
        vram_gb_bf16_lora=5.0,
        max_length=1024,
        per_device_batch=1,
        grad_accum=8,
        notes="8-12 GB laptop GPU (RTX 3060/4060).",
    ),
    "T4": Tier(
        name="T4",
        model_id="unsloth/Qwen3.5-4B",
        vram_gb_bf16_lora=10.0,
        max_length=1024,
        per_device_batch=1,
        grad_accum=16,
        notes="Free Colab T4 (16 GB) — the default path for this lab.",
    ),
    "BIGGPU": Tier(
        name="BIGGPU",
        model_id="Qwen/Qwen3.5-9B",
        vram_gb_bf16_lora=22.0,
        max_length=2048,
        per_device_batch=2,
        grad_accum=8,
        notes="L4 22.5 GB / A100 40 GB / RTX 3090-4090.",
    ),
}

DEFAULT_TIER = "T4"


def get_tier(name: str | None = None) -> Tier:
    """Resolve the active tier from an argument, then $COMPUTE_TIER, then the default.

    $BASE_MODEL (any Hugging Face model id) replaces the tier's default model and keeps
    the tier's hardware settings. Students are free to choose their own base model; every
    stage reads it from here, so baselines and fine-tune always use the same one.
    """
    key = (name or os.environ.get("COMPUTE_TIER") or DEFAULT_TIER).upper()
    if key not in TIERS:
        raise ValueError(
            f"Unknown COMPUTE_TIER={key!r}. Pick one of: {', '.join(TIERS)}"
        )
    tier = TIERS[key]
    override = os.environ.get("BASE_MODEL", "").strip()
    return replace(tier, model_id=override) if override else tier


# --- Training configuration -------------------------------------------------
# The deck's §11 result: the learning rate should be set on the *scale* of ~10x the
# full-fine-tune LR, and that scale matters more than the exact value. A full FT of a
# 4B model sits near 1e-5, so LoRA lands near 1e-4.
FULL_FT_LR = 1e-5
LORA_LR_MULTIPLIER = 10.0
LORA_LR = FULL_FT_LR * LORA_LR_MULTIPLIER          # 1e-4

# §11.4: LoRA tolerates large batches worse than full FT, and raising rank does not
# fix it. Keep the effective batch under 32.
MAX_EFFECTIVE_BATCH = 32


@dataclass(frozen=True)
class LoraSpec:
    """A named LoRA configuration. NB3 trains `correct`; NB4 trains the rest as contrasts."""
    key: str
    r: int | None          # None => computed to match `correct`'s parameter budget
    alpha: int | None      # None => set to 2*r once r is known
    target: str               # "text-linear" | "attn-only"
    lr: float
    load_in_4bit: bool
    label: str
    teaches: str

    @property
    def alpha_over_r(self) -> float | None:
        if self.r is None or self.alpha is None:
            return None
        return self.alpha / self.r

    def resolved(self, r: int) -> "LoraSpec":
        """Fill in a computed rank (alpha follows the deck's 2r invariant, §10.3)."""
        from dataclasses import replace
        return replace(self, r=r, alpha=2 * r)


SPECS: dict[str, LoraSpec] = {
    "correct": LoraSpec(
        key="correct", r=16, alpha=32, target="text-linear", lr=LORA_LR,
        load_in_4bit=False,
        label="all-linear · r=16 · LR 10x · 16-bit",
        teaches="The deck's low-regret configuration (§11).",
    ),
    # r=None => resolved at runtime by modeling.matched_rank() so this run sits on the
    # SAME trainable-parameter budget as `correct`. On Qwen3.5-4B that lands near r=90.
    # Hardcoding a rank here would compare budgets instead of placements.
    "attn_only": LoraSpec(
        key="attn_only", r=None, alpha=None, target="attn-only", lr=LORA_LR,
        load_in_4bit=False,
        label="q,v only · r=matched · LR 10x · 16-bit",
        teaches="Mistake #1 (§11.2): attention-only placement, rank raised to *match "
                "parameter count*. If rank were the lever, this would win.",
    ),
    "wrong_lr": LoraSpec(
        key="wrong_lr", r=16, alpha=32, target="text-linear", lr=FULL_FT_LR,
        load_in_4bit=False,
        label="all-linear · r=16 · LR 1x (full-FT scale) · 16-bit",
        teaches="Mistake #2 (§11.3): a full-fine-tune learning rate applied to LoRA.",
    ),
    "qlora": LoraSpec(
        key="qlora", r=16, alpha=32, target="text-linear", lr=LORA_LR,
        load_in_4bit=True,
        label="all-linear · r=16 · LR 10x · 4-bit QLoRA",
        teaches="The vendor says do NOT use QLoRA on Qwen3.5 (§13). Measure the cost "
                "yourself instead of taking either side on faith.",
    ),
}


# --- prompts -----------------------------------------------------------------------
# These live here, not in generate.py, because BOTH training and evaluation need them
# and holding two copies is how F-31 happened: the lab trained on one prompt shape and
# scored on another, and every adapter came out at target=0.000.
NAIVE_PROMPT = "Phân loại ticket sau."

OPTIMIZED_PROMPT = """Bạn là hệ thống phân loại ticket CSKH. Trả về DUY NHẤT một object JSON, không kèm giải thích, không kèm markdown fence.

Schema bắt buộc — đúng 4 khóa:
{"intent": ..., "urgency": ..., "product": ..., "sentiment": ...}

intent    ∈ doi_tra | van_chuyen | hoan_tien | san_pham_loi | hoi_thong_tin
urgency   ∈ cao | trung_binh | thap
sentiment ∈ tieu_cuc | trung_tinh | tich_cuc
product   = tên sản phẩm xuất hiện nguyên văn trong ticket

Ví dụ:
Ticket: "Shop ơi, mình đặt bàn phím cơ mã đơn DH123456. Giao hàng chậm. Đã 3 ngày rồi. Nhờ shop kiểm tra."
JSON: {"intent": "van_chuyen", "urgency": "trung_binh", "product": "bàn phím cơ", "sentiment": "trung_tinh"}"""


CONTRAST_KEYS = ["attn_only", "wrong_lr", "qlora"]

EPOCHS_DEFAULT = 2.0


def training_epochs(override: float | None = None) -> float:
    """The epoch budget -- ONE number, shared by NB3's baseline and NB4's contrasts.

    NB4's contrasts must run the SAME number of optimizer steps as NB3's `correct` run,
    otherwise the autopsy measures step budget instead of configuration. Both notebooks
    turn this into a step count with `train.planned_steps(n_examples, tier, epochs)`.

    Why a function and not two constants. This started as `CONTRAST_MAX_STEPS = 60`,
    calibrated against a "~10 minutes" estimate; measured on a free-Colab T4 at 48.5
    s/step that is 48 minutes per contrast, and 2x the 30 steps NB3 actually runs -- so
    every contrast was trained twice as long as the baseline it is compared against.
    The first fix made both notebooks derive the count, but left NB3 reading $EPOCHS
    while NB4 read a frozen constant: setting `EPOCHS=1` re-opened the same divergence
    through the front door. One reader, one env var, no way to set half of it.

    `EPOCHS=1` is the supported lever when you are time-boxed -- it halves NB3 *and*
    NB4. Submit with the default, and `scripts/verify.py` cross-checks the step budget
    recorded in `results/runs.csv` so the fairness of the comparison is a checked fact
    rather than a promise.
    """
    if override is not None:
        return float(override)
    raw = os.environ.get("EPOCHS", "").strip()
    return float(raw) if raw else EPOCHS_DEFAULT
