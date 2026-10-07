"""Target-module resolution and LoRA parameter accounting.

Two things here are lab-specific and worth reading before you use them.

**1. "all-linear" is a lie on a multimodal base.**
Qwen3.5 ships as `Qwen3_5ForConditionalGeneration`: a text decoder *plus* a vision
tower. PEFT's `target_modules="all-linear"` walks every `nn.Linear` in the model, so
it happily attaches adapters to the vision encoder you are not training on. You get a
bigger adapter, slower steps, and a checkpoint that is wrong to merge. The deck says
"all-linear" (§11.2) because the study behind it used text-only models — on a 2026
multimodal checkpoint you want *all text-decoder linear layers*, which is what
`resolve_target_modules(model, "text-linear")` returns.

**2. Fair contrasts need matched parameter counts, not matched ranks.**
The claim in §11.2 is that attention-only placement loses to full placement *at the
same parameter budget*. Comparing `q,v @ r=16` against `all-linear @ r=16` compares
budgets, not placements, and proves nothing. `matched_rank()` solves for the rank that
puts attention-only on the same budget, so the only variable left is *where*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Vision / multimodal submodules to keep adapters away from when fine-tuning text.
_VISION_HINTS = ("visual", "vision_tower", "vision_model", "image_", "patch_embed", "merger")

# Attention projections, by the names the Qwen3.5 text decoder uses.
_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
_QV_SUFFIXES = ("q_proj", "v_proj")


@dataclass(frozen=True)
class LinearInfo:
    name: str
    in_features: int
    out_features: int

    @property
    def lora_params_at(self) -> int:  # per unit of rank
        return self.in_features + self.out_features


def iter_linear_modules(model) -> list[LinearInfo]:
    """Every nn.Linear in the model, with its shape. Import-light: no torch type import."""
    out: list[LinearInfo] = []
    for name, mod in model.named_modules():
        if mod.__class__.__name__ not in ("Linear", "Linear4bit", "Params4bit", "LoraLinear"):
            # bitsandbytes/peft wrap Linear; match on the attribute shape instead.
            if not (hasattr(mod, "in_features") and hasattr(mod, "out_features")):
                continue
        if not (hasattr(mod, "in_features") and hasattr(mod, "out_features")):
            continue
        out.append(LinearInfo(name, int(mod.in_features), int(mod.out_features)))
    return out


def is_vision(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _VISION_HINTS)


def is_head(name: str) -> bool:
    """The LM head / embeddings are excluded — adapting them is a different decision."""
    tail = name.split(".")[-1]
    return tail in ("lm_head", "embed_tokens", "score", "classifier")


def resolve_target_modules(model, mode: str = "text-linear") -> list[str]:
    """Return the *suffix* names PEFT should target, for the given placement mode.

    mode="text-linear" -> every linear layer of the text decoder (the deck's
                          "all-linear", corrected for the vision tower)
    mode="attn-only"   -> q_proj and v_proj only (the 2024-era default; Mistake #1)
    mode="attn-full"   -> q,k,v,o (attention, but complete)
    """
    suffixes: set[str] = set()
    for lin in iter_linear_modules(model):
        if is_vision(lin.name) or is_head(lin.name):
            continue
        tail = lin.name.split(".")[-1]
        if mode == "attn-only":
            if tail in _QV_SUFFIXES:
                suffixes.add(tail)
        elif mode == "attn-full":
            if tail in _ATTN_SUFFIXES:
                suffixes.add(tail)
        elif mode == "text-linear":
            suffixes.add(tail)
        else:
            raise ValueError(f"unknown placement mode {mode!r}")
    if not suffixes:
        raise RuntimeError(
            f"resolve_target_modules found nothing for mode={mode!r}. "
            "Print `[n for n,_ in model.named_modules()]` and check the layer names."
        )

    # PEFT matches target_modules by *suffix*. If a vision-tower layer happens to end
    # in one of the same names (e.g. a vision block that also calls its projection
    # `q_proj`), returning suffixes would silently adapt the vision tower anyway.
    # Detect that collision and fall back to fully-qualified names.
    collides = any(
        is_vision(lin.name) and lin.name.split(".")[-1] in suffixes
        for lin in iter_linear_modules(model)
    )
    if collides:
        return sorted(
            lin.name
            for lin in iter_linear_modules(model)
            if not is_vision(lin.name)
            and not is_head(lin.name)
            and lin.name.split(".")[-1] in suffixes
        )
    return sorted(suffixes)


def count_lora_params(model, target_suffixes: list[str], r: int) -> int:
    """Analytic trainable-parameter count for a LoRA at rank r on those suffixes.

    LoRA adds B (out x r) and A (r x in) per targeted layer => r * (in + out).
    """
    targets = set(target_suffixes)
    total = 0
    for lin in iter_linear_modules(model):
        if is_vision(lin.name) or is_head(lin.name):
            continue
        if lin.name.split(".")[-1] in targets:
            total += r * lin.lora_params_at
    return total


def matched_rank(model, base_suffixes: list[str], base_r: int, other_suffixes: list[str]) -> int:
    """Rank that puts `other_suffixes` on the same parameter budget as base at base_r.

    Returns at least 1. Rounds to the nearest integer rank — exact matching is
    impossible when the layer shapes differ, so report the achieved counts too.
    """
    budget = count_lora_params(model, base_suffixes, base_r)
    per_rank = count_lora_params(model, other_suffixes, 1)
    if per_rank == 0:
        raise RuntimeError(f"no layers matched {other_suffixes!r}")
    return max(1, round(budget / per_rank))


def describe_placement(model, r: int = 16) -> list[dict]:
    """Table for NB4: placement x rank x parameter count. Print this before training."""
    text = resolve_target_modules(model, "text-linear")
    qv = resolve_target_modules(model, "attn-only")
    qv_r = matched_rank(model, text, r, qv)
    rows = [
        {"placement": "text-linear", "modules": len(text), "r": r,
         "trainable": count_lora_params(model, text, r)},
        {"placement": "attn-only(q,v)", "modules": len(qv), "r": r,
         "trainable": count_lora_params(model, qv, r)},
        {"placement": "attn-only(q,v) matched", "modules": len(qv), "r": qv_r,
         "trainable": count_lora_params(model, qv, qv_r)},
    ]
    return rows


def layer_type_summary(model_config) -> dict:
    """Deck §7.4 made visible: the 2026 bases interleave linear and full attention.

    Qwen3.5's text config carries `layer_types` / `full_attention_interval`. Printing
    this in the lab is the point where the architecture section stops being a slide.
    """
    cfg = getattr(model_config, "text_config", model_config)
    types = getattr(cfg, "layer_types", None)
    summary: dict = {
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "full_attention_interval": getattr(cfg, "full_attention_interval", None),
        "linear_num_key_heads": getattr(cfg, "linear_num_key_heads", None),
    }
    if types:
        counts: dict[str, int] = {}
        for t in types:
            counts[str(t)] = counts.get(str(t), 0) + 1
        summary["layer_types"] = counts
    return summary
