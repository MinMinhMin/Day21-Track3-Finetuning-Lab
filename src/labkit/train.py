"""Thin, version-defensive wrapper around TRL's SFTTrainer.

Why a wrapper at all: the lab this replaces shipped a page of monkey-patches for
`tokenizer=` vs `processing_class=`, `evaluation_strategy` vs `eval_strategy`, and a
`packing=False` workaround. Those were all *pre-1.0 TRL fossils*. Rather than ship a
new set of fossils, this module asks the installed TRL what it accepts and drops what
it does not — so the lab keeps running when TRL moves again, and tells you what it
dropped instead of failing at step 0.

The defaults encode the deck:
  * `target_modules` = text-decoder linear layers (§11.2, corrected for the vision tower)
  * `learning_rate`  = ~10x the full-FT scale (§11.3)
  * effective batch  < 32 (§11.4)
  * the loss mask comes from `labkit.data.to_training_dataset()`, NOT from TRL's
    `assistant_only_loss` — on Qwen3.5 (no `{% generation %}` markers) TRL >= 1.10 either
    raises or substitutes its own patched template, so the mask is not NB1's
    (see check_mask_agreement.py)
  * `loss_type="chunked_nll"` — the memory-efficient loss used by the lab. Before
    SFTTrainer construction, `prepare_chunked_loss_forward()` adapts Qwen3.5's
    `functools.partial` forward to the bound-method shape TRL expects.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import types
import warnings

from . import device
from .config import MAX_EFFECTIVE_BATCH, LoraSpec, Tier

WARMUP_FRACTION = 0.1


def planned_steps(n_examples: int, tier: Tier, epochs: float) -> int:
    """Optimizer steps that `epochs` over `n_examples` takes on `tier`.

    NB3 sets an *epoch* budget and lets the Trainer derive the step count; NB4's
    contrasts set `max_steps` directly. The autopsy only means anything if both land on
    the SAME number, so both go through this function rather than one of them hardcoding
    a guess. `n_examples` must be the count AFTER `data.to_training_dataset()`, which
    drops examples with zero supervised tokens.
    """
    per_epoch = math.ceil(n_examples / tier.effective_batch)
    return max(1, math.ceil(per_epoch * epochs))


def _accepted_fields(cls) -> set[str]:
    """Field names `cls` will accept, whether it is a dataclass or a plain __init__."""
    names: set[str] = set()
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    try:
        sig = inspect.signature(cls.__init__)
        names |= {p for p in sig.parameters if p != "self"}
    except (TypeError, ValueError):  # pragma: no cover - builtins
        pass
    return names


def filter_kwargs(cls, desired: dict, *, label: str = "config") -> tuple[dict, list[str]]:
    """Keep only the kwargs `cls` accepts. Returns (kept, dropped_names).

    Dropped keys are reported, never swallowed: if your TRL is too old for
    `padding_free`, you want to *know* that packing is now unsafe, not discover it in
    a loss curve.
    """
    ok = _accepted_fields(cls)
    if not ok:                                    # pragma: no cover - defensive
        return dict(desired), []
    kept = {k: v for k, v in desired.items() if k in ok}
    dropped = sorted(set(desired) - set(kept))
    if dropped:
        warnings.warn(
            f"{label}: installed {cls.__name__} does not accept {dropped}. "
            "They were dropped. Check your TRL/PEFT version against requirements.txt.",
            RuntimeWarning,
            stacklevel=2,
        )
    return kept, dropped


def sft_config_kwargs(
    tier: Tier,
    spec: LoraSpec,
    output_dir: str,
    *,
    max_steps: int | None = None,
    total_steps: int | None = None,
    num_train_epochs: float = 1.0,
    mask_mode: str = "assistant-only",
    seed: int = 42,
    precision: str | None = None,
) -> dict:
    """The SFTConfig we *want*. Pass through `filter_kwargs` before constructing."""
    if tier.effective_batch > MAX_EFFECTIVE_BATCH:
        raise ValueError(
            f"effective batch {tier.effective_batch} exceeds {MAX_EFFECTIVE_BATCH} "
            "(deck §11.4: LoRA tolerates large batches worse than full FT, and raising "
            "rank does not fix it). Lower grad_accum for this tier."
        )
    kw = dict(
        output_dir=output_dir,
        max_length=tier.max_length,               # NOT max_seq_length (renamed in TRL v1)
        per_device_train_batch_size=tier.per_device_batch,
        gradient_accumulation_steps=tier.grad_accum,
        learning_rate=spec.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        packing=False,       # we supply pre-tokenized labels -- see the note below
        # Same NLL math, but chunks the LM-head projection to avoid materializing all
        # sequence×vocabulary logits. Adapt Qwen3.5's partial forward before SFTTrainer.
        loss_type="chunked_nll",
        gradient_checkpointing=True,
    )
    # `warmup_ratio` does not exist any more. transformers v5 / TRL 1.10 expose only
    # `warmup_steps` (measured on Colab 2026-08-20: SFTConfig warm-fields == ['warmup_steps']).
    # Passing the ratio does not raise -- filter_kwargs drops it with a warning and the run
    # silently trains with NO warmup, which is exactly the class of quiet failure this lab
    # is about. Convert the deck's 10% into an absolute step count.
    steps = max_steps if max_steps is not None else total_steps
    if steps:
        kw["warmup_steps"] = max(1, round(WARMUP_FRACTION * steps))

    # Precision follows the DEVICE, not the fashion. A free-Colab T4 is Turing and has
    # no bf16 at all — see labkit/device.py. Setting the wrong one here either errors at
    # trainer construction or silently trains in fp32.
    prec = device.precision(precision)
    kw["bf16"] = prec == "bf16"
    kw["fp16"] = prec == "fp16"

    # `padding_free` is enabled only where it is both safe and useful — see
    # device.supports_padding_free(). On the default T4 it is neither: Turing has no
    # FlashAttention-2, and per_device_batch=1 leaves no padding to remove.
    # It also conflicts with our setup: TRL raises
    #   "When padding_free=True without packing, max_length is not enforced"
    # because we pass pre-tokenized labels (packing off). Our inputs ARE already
    # truncated by build_example(), so when padding-free IS available we hand TRL
    # max_length=None to satisfy that check honestly rather than silencing it.
    kw["padding_free"] = device.supports_padding_free(tier.per_device_batch)
    if kw["padding_free"]:
        kw["max_length"] = None

    # NOTE — deliberately NOT setting `assistant_only_loss`.
    # TRL derives that mask from `{% generation %}` markers in the chat template, and
    # Qwen3.5's template has none. TRL >= 1.10 then raises ValueError (unsloth template)
    # or patches the official template — a mask that also covers the empty <think>
    # block. The raw tokenizer mask is ZERO tokens. See scripts/check_mask_agreement.py.
    # Instead the dataset is pre-tokenized by labkit.data.to_training_dataset(), so the
    # loss covers exactly the mask verified in NB1. That also forces packing off:
    # packing concatenates examples and would invalidate the label alignment.
    if max_steps is not None:
        kw["max_steps"] = max_steps
    return kw


def prepare_chunked_loss_forward(model) -> dict:
    """Make Qwen3.5's partial ``forward`` compatible with TRL's chunked-NLL patch.

    TRL wraps ``model.forward`` as a bound method and reads
    ``original_forward.__func__`` to preserve its signature. Qwen3.5 exposes its
    decorated forward as ``functools.partial`` instead. Bind a small delegating method
    and give it the effective partial signature with a synthetic leading ``self``.
    Calls still go through the original partial, including any arguments it captured.
    Other model implementations are left untouched.
    """
    loss_device_fix = align_chunked_loss_label_devices()
    original_forward = getattr(model, "forward", None)
    if not isinstance(original_forward, functools.partial):
        return {
            "patched": False,
            "reason": "forward is not functools.partial",
            "loss_device_alignment": loss_device_fix,
        }

    effective_signature = inspect.signature(original_forward)
    params = list(effective_signature.parameters.values())
    self_kind = (inspect.Parameter.POSITIONAL_ONLY
                 if params and params[0].kind is inspect.Parameter.POSITIONAL_ONLY
                 else inspect.Parameter.POSITIONAL_OR_KEYWORD)
    unbound_signature = effective_signature.replace(
        parameters=[inspect.Parameter("self", self_kind), *params]
    )

    def _forward_compat(self, *args, **kwargs):
        return original_forward(*args, **kwargs)

    _forward_compat.__name__ = "forward"
    _forward_compat.__qualname__ = f"{type(model).__name__}.forward"
    _forward_compat.__signature__ = unbound_signature
    model.forward = types.MethodType(_forward_compat, model)
    return {
        "patched": True,
        "signature": str(effective_signature),
        "loss_device_alignment": loss_device_fix,
    }


def align_chunked_loss_label_devices() -> dict:
    """Move chunked-NLL labels to ``hidden_states.device`` before TRL indexes them.

    With ``device_map="auto"`` on Kaggle's two T4s, the model can produce its final
    hidden states on ``cuda:1`` while Trainer leaves the labels on the input device.
    TRL 1.10 builds an index tensor from those labels and applies it to hidden states,
    which raises when the two CUDA devices differ. Patch the private helper narrowly:
    only ``labels`` / ``shift_labels`` are copied, and only when their device differs.
    """
    try:
        import importlib
        sft_trainer = importlib.import_module("trl.trainer.sft_trainer")
        original_loss = sft_trainer._chunked_cross_entropy_loss
    except (ImportError, AttributeError) as exc:
        return {"patched": False, "reason": f"TRL chunked loss helper unavailable: {exc}"}

    if getattr(original_loss, "_labkit_align_label_devices", False):
        return {"patched": False, "reason": "already patched"}

    signature = inspect.signature(original_loss)
    if "hidden_states" not in signature.parameters:
        return {
            "patched": False,
            "reason": "TRL chunked loss signature has no hidden_states parameter",
        }

    @functools.wraps(original_loss)
    def _loss_with_aligned_labels(hidden_states, *args, **kwargs):
        bound = signature.bind_partial(hidden_states, *args, **kwargs)
        target_device = hidden_states.device
        for name in ("labels", "shift_labels"):
            labels = bound.arguments.get(name)
            if labels is not None and labels.device != target_device:
                bound.arguments[name] = labels.to(target_device)
        return original_loss(*bound.args, **bound.kwargs)

    _loss_with_aligned_labels._labkit_align_label_devices = True
    sft_trainer._chunked_cross_entropy_loss = _loss_with_aligned_labels
    return {"patched": True, "target_device": "hidden_states.device"}


def install_finite_metrics_guard(trainer) -> None:
    """Abort training at the first logged NaN/Inf loss or gradient norm."""
    from transformers import TrainerCallback

    class _FiniteMetricsGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for name in ("loss", "grad_norm", "eval_loss"):
                value = (logs or {}).get(name)
                if value is None:
                    continue
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError, OverflowError):
                    finite = False
                if not finite:
                    raise FloatingPointError(
                        f"Training stopped: logged {name}={value!r} at step "
                        f"{state.global_step}; no adapter should be saved."
                    )
            return control

    trainer.add_callback(_FiniteMetricsGuard())


def assert_finite_training(model, training_loss: float) -> None:
    """Refuse to save an adapter if the loss or any trainable weight is non-finite."""
    if not math.isfinite(float(training_loss)):
        raise FloatingPointError(f"Training loss is not finite: {training_loss!r}")

    import torch

    for name, param in model.named_parameters():
        if param.requires_grad and not bool(torch.isfinite(param.detach()).all().item()):
            raise FloatingPointError(
                f"Trainable parameter {name!r} contains NaN/Inf; refusing to save adapter."
            )


def align_trainable_precision(model, precision: str | None = None) -> dict:
    """Make the trainable params something fp16's GradScaler can actually unscale.

    Measured on a free-Colab T4 (`scripts/probe_precision.py --trainer qlora`): the
    model handed to SFTTrainer has NO bfloat16 parameters, and the model SFTTrainer
    hands back has 496 of them -- every LoRA weight -- while `fp16=True` and a
    GradScaler is attached. Training then dies at the first optimizer step with

        NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
                             not implemented for 'BFloat16'

    because that CUDA kernel has no BFloat16 overload. On a Turing card there is no
    bf16 hardware at all, so the cast is not just unsupported, it is meaningless.

    This is the exact failure `labkit/device.py` was written about -- "tutorials hardcode
    bf16 because every 2026 tutorial is written on an A100" -- except the hardcoding is
    inside the training library, downstream of both the quantization config and the
    `fp16`/`bf16` flags we set. So it cannot be fixed by configuring TRL; it has to be
    corrected on the model TRL returns.

    fp32 is the target, not fp16: master weights in fp32 with fp16 autocast is the
    standard mixed-precision setup, and it is what `prepare_model_for_kbit_training`
    produces on its own. Call this after constructing the Trainer and before `.train()`
    -- the optimizer is not built until then, so re-typing the parameters is safe.

    Returns a summary of what moved, so a run that needed the fix says so out loud
    instead of quietly working for reasons nobody can see.
    """
    import torch

    prec = device.precision(precision)
    if prec != "fp16":
        return {"precision": prec, "recast": 0}

    moved = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
            moved += 1
    total = sum(1 for p in model.parameters() if p.requires_grad)
    return {"precision": prec, "recast": moved, "trainable_tensors": total}


def lora_config_kwargs(spec: LoraSpec, target_modules: list[str]) -> dict:
    if spec.r is None or spec.alpha is None:
        raise ValueError(
            f"spec {spec.key!r} has an unresolved rank. Call "
            "`spec.resolved(modeling.matched_rank(...))` first — see NB4."
        )
    return dict(
        r=spec.r,
        lora_alpha=spec.alpha,                    # §10.3 invariant: alpha = 2r
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def summarize_run(spec: LoraSpec, tier: Tier, target_modules: list[str],
                  trainable: int, seconds: float, peak_vram_gb: float | None) -> dict:
    """One row of `results/runs.csv`. Same shape for every run so runs are comparable."""
    return {
        "run": spec.key,
        "label": spec.label,
        "tier": tier.name,
        "model": tier.model_id,
        # Recorded because wall-clock numbers are NOT comparable across precisions, and
        # this lab has already shipped one set of timings measured on a path (emulated
        # bf16 on a T4) that was later removed. A row without its precision is a row you
        # cannot compare to anything.
        "precision": device.precision(),
        "placement": spec.target,
        "n_target_modules": len(target_modules),
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "learning_rate": spec.lr,
        "load_in_4bit": spec.load_in_4bit,
        "trainable_params": trainable,
        "train_seconds": round(seconds, 1),
        "peak_vram_gb": None if peak_vram_gb is None else round(peak_vram_gb, 2),
    }
