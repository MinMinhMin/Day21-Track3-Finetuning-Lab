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
FP16_INITIAL_SCALE = 128.0


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
        logging_first_step=True,
        # The default filter substitutes previous losses for NaN micro-batches.
        # With accumulation it can inflate the logged loss exponentially, hiding
        # the actual failure (the Kaggle run printed 1.836e7).
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to="none",
        seed=seed,
        packing=False,       # we supply pre-tokenized labels -- see the note below
        # Same NLL math, but chunks the LM-head projection to avoid materializing all
        # sequence×vocabulary logits. Adapt Qwen3.5's partial forward before SFTTrainer.
        loss_type="chunked_nll",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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


def stabilize_qwen_fp16(model) -> dict:
    """Keep Qwen3.5's DeltaNet L2 normalization in FP32 on the FP16 path.

    Transformers 5.15 normalizes Q/K *before* its FP32 cast. For small or zero
    FP16 vectors, rsqrt's backward overflows (0 * inf -> NaN), even with an
    unscaled, finite forward loss. Returning FP32 normalized vectors fixes both
    the chunked and recurrent reference paths; their final output cast still
    preserves the model's original dtype. No base weights are upcast here.
    """
    if device.precision() != "fp16":
        return {"patched": False, "reason": "not FP16"}
    if not getattr(model.config, "model_type", "").startswith("qwen3_5"):
        return {"patched": False, "reason": "not Qwen3.5"}

    import importlib
    import torch

    module = importlib.import_module(type(model).__module__)
    original_norm = getattr(module, "l2norm", None)
    if original_norm is None:
        raise RuntimeError("Qwen3.5 l2norm helper changed; check the pinned Transformers version.")
    if getattr(original_norm, "_labkit_fp32_norm", False):
        return {"patched": False, "reason": "already patched", "norm_dtype": "fp32"}

    @functools.wraps(original_norm)
    def _fp32_l2norm(x, *args, **kwargs):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return original_norm(x.float(), *args, **kwargs)

    _fp32_l2norm._labkit_fp32_norm = True
    module.l2norm = _fp32_l2norm
    return {"patched": True, "norm_dtype": "fp32"}


def prepare_chunked_loss_forward(model) -> dict:
    """Make Qwen3.5's partial ``forward`` compatible with TRL's chunked-NLL patch.

    TRL wraps ``model.forward`` as a bound method and reads
    ``original_forward.__func__`` to preserve its signature. Qwen3.5 exposes its
    decorated forward as ``functools.partial`` instead. Bind a small delegating method
    and give it the effective partial signature with a synthetic leading ``self``.
    Calls still go through the original partial, including any arguments it captured.
    Other model implementations are left untouched.
    """
    numeric_fix = stabilize_qwen_fp16(model)
    loss_device_fix = align_chunked_loss_devices()
    original_forward = getattr(model, "forward", None)
    if not isinstance(original_forward, functools.partial):
        return {
            "patched": False,
            "reason": "forward is not functools.partial",
            "loss_device_alignment": loss_device_fix,
            "qwen_numerics": numeric_fix,
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
        "qwen_numerics": numeric_fix,
    }


def align_chunked_loss_devices() -> dict:
    """Align chunked-NLL tensors on the output-head device before the loss.

    With ``device_map="auto"`` on Kaggle's two T4s, the model can produce its final
    hidden states on a different device from the output embedding / ``lm_head``.
    TRL's chunked projection multiplies these tensors directly, so the loss fails if
    they are split across devices. Align hidden states and labels to the lm-head device;
    this transfers the much smaller activation instead of copying the large head weight.
    """
    try:
        import importlib
        sft_trainer = importlib.import_module("trl.trainer.sft_trainer")
        original_loss = sft_trainer._chunked_cross_entropy_loss
    except (ImportError, AttributeError) as exc:
        return {"patched": False, "reason": f"TRL chunked loss helper unavailable: {exc}"}

    if getattr(original_loss, "_labkit_align_devices", False):
        return {"patched": False, "reason": "already patched",
                "target_device": "lm_head_weight.device", "projection_dtype": "fp32"}

    signature = inspect.signature(original_loss)
    if "hidden_states" not in signature.parameters:
        return {
            "patched": False,
            "reason": "TRL chunked loss signature has no hidden_states parameter",
        }

    # Disabling autocast at the outer loss alone is insufficient: checkpoint
    # recomputation restores the AMP context. Protect the per-chunk function so
    # projection and softmax stay FP32 during both forward and backward.
    original_chunk = getattr(sft_trainer, "_chunk", None)
    if original_chunk is None:
        raise RuntimeError("TRL chunk helper changed; install the pinned TRL version.")
    if not getattr(original_chunk, "_labkit_fp32_projection", False):
        @functools.wraps(original_chunk)
        def _fp32_chunk(h, w, b, *args, **kwargs):
            import torch
            with torch.autocast(device_type=h.device.type, enabled=False):
                return original_chunk(h.float(), w.float(),
                                      None if b is None else b.float(), *args, **kwargs)

        _fp32_chunk._labkit_fp32_projection = True
        sft_trainer._chunk = _fp32_chunk

    @functools.wraps(original_loss)
    def _loss_with_aligned_devices(hidden_states, *args, **kwargs):
        bound = signature.bind_partial(hidden_states, *args, **kwargs)
        lm_head_weight = bound.arguments.get("lm_head_weight")
        if lm_head_weight is None:
            return original_loss(*bound.args, **bound.kwargs)

        target_device = lm_head_weight.device
        hidden = bound.arguments.get("hidden_states")
        if hidden is not None and hidden.device != target_device:
            bound.arguments["hidden_states"] = hidden.to(target_device)
        for name in ("labels", "shift_labels", "lm_head_bias"):
            value = bound.arguments.get(name)
            if value is not None and value.device != target_device:
                bound.arguments[name] = value.to(target_device)
        return original_loss(*bound.args, **bound.kwargs)

    _loss_with_aligned_devices._labkit_align_devices = True
    sft_trainer._chunked_cross_entropy_loss = _loss_with_aligned_devices
    return {"patched": True, "target_device": "lm_head_weight.device", "projection_dtype": "fp32"}


def configure_fp16_scaler(trainer) -> dict:
    """Start AMP conservatively; its default 65536 can overflow FP16 backward.

    Use GradScaler's public state API before its first step. Dynamic scaling and
    overflow skipping remain enabled. NB3/NB4 start at the same scale; preflight
    can lower it when that model's scaled backward overflows.
    """
    scaler = trainer.accelerator.scaler
    if not trainer.args.fp16 or scaler is None or not scaler.is_enabled():
        return {"changed": False, "reason": "no FP16 GradScaler"}
    state = scaler.state_dict()
    state["scale"] = FP16_INITIAL_SCALE
    scaler.load_state_dict(state)
    return {"changed": True, "initial_scale": scaler.get_scale()}


def preflight_training(trainer) -> dict:
    """Check a real batch's scaled backward before any optimizer updates.

    Probe through the trainer's collator, input placement, AMP context and patched
    model. A finite forward can still overflow during scaled FP16 backward. Retry
    the identical batch/RNG with GradScaler's backoff factor, at most 16 times.
    Commit the lower scale only after all gradients are finite and some nonzero.
    Preserve weights/RNG and always clear gradients, including on error. Loss NaN
    and persistent backward NaN remain fatal; no optimizer step is performed here.
    """
    import torch

    model = trainer.model
    was_training = model.training
    scaler = trainer.accelerator.scaler
    scale = scaler.get_scale() if scaler is not None and scaler.is_enabled() else 1.0
    initial_scale = scale
    can_backoff = (getattr(trainer.args, "fp16", False)
                   and scaler is not None and scaler.is_enabled())
    backoff_factor = scaler.get_backoff_factor() if can_backoff else None
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.train()
            model.zero_grad(set_to_none=True)
            if trainer.args.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs or {})
            inputs = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[0]]))
            if not bool((inputs["labels"][..., 1:] != -100).any()):
                raise ValueError("Preflight batch has no supervised next-token labels.")
            for backoffs in range(17):
                model.zero_grad(set_to_none=True)
                # Each attempt sees identical dropout and checkpoint recomputation.
                with torch.random.fork_rng(devices=cuda_devices):
                    with trainer.accelerator.autocast():
                        loss = model(**inputs, use_cache=False).loss
                    if not bool(torch.isfinite(loss).all()):
                        raise FloatingPointError(
                            f"Preflight loss is non-finite: {loss.detach().float().item()}")
                    (loss * scale).backward()
                n_grad = 0
                nonzero = False
                bad_gradient = None
                for name, param in model.named_parameters():
                    if not param.requires_grad or param.grad is None:
                        continue
                    n_grad += 1
                    if not bool(torch.isfinite(param.grad).all()):
                        bad_gradient = name
                        break
                    nonzero = nonzero or bool(param.grad.ne(0).any())
                if bad_gradient is None:
                    if not n_grad or not nonzero:
                        raise FloatingPointError("Preflight has no nonzero trainable gradients.")
                    if can_backoff and scale != initial_scale:
                        state = scaler.state_dict()
                        state["scale"] = scale
                        state["_growth_tracker"] = 0
                        scaler.load_state_dict(state)
                    return {"loss": loss.detach().float().item(), "loss_scale": scale,
                            "initial_loss_scale": initial_scale, "scale_backoffs": backoffs,
                            "gradient_tensors": n_grad, "finite": True}
                if not can_backoff or backoffs == 16:
                    raise FloatingPointError(
                        f"Preflight gradient {bad_gradient!r} contains NaN/Inf "
                        f"(loss_scale={scale}, scale_backoffs={backoffs}, "
                        f"loss={loss.detach().float().item()}). "
                        "No optimizer update or adapter save was performed. "
                        "FP16 scale calibration could not produce finite gradients.")
                next_scale = scale * backoff_factor
                warnings.warn(
                    f"Preflight FP16 overflow in {bad_gradient!r} at loss_scale={scale}; "
                    f"retrying identical batch at {next_scale}. No optimizer update performed.",
                    RuntimeWarning, stacklevel=2)
                scale = next_scale
                del loss
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def install_finite_metrics_guard(trainer):
    """Reject invalid training, allowing only AMP-confirmed skipped overflows.

    A logged NaN grad_norm can describe an update GradScaler already skipped.
    Verify that Accelerate reports the skip, the scale decreased, and adapter
    weights remain finite before accepting it. Loss/entropy NaNs always fail.
    """
    from transformers import TrainerCallback

    class _FiniteMetricsGuard(TrainerCallback):
        def __init__(self):
            self.attempted = 0
            self.updated = 0
            self.skipped = 0
            self._scale_before_step = None
            self._confirmed_overflow_step = None

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            scaler = trainer.accelerator.scaler
            self._scale_before_step = (
                scaler.get_scale() if scaler is not None and scaler.is_enabled() else None)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.attempted += 1
            self._confirmed_overflow_step = None
            if trainer.accelerator.optimizer_step_was_skipped:
                scaler = trainer.accelerator.scaler
                scale_after = (
                    scaler.get_scale() if scaler is not None and scaler.is_enabled() else None)
                confirmed = (args.fp16 and self._scale_before_step is not None
                             and scale_after is not None
                             and 0 < scale_after < self._scale_before_step)
                if not confirmed:
                    raise FloatingPointError(
                        f"Optimizer update skipped at step {state.global_step} without "
                        "a confirmed FP16 GradScaler backoff; refusing to continue.")
                assert_finite_training(trainer.model, 0.0)
                self.skipped += 1
                self._confirmed_overflow_step = state.global_step
                warnings.warn(
                    f"AMP overflow at step {state.global_step}: optimizer update skipped, "
                    f"loss scale {self._scale_before_step:g} -> {scale_after:g}; "
                    "trainable weights remain finite. Continuing with the lower scale.",
                    RuntimeWarning, stacklevel=2)
            else:
                self.updated += 1
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            for name in ("loss", "eval_loss", "entropy", "grad_norm"):
                value = (logs or {}).get(name)
                if value is None:
                    continue
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError, OverflowError):
                    finite = False
                if not finite:
                    if name == "grad_norm" and self._confirmed_overflow_step == state.global_step:
                        continue
                    raise FloatingPointError(
                        f"Training stopped: logged {name}={value!r} at step "
                        f"{state.global_step}; no adapter should be saved."
                    )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if not self.updated:
                raise FloatingPointError(
                    "Training performed no successful optimizer update; refusing to save adapter.")
            return control

        def summary(self) -> dict:
            scaler = trainer.accelerator.scaler
            return {
                "optimizer_steps_attempted": self.attempted,
                "optimizer_updates": self.updated,
                "amp_skipped_steps": self.skipped,
                "amp_final_scale": (scaler.get_scale()
                                    if scaler is not None and scaler.is_enabled() else None),
            }

    guard = _FiniteMetricsGuard()
    trainer.add_callback(guard)
    return guard


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
