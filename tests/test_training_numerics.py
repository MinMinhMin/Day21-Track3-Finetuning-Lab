"""Real tensor regressions for the Kaggle FP16 failure (no model download)."""
from types import SimpleNamespace

import pytest

from labkit import device, train


@pytest.fixture
def qwen(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    monkeypatch.setattr(device, "precision", lambda explicit=None: explicit or "fp16")
    # Restore upstream globals after each test: the fix is installed process-wide.
    monkeypatch.setattr(modeling_qwen3_5, "l2norm", modeling_qwen3_5.l2norm)
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_conv_kernel_dim=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2, max_position_embeddings=256,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [3, 3, 2]},
    )
    return torch, modeling_qwen3_5, Qwen3_5ForCausalLM(config).half()


def test_fp16_zero_query_normalization_has_finite_backward(qwen):
    """Moving the FP32 cast after rsqrt reproduces 0 * inf -> NaN in backward."""
    torch, module, model = qwen
    train.prepare_chunked_loss_forward(model)
    x = torch.zeros(1, 8, dtype=torch.float16, requires_grad=True)
    module.l2norm(x).sum().backward()
    # At x=0 the Jacobian is I / sqrt(1e-6), independently of the implementation.
    torch.testing.assert_close(x.grad, torch.full_like(x, 1000.), rtol=.002, atol=0.)


def test_chunked_projection_remains_finite_inside_fp16_autocast(monkeypatch):
    """FP32 casts alone do not protect matmul while AMP is enabled."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from trl.trainer import sft_trainer as sft
    monkeypatch.setattr(sft, "_chunk", sft._chunk)
    monkeypatch.setattr(sft, "_chunked_cross_entropy_loss", sft._chunked_cross_entropy_loss)
    train.align_chunked_loss_devices()
    h = torch.tensor([[[200., 200.], [200., 200.]]], requires_grad=True)
    w = torch.tensor([[200., 200.], [199., 199.]])
    labels = torch.tensor([[-100, 0]])
    with torch.autocast("cpu", dtype=torch.float16):
        loss, correct, entropy, count = sft._chunked_cross_entropy_loss(h, w, 1, labels=labels)
    loss.backward()
    # Logits are 80000 and 79600: class 0 has effectively zero loss/entropy.
    assert loss.item() == pytest.approx(0., abs=1e-6)
    assert entropy.item() == pytest.approx(0., abs=1e-6)
    assert correct.item() == count.item() == 1
    assert bool(h.grad.isfinite().all())


def test_fp16_scaler_starts_without_overflowing_the_first_backward():
    torch = pytest.importorskip("torch")
    scaler = torch.amp.GradScaler("cpu")
    trainer = SimpleNamespace(args=SimpleNamespace(fp16=True),
                              accelerator=SimpleNamespace(scaler=scaler))
    train.configure_fp16_scaler(trainer)
    x = torch.tensor(1., dtype=torch.float16, requires_grad=True)
    scaler.scale(x.float()).backward()
    assert bool(x.grad.isfinite()), "65536 exceeds FP16's finite range before any update"


@pytest.mark.parametrize("target,lr", [
    ("text-linear", 1e-4), ("attn-only", 1e-4), ("text-linear", 1e-5),
])
def test_qwen_fp16_lora_updates_have_finite_gradients(qwen, monkeypatch, target, lr):
    """Exercise the actual hybrid decoder, PEFT, TRL loss and checkpointed backward."""
    torch, _, model = qwen
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from peft import LoraConfig, get_peft_model
    from trl.trainer import sft_trainer as sft
    from labkit import modeling

    monkeypatch.setattr(sft, "_chunk", sft._chunk)
    monkeypatch.setattr(sft, "_chunked_cross_entropy_loss", sft._chunked_cross_entropy_loss)
    targets = modeling.resolve_target_modules(model, target)
    train.prepare_chunked_loss_forward(model)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets))
    train.align_trainable_precision(model)
    sft._patch_chunked_ce_lm_head(model.get_base_model(), chunk_size=16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    ids = torch.randint(0, 64, (1, 96))
    labels = ids.clone()
    labels[:, :48] = -100
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cpu")
    trainer = SimpleNamespace(args=SimpleNamespace(fp16=True),
                              accelerator=SimpleNamespace(scaler=scaler))
    train.configure_fp16_scaler(trainer)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.float16):
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        assert bool(loss.isfinite())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(bool(g.isfinite().all()) for g in grads)
        assert any(bool(g.ne(0).any()) for g in grads), "finite zeros would not train an adapter"
        scaler.step(optimizer)
        scaler.update()
    train.assert_finite_training(model, loss.item())


def test_nonfinite_loss_is_not_hidden_by_logging_filter():
    from labkit.config import SPECS, get_tier
    kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "out", precision="fp16")
    assert kw.get("logging_nan_inf_filter") is False


def test_preflight_rejects_nonfinite_gradients_and_clears_them():
    torch = pytest.importorskip("torch")

    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad * float("nan")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, **kwargs):
            return SimpleNamespace(loss=BadBackward.apply(self.weight))

    model = Model()
    trainer = SimpleNamespace(
        model=model, train_dataset=[{"input_ids": [0, 1], "labels": [-100, 1]}],
        args=SimpleNamespace(gradient_checkpointing=False),
        accelerator=SimpleNamespace(scaler=None, autocast=lambda: torch.autocast("cpu", enabled=False)),
        data_collator=lambda rows: {k: torch.tensor([v]) for k, v in rows[0].items()},
        _prepare_inputs=lambda inputs: inputs,
    )
    with pytest.raises(FloatingPointError, match="weight"):
        train.preflight_training(trainer)
    assert model.weight.grad is None
    assert model.weight.item() == 1.


def test_preflight_preserves_weights_rng_and_training_mode():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.weight.square() * (torch.rand(()) + 1))

    model = Model().eval()
    before_rng = torch.random.get_rng_state().clone()
    trainer = SimpleNamespace(
        model=model, train_dataset=[{"input_ids": [0, 1], "labels": [-100, 1]}],
        args=SimpleNamespace(gradient_checkpointing=False),
        accelerator=SimpleNamespace(scaler=None, autocast=lambda: torch.autocast("cpu", enabled=False)),
        data_collator=lambda rows: {k: torch.tensor([v]) for k, v in rows[0].items()},
        _prepare_inputs=lambda inputs: inputs,
    )
    result = train.preflight_training(trainer)
    assert result["finite"] and result["gradient_tensors"] == 1
    assert model.weight.item() == 1. and model.weight.grad is None
    assert model.training is False
    assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_sft_preflight_and_training_use_the_same_finite_path(qwen, monkeypatch, tmp_path):
    """Check the notebook sequence against real SFTTrainer, collator and optimizer."""
    torch, _, model = qwen
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer
    from trl.trainer import sft_trainer as sft
    from labkit import modeling

    monkeypatch.setattr(sft, "_chunk", sft._chunk)
    monkeypatch.setattr(sft, "_chunked_cross_entropy_loss", sft._chunked_cross_entropy_loss)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"t{i}": i for i in range(3, 64)}}
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
                                 pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    ids = torch.randint(3, 64, (96,)).tolist()
    ds = Dataset.from_list([{"input_ids": ids, "labels": [-100] * 48 + ids[48:]}] * 2)
    targets = modeling.resolve_target_modules(model, "text-linear")
    train.prepare_chunked_loss_forward(model)
    # CPU has no CUDA AMP; other tests cover FP16 autocast/scaling. Frozen weights
    # remain half here, exercising the original failing normalization dtype.
    args = SFTConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False, max_length=128,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=2,
        report_to="none", save_strategy="no", logging_steps=1,
        logging_nan_inf_filter=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, loss_type="chunked_nll",
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=targets))
    train.align_trainable_precision(trainer.model)
    train.configure_fp16_scaler(trainer)
    train.install_finite_metrics_guard(trainer)
    assert train.preflight_training(trainer)["finite"] is True
    assert trainer.state.global_step == 0
    assert all(p.grad is None for p in trainer.model.parameters())
    result = trainer.train()
    train.assert_finite_training(trainer.model, result.training_loss)
    assert trainer.state.global_step == 2
    trainer.model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()


@pytest.mark.parametrize("shifted", [False, True])
def test_chunked_loss_backward_across_two_gpus(monkeypatch, shifted):
    """Runs on Kaggle T4x2 during smoke; skipped on a CPU or single GPU."""
    import math
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    pytest.importorskip("trl")
    from trl.trainer import sft_trainer as sft
    monkeypatch.setattr(sft, "_chunk", sft._chunk)
    monkeypatch.setattr(sft, "_chunked_cross_entropy_loss", sft._chunked_cross_entropy_loss)
    train.align_chunked_loss_devices()
    h = torch.tensor([[[1., 2.], [3., 4.]]], device="cuda:1", requires_grad=True)
    w = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], device="cuda:0")
    labels = torch.tensor([[1, -100]] if shifted else [[-100, 1]], device="cuda:1")
    kw = {"shift_labels" if shifted else "labels": labels}
    with torch.autocast("cuda", dtype=torch.float16):
        loss, _, entropy, count = sft._chunked_cross_entropy_loss(h, w, 1, **kw)
    loss.backward()
    assert loss.device == w.device
    assert loss.item() == pytest.approx(math.log(1 + math.exp(-1) + math.exp(-3)), abs=1e-5)
    assert count.item() == 1 and bool(entropy.isfinite())
    assert h.grad.device == h.device and bool(h.grad.isfinite().all())
