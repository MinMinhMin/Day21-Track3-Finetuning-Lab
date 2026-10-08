"""Preflight must calibrate real FP16 overflow without updating an adapter."""
from types import SimpleNamespace
import warnings

import pytest

from labkit import train


def _trainer(torch, model, *, fp16=True):
    return SimpleNamespace(
        model=model, train_dataset=[{"input_ids": [0, 1], "labels": [-100, 1]}],
        args=SimpleNamespace(fp16=fp16, gradient_checkpointing=False),
        accelerator=SimpleNamespace(
            scaler=torch.amp.GradScaler("cpu", init_scale=128.),
            autocast=lambda: torch.autocast("cpu", dtype=torch.float16)),
        data_collator=lambda rows: {k: torch.tensor([v]) for k, v in rows[0].items()},
        _prepare_inputs=lambda inputs: inputs,
    )


@pytest.mark.parametrize("gain,expected_scale", [(800., 64.), (131072., .25)])
def test_preflight_backs_off_real_fp16_overflow_and_keeps_weights(gain, expected_scale):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.0001))
            self.draws = []

        def forward(self, **kwargs):
            self.draws.append(torch.rand(()).item())
            h = self.weight.half()
            # Both terms are finite, but their scaled backward branches can be
            # inf - inf -> NaN. Gradients recover even if the scale must be < 1.
            return SimpleNamespace(loss=(h * (gain * .75) - h * gain).float() + 20.)

    model = Model().eval()
    before = model.weight.detach().clone()
    rng = torch.random.get_rng_state().clone()
    trainer = _trainer(torch, model)
    with pytest.warns(RuntimeWarning, match="Preflight FP16 overflow"):
        result = train.preflight_training(trainer)
    assert result["finite"] is True
    assert result["loss_scale"] == expected_scale
    assert result["scale_backoffs"] > 0
    assert trainer.accelerator.scaler.get_scale() == expected_scale
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None
    assert model.training is False
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert len(model.draws) > 1 and len(set(model.draws)) == 1
    # The calibrated scale must also work through the actual GradScaler update.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model().loss
    trainer.accelerator.scaler.scale(loss).backward()
    trainer.accelerator.scaler.step(optimizer)
    trainer.accelerator.scaler.update()
    assert bool(model.weight.isfinite())
    assert not torch.equal(model.weight, before)


def test_preflight_keeps_an_already_finite_scaler_state():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.weight.square())

    trainer = _trainer(torch, Model())
    state = trainer.accelerator.scaler.state_dict()
    state["scale"] = 64.
    state["_growth_tracker"] = 5
    trainer.accelerator.scaler.load_state_dict(state)
    with warnings.catch_warnings(record=True) as notices:
        result = train.preflight_training(trainer)
    assert not notices
    assert result["loss_scale"] == 64. and result["scale_backoffs"] == 0
    assert trainer.accelerator.scaler.state_dict() == state
    assert trainer.model.weight.item() == 1. and trainer.model.weight.grad is None


def test_preflight_does_not_retry_nonfinite_forward_loss():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(loss=self.weight * float("nan"))

    model = Model()
    trainer = _trainer(torch, model)
    with pytest.raises(FloatingPointError, match="loss is non-finite"):
        train.preflight_training(trainer)
    assert model.calls == 1 and model.weight.grad is None
    assert trainer.accelerator.scaler.get_scale() == 128.


def test_preflight_still_rejects_persistent_bad_backward():
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
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(loss=BadBackward.apply(self.weight))

    model = Model().eval()
    trainer = _trainer(torch, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(FloatingPointError, match="weight.*NaN/Inf"):
            train.preflight_training(trainer)
    assert 1 < model.calls <= 17, "scale calibration must be bounded"
    assert model.weight.item() == 1. and model.weight.grad is None
    assert model.training is False
    assert trainer.accelerator.scaler.get_scale() == 128., "failed preflight restores scaler"


def test_preflight_calibrates_real_nf4_backward():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.full((1, 64), .0001))
            self.base = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16,
                                         quant_type="nf4", compress_statistics=True)
            self.base.weight = bnb.nn.Params4bit(
                torch.eye(64) * 2048., requires_grad=False,
                quant_type="nf4", compress_statistics=True).to("cpu")

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.base(self.adapter).float().sum())

    model = Model()
    trainer = _trainer(torch, model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    packed_before = model.base.weight.data_ptr()
    with pytest.warns(RuntimeWarning, match="Preflight FP16 overflow"):
        result = train.preflight_training(trainer)
    assert result["finite"] is True and result["loss_scale"] == 16.
    assert model.base.weight.dtype == torch.uint8
    assert model.base.weight.data_ptr() == packed_before
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]) and p.grad is None
    optimizer = torch.optim.AdamW([model.adapter], lr=1e-4)
    with trainer.accelerator.autocast():
        loss = model().loss
    trainer.accelerator.scaler.scale(loss).backward()
    trainer.accelerator.scaler.step(optimizer)
    trainer.accelerator.scaler.update()
    assert bool(model.adapter.isfinite().all())
    assert not torch.equal(model.adapter, before["adapter"])
