"""Recoverable FP16 overflow must not be mistaken for a poisoned adapter."""
from types import SimpleNamespace
import warnings

import pytest

from labkit import train


@pytest.fixture
def run():
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("transformers")
    from accelerate import Accelerator
    from transformers import TrainerControl, TrainerState

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.0001))

        def forward(self):
            h = self.weight.half()
            # Finite forward ~0.04. At scale 128, the FP16 gradient branches
            # overflow with opposite signs (inf - inf -> NaN); scale 64 recovers.
            return (h * 600 - h * 800).float() + .06

    acc = Accelerator(cpu=True)
    acc.scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    model = Model()
    optimizer = acc.prepare_optimizer(torch.optim.AdamW(model.parameters(), lr=1e-4))
    callbacks = []
    trainer = SimpleNamespace(model=model, accelerator=acc, add_callback=callbacks.append)
    train.install_finite_metrics_guard(trainer)
    return SimpleNamespace(torch=torch, model=model, acc=acc, optimizer=optimizer,
                           guard=callbacks[-1], args=SimpleNamespace(fp16=True),
                           state=TrainerState(), control=TrainerControl())


def _update(run):
    """Drive the actual AMP/optimizer sequence used by Trainer, including clipping."""
    run.optimizer.zero_grad()
    loss = run.model()
    run.acc.scaler.scale(loss).backward()
    run.acc.scaler.unscale_(run.optimizer.optimizer)
    norm = run.torch.nn.utils.clip_grad_norm_(run.model.parameters(), 1.)
    run.guard.on_pre_optimizer_step(run.args, run.state, run.control)
    run.optimizer.step()
    run.model.zero_grad(set_to_none=True)
    run.state.global_step += 1
    run.guard.on_step_end(run.args, run.state, run.control)
    return {"loss": loss.item(), "grad_norm": norm.item()}


def test_confirmed_amp_skip_can_recover_without_changing_weights(run):
    before = run.model.weight.detach().clone()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        logs = _update(run)
        assert run.acc.optimizer_step_was_skipped is True
        assert run.acc.scaler.get_scale() == 64.
        assert run.torch.equal(run.model.weight, before)
        assert bool(run.torch.isfinite(run.model.weight))
        run.guard.on_log(run.args, run.state, run.control, logs=logs)
    assert any("AMP" in str(w.message) for w in notices)
    # The following real update is finite and changes the adapter parameter.
    logs = _update(run)
    assert run.acc.optimizer_step_was_skipped is False
    assert not run.torch.equal(run.model.weight, before)
    run.guard.on_log(run.args, run.state, run.control, logs=logs)
    run.guard.on_train_end(run.args, run.state, run.control)
    summary = run.guard.summary()
    assert summary["optimizer_steps_attempted"] == 2
    assert summary["optimizer_updates"] == 1
    assert summary["amp_skipped_steps"] == 1


@pytest.mark.parametrize("name", ["loss", "eval_loss", "entropy", "grad_norm"])
def test_unconfirmed_nonfinite_metrics_still_stop_training(run, name):
    with pytest.raises(FloatingPointError, match=name):
        run.guard.on_log(run.args, run.state, run.control,
                         logs={"loss": .04, name: float("nan")})


def test_nonfinite_loss_still_stops_after_confirmed_amp_skip(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _update(run)
    with pytest.raises(FloatingPointError, match="loss"):
        run.guard.on_log(run.args, run.state, run.control,
                         logs={"loss": float("nan"), "grad_norm": float("nan")})


def test_amp_skip_does_not_allow_poisoned_weights(run):
    # Execute a real skipped AMP step, then poison the weight before step-end.
    loss = run.model()
    run.acc.scaler.scale(loss).backward()
    run.acc.scaler.unscale_(run.optimizer.optimizer)
    run.guard.on_pre_optimizer_step(run.args, run.state, run.control)
    run.optimizer.step()
    assert run.acc.optimizer_step_was_skipped is True
    with run.torch.no_grad():
        run.model.weight.fill_(float("nan"))
    run.state.global_step = 1
    with pytest.raises(FloatingPointError, match="weight"):
        run.guard.on_step_end(run.args, run.state, run.control)


def test_run_with_no_successful_updates_cannot_be_saved(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _update(run)
    with pytest.raises(FloatingPointError, match="optimizer update"):
        run.guard.on_train_end(run.args, run.state, run.control)
