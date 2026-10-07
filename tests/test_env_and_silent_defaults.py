"""Regression tests for the silent-configuration defects (F-25..F-30).

Every case here is a failure that produced no error message: a `.env` that was written,
read by nobody, and obeyed by nothing; a gatekeeper that green-lit a run it had the data
to reject; a verdict decided by string-matching its own prose. They are grouped in one
file because they share that shape, not because they share a module.
"""
from __future__ import annotations

import os
import warnings

import pytest

from labkit import env as labenv
from labkit import evaluate as ev
from labkit.config import get_tier


# --- F-25: .env is actually read ---------------------------------------------

def test_parse_handles_the_forms_dotenv_example_uses():
    parsed = labenv.parse(
        "# comment\n"
        "\n"
        "COMPUTE_TIER=LAPTOP\n"
        "export MASK_MODE=masked-think\n"
        'QUOTED="two words"\n'
        "SINGLE='x'\n"
        "EPOCHS=3   # trailing comment\n"
        "NOT_A_PAIR\n"
    )
    assert parsed == {
        "COMPUTE_TIER": "LAPTOP",
        "MASK_MODE": "masked-think",
        "QUOTED": "two words",
        "SINGLE": "x",
        "EPOCHS": "3",
    }


def test_dotenv_reaches_get_tier(tmp_path, monkeypatch):
    """The bug: README/HARDWARE-GUIDE/.env.example all promised this and it never worked."""
    monkeypatch.delenv("COMPUTE_TIER", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("COMPUTE_TIER=LAPTOP\n", encoding="utf-8")
    labenv.load_dotenv(dotenv)
    assert os.environ["COMPUTE_TIER"] == "LAPTOP"
    assert get_tier().name == "LAPTOP"


def test_explicit_environment_beats_the_file(tmp_path, monkeypatch):
    """`EVAL_LIMIT=8 make pipeline` and CI must override .env, not lose to it."""
    monkeypatch.setenv("COMPUTE_TIER", "BIGGPU")
    dotenv = tmp_path / ".env"
    dotenv.write_text("COMPUTE_TIER=CPU\n", encoding="utf-8")
    assert labenv.load_dotenv(dotenv) == {}
    assert get_tier().name == "BIGGPU"


def test_dotenv_also_carries_epochs(tmp_path, monkeypatch):
    """EPOCHS is the lever `training_epochs()` reads; it must survive the file too."""
    from labkit.config import training_epochs

    monkeypatch.delenv("EPOCHS", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("EPOCHS=1\n", encoding="utf-8")
    labenv.load_dotenv(dotenv)
    assert training_epochs() == 1.0


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert labenv.load_dotenv(tmp_path / "nope.env") == {}


# --- F-29: the verdict comes from the numbers, not from the prose ------------

def _scores(target, regression):
    return ev.GroupScores(target=target, regression=regression, format=1.0,
                          latency_ms=1.0, n=10)


def test_verdict_survives_rewording_its_reasons():
    """It used to be `not any(r.startswith(("target", "general")) ...)`. A graded
    pass/fail must not depend on the first word of a human-readable sentence — least of
    all in a lab whose prose is Vietnamese."""
    base = _scores(0.50, 0.70)
    assert ev.regression_gate(_scores(0.80, 0.70), base).passed is True
    assert ev.regression_gate(_scores(0.40, 0.70), base).passed is False   # lost target
    assert ev.regression_gate(_scores(0.80, 0.50), base).passed is False   # forgot
    assert ev.regression_gate(_scores(0.50, 0.70), base).passed is False   # tie != win


def test_small_regression_inside_tolerance_still_passes():
    """A run may trade a little general capability for target gain — that is what the
    tolerance is for. Deliberately not testing the exact knife-edge: `0.70 - 0.02` is
    0.6799999999999999 in binary floating point, so an equality test there asserts
    something about IEEE 754 rather than about the gate.
    """
    base = _scores(0.50, 0.70)
    inside = ev.regression_gate(_scores(0.60, 0.70 - ev.REGRESSION_TOLERANCE / 2), base)
    assert inside.passed is True
    outside = ev.regression_gate(_scores(0.60, 0.70 - ev.REGRESSION_TOLERANCE * 2), base)
    assert outside.passed is False


# --- F-30: mask modes announce themselves when they cannot do anything -------

def test_reasoningless_corpus_warns_for_think_modes():
    """The shipped corpus is 250 bare-JSON answers, and the generation prompt already
    closes an empty <think></think>, so `masked-think` and `response-only` are
    byte-identical to `assistant-only` there. Silent no-ops are what this lab is about."""
    from labkit import data
    from fake_tokenizer import FakeTokenizer

    records = [{"instruction": "i", "input": "x", "output": '{"a": 1}'}]
    with pytest.warns(RuntimeWarning, match="no-op on this corpus"):
        data.to_training_dataset(FakeTokenizer(), records, max_length=128,
                                 mask_mode="masked-think")


def test_no_warning_when_the_corpus_has_traces():
    from labkit import data
    from fake_tokenizer import FakeTokenizer

    records = [{"instruction": "i", "input": "x",
                "output": '<think>vi sao</think>\n{"a": 1}'}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        data.to_training_dataset(FakeTokenizer(), records, max_length=128,
                                 mask_mode="masked-think")


def test_assistant_only_never_warns():
    """The default mode is not affected by any of this."""
    from labkit import data
    from fake_tokenizer import FakeTokenizer

    records = [{"instruction": "i", "input": "x", "output": '{"a": 1}'}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        data.to_training_dataset(FakeTokenizer(), records, max_length=128,
                                 mask_mode="assistant-only")


def test_base_model_override_keeps_tier_settings(monkeypatch):
    """Students may pick their own base model; the tier's hardware settings stay."""
    from labkit.config import get_tier
    monkeypatch.setenv("BASE_MODEL", "Qwen/Qwen3.5-2B")
    t = get_tier("T4")
    assert t.model_id == "Qwen/Qwen3.5-2B"
    assert (t.max_length, t.per_device_batch, t.grad_accum) == (1024, 1, 16)
    monkeypatch.delenv("BASE_MODEL")
    assert get_tier("T4").model_id == "unsloth/Qwen3.5-4B"
