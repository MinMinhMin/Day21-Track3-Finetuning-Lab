"""The loss-mask tests. If these fail, the lab teaches the wrong thing."""
import pytest
from fake_tokenizer import FakeTokenizer
from labkit import data
from labkit.config import NAIVE_PROMPT


ANSWER = "Ha Noi la thu do Viet Nam."
QUESTION = "Thu do Viet Nam la gi?"
MSGS = [
    {"role": "user", "content": QUESTION},
    {"role": "assistant", "content": ANSWER},
]
THINK_MSGS = [
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "<think>cong hai so</think>4"},
]


def test_assistant_only_supervises_the_answer_not_the_question():
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    supervised = data.decode_supervised(tok, ex)
    assert ANSWER in supervised
    assert QUESTION not in supervised, "the prompt leaked into the loss"
    assert 0 < ex.n_supervised < ex.n_total


def test_everything_mode_is_the_classic_bug():
    """`everything` supervises the prompt too — the §22 'model writes your question back'."""
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="everything")
    supervised = data.decode_supervised(tok, ex)
    assert QUESTION in supervised and ANSWER in supervised
    assert ex.n_supervised == ex.n_total
    assert ex.supervised_fraction == 1.0


def test_masked_think_excludes_the_reasoning_block():
    tok = FakeTokenizer()
    full = data.build_example(tok, THINK_MSGS, mask_mode="assistant-only")
    masked = data.build_example(tok, THINK_MSGS, mask_mode="masked-think")
    assert "cong hai so" in data.decode_supervised(tok, full)
    assert "cong hai so" not in data.decode_supervised(tok, masked)
    assert "4" in data.decode_supervised(tok, masked)
    assert masked.n_supervised < full.n_supervised


def test_response_only_also_drops_reasoning():
    tok = FakeTokenizer()
    ex = data.build_example(tok, THINK_MSGS, mask_mode="response-only")
    sup = data.decode_supervised(tok, ex)
    assert "cong hai so" not in sup and "4" in sup


def test_masking_is_a_partition():
    """Every token is either supervised or masked — never both, never neither."""
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    kept = sum(1 for l in ex.labels if l != data.IGNORE_INDEX)
    dropped = sum(1 for l in ex.labels if l == data.IGNORE_INDEX)
    assert kept + dropped == len(ex.input_ids) == ex.n_total
    assert len(ex.labels) == len(ex.input_ids)


def test_supervised_labels_equal_their_input_ids():
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    for tid, lab in zip(ex.input_ids, ex.labels):
        assert lab in (data.IGNORE_INDEX, tid), "label must be -100 or the token itself"


def test_multi_turn_supervises_every_assistant_turn():
    tok = FakeTokenizer()
    msgs = [
        {"role": "user", "content": "cau mot"},
        {"role": "assistant", "content": "dap mot"},
        {"role": "user", "content": "cau hai"},
        {"role": "assistant", "content": "dap hai"},
    ]
    ex = data.build_example(tok, msgs, mask_mode="assistant-only")
    sup = data.decode_supervised(tok, ex)
    assert "dap mot" in sup and "dap hai" in sup
    assert "cau mot" not in sup and "cau hai" not in sup


def test_truncation_keeps_labels_aligned():
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, max_length=20, mask_mode="assistant-only")
    assert len(ex.input_ids) == len(ex.labels) == 20
    assert ex.n_total == 20


def test_thinking_survives_detects_a_stripping_template():
    good = data.thinking_survives(FakeTokenizer(strip_thinking=False))
    bad = data.thinking_survives(FakeTokenizer(strip_thinking=True))
    assert good["ok"] is True and good["body_present"] is True
    assert bad["ok"] is False, "a template that deletes <think> must be flagged"
    assert "STRIPS" in bad["verdict"]


def test_unstable_template_raises_instead_of_masking_wrong():
    tok = FakeTokenizer(prefix_unstable=True)
    with pytest.raises(data.TemplateNotPrefixStable):
        data.build_example(tok, MSGS, mask_mode="assistant-only")


def test_bad_mask_mode_rejected():
    with pytest.raises(ValueError):
        data.build_example(FakeTokenizer(), MSGS, mask_mode="all-of-it")


def test_token_stats_percentiles_and_pow2():
    st = data.token_stats(list(range(1, 101)))
    assert st["n"] == 100 and st["p50"] == 50 and st["max"] == 100
    assert st["p95"] >= st["p50"]
    assert st["suggested_max_length"] in (256, 512, 1024, 2048)


def test_token_stats_empty():
    assert data.token_stats([]) == {"n": 0}


def test_to_messages_normalizes_alpaca_and_chat():
    rec = {"instruction": "dich cau nay", "input": "hello", "output": "xin chao"}

    # Default (F-31): the shape evaluation actually sends — short system prompt, bare
    # input in the user turn. The instruction is deliberately NOT folded in; internalising
    # it is what the fine-tune is for.
    m = data.to_messages(rec)
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[0]["content"] == NAIVE_PROMPT
    assert m[1]["content"] == "hello"
    assert "dich cau nay" not in m[1]["content"], (
        "folding the instruction into the user turn is the pre-F-31 shape; evaluation "
        "never sends it, so training on it cannot transfer")
    assert m[2] == {"role": "assistant", "content": "xin chao"}

    # system=None reproduces the old shape on purpose, so NB1 can show both.
    old = data.to_messages(rec, system=None)
    assert old[0]["role"] == "user"
    assert "dich cau nay" in old[0]["content"] and "hello" in old[0]["content"]

    passthrough = [{"role": "user", "content": "x"}]
    assert data.to_messages({"messages": passthrough}) == passthrough


def test_split_is_deterministic_and_disjoint():
    recs = list(range(100))
    a1, b1 = data.split(recs, seed=42)
    a2, b2 = data.split(recs, seed=42)
    assert a1 == a2 and b1 == b2
    assert len(a1) == 90 and len(b1) == 10
    assert not (set(a1) & set(b1))
    assert sorted(a1 + b1) == recs


# --- regressions from the Colab simulation (2026-08-21) ---------------------
# The first implementation diffed TOKEN lists around each assistant turn. That is
# wrong on Qwen3.5: the generation prompt ends `<think>\n` while the full render
# continues `\n</think>`, and consecutive newlines merge into a single different
# token. The strings are prefix-related; the token lists are not. These tests pin
# the offset-based behaviour that replaced it.

def test_think_scaffold_does_not_trigger_a_false_prefix_alarm():
    """The exact crash the Colab run hit: plain Q->A data on a reasoning template."""
    tok = FakeTokenizer()          # think_scaffold on, newline-merging on
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    assert ex.n_supervised > 0
    assert ANSWER in data.decode_supervised(tok, ex)


def test_token_lists_really_are_not_prefix_related():
    """Guards the fixture itself: if this stops holding, the regression above is vacuous."""
    tok = FakeTokenizer()
    prefix_ids = tok.apply_chat_template(MSGS[:1], tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(MSGS, tokenize=True)
    assert full_ids[:len(prefix_ids)] != prefix_ids, (
        "the fake no longer reproduces the token-merge boundary — "
        "the offset-masking regression test would pass for the wrong reason"
    )
    # ...but the STRINGS are prefix-related, which is why characters work.
    prefix_txt = tok.apply_chat_template(MSGS[:1], add_generation_prompt=True)
    full_txt = tok.apply_chat_template(MSGS)
    assert full_txt.startswith(prefix_txt)


def test_eos_token_is_supervised():
    """<|im_end|> must be in the loss or the model never learns to stop (deck §22)."""
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    assert tok.eos_token in data.decode_supervised(tok, ex)


def test_slow_tokenizer_fails_loudly_not_silently():
    tok = FakeTokenizer(fast=False)
    with pytest.raises(RuntimeError, match="offset mappings"):
        data.build_example(tok, MSGS, mask_mode="assistant-only")


def test_empty_think_block_is_excluded_by_masked_think():
    """Qwen3.5 emits <think>\\n\\n</think> even for plain answers; masked-think skips it."""
    tok = FakeTokenizer()
    plain = data.build_example(tok, MSGS, mask_mode="assistant-only")
    masked = data.build_example(tok, MSGS, mask_mode="masked-think")
    assert "</think>" in data.decode_supervised(tok, plain)
    assert "</think>" not in data.decode_supervised(tok, masked)
    assert ANSWER in data.decode_supervised(tok, masked)


# --- pre-tokenized training set (F-10 fix) ----------------------------------

def test_to_training_dataset_shape_and_mask():
    tok = FakeTokenizer()
    rows = data.to_training_dataset(
        tok, [{"instruction": QUESTION, "input": "", "output": ANSWER}] * 3,
        max_length=512)
    assert len(rows) == 3
    for r in rows:
        assert set(r) == {"input_ids", "labels", "attention_mask"}
        assert len(r["input_ids"]) == len(r["labels"]) == len(r["attention_mask"])
        sup = sum(1 for x in r["labels"] if x != data.IGNORE_INDEX)
        assert 0 < sup < len(r["labels"]), "must supervise some but not all tokens"


def test_to_training_dataset_keeps_only_rows_that_teach_something():
    """Every returned row must supervise at least one token — silently training on
    all-masked rows is how a run produces a loss curve that means nothing."""
    tok = FakeTokenizer()
    rows = data.to_training_dataset(
        tok, [{"instruction": QUESTION, "input": "", "output": ANSWER}] * 4,
        max_length=512)
    assert len(rows) == 4
    assert all(any(x != data.IGNORE_INDEX for x in r["labels"]) for r in rows)


def test_to_training_dataset_raises_if_nothing_is_supervised():
    tok = FakeTokenizer()
    with pytest.raises(RuntimeError, match="zero supervised tokens"):
        data.to_training_dataset(
            tok, [{"instruction": QUESTION, "input": "", "output": ANSWER}],
            max_length=6)
