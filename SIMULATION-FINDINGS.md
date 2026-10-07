# Simulation findings — Lab 21

Student simulation of the documented path. Two environments:

| Env | Hardware | Python | Purpose |
|---|---|---|---|
| **local** | Apple M4, 16 GB, MPS (no CUDA) | 3.14.3 | fast reproduction + fixes |
| **Colab** | free-tier **Tesla T4 16 GB** | 3.x (Colab) | the documented default path |

Stack resolved on both (from PyPI, 2026-08-21):
`torch 2.13.0 · transformers 5.15.1 · trl 1.10.0 · peft 0.20.0 · accelerate 1.14.0 ·
datasets 5.0.1` — `bitsandbytes` correctly skipped on macOS by the platform marker.

---

## F-01 — NB1 crashed on the lab's own default model — **FIXED**

**Severity: critical.** The lab could not get past its first notebook on `Qwen3.5-4B`.

**Symptom.** `TemplateNotPrefixStable: turn 1: rendering messages[:2] does not extend
messages[:1]`. Raised on *ordinary* prompt→answer data, not an edge case.

**Root cause — subtle and worth reading.** `build_example()` diffed **token lists**
around each assistant turn:

```
generation prompt ends:  ...<|im_start|>assistant\n<think>\n
full render continues:   ...<think>\n\n</think>\n\n{answer}<|im_end|>\n
```

The *strings* are prefix-related (`full.startswith(prefix)` is `True`). The *token
lists* are not: the prefix's trailing `\n` is one token, but in the full render `\n\n`
merges into a single **different** token. Diffing tokens therefore compares
non-comparable sequences.

Two aggravating factors specific to 2026 reasoning models:
* Qwen3.5 emits `<think>\n\n</think>\n\n` **even when the answer contains no
  reasoning**, so every sample crossed this boundary.
* The template also *normalizes* author-supplied `<think>` blocks, so hand-written
  reasoning data renders differently from what you wrote.

**Fix.** Render to text, tokenize once with `return_offsets_mapping=True`, and
supervise tokens whose character span falls inside `[len(prefix), len(upto))`.
Verified: this reproduces `apply_chat_template(tokenize=True)` ids **exactly**, and
special tokens carry real offsets, so `<|im_end|>` — the stop signal — stays
supervised.

**Why the test suite missed it.** The fake tokenizer was a plain ChatML renderer with
character-level tokenization: no think scaffold, no token merging. It could not
express the failure. The fake now reproduces both behaviours, plus 5 regression tests
(58 → 63), including one that asserts the *fixture itself* still produces
non-prefix-related token lists — otherwise the regression test would pass vacuously.

---

## F-02 — `resolve_target_modules` picks up hybrid-attention projections — **BY DESIGN, documented**

On the real `Qwen3.5-4B`, `text-linear` resolves to **12** suffixes, not the 7 a plain
transformer has:

```
down_proj gate_proj up_proj  q_proj k_proj v_proj o_proj
in_proj_a in_proj_b in_proj_qkv in_proj_z out_proj   <- Gated DeltaNet layers
```

The extra five are the **linear-attention** layers (deck §7.4: 24 linear + 8 full
attention, a 3:1 interleave). Adapting them is correct — they are part of the text
decoder — but it changes the arithmetic: matched rank for attention-only is **r≈283**
on the real model versus r≈90 on a plain-transformer shape. `matched_rank()` computes
this at runtime, so the contrast stays fair. Confirmed the vision tower is still
excluded.

---

## F-03 — `AutoModelForCausalLM` accepts the multimodal config — **NO ACTION**

De-risked without downloading weights via `init_empty_weights()` +
`from_config`. `Qwen3_5Config` (architectures: `Qwen3_5ForConditionalGeneration`)
loads as `Qwen3_5ForCausalLM`. `generate.load_base()` works on the default tier.

---

## F-04 — `transformers` does not depend on `jinja2` — **FIXED (pre-Colab)**

`apply_chat_template` raises `ImportError` without it. Would have broken NB1 on cell
one for every student. Pinned in both requirements files.

---

## F-05 — Colab free tier allows only ONE GPU session — **DOCUMENTED**

Opening a second lab notebook while the first holds a runtime gives *"Quá nhiều phiên
đang hoạt động"* and the second silently never starts. Students running NB1 in one tab
and NB3 in another will hit this. Belongs in HARDWARE-GUIDE.md.

---

## F-06 — a "16 GB" T4 gives **14.6 GB** usable — **FIXED**

Colab reports `VRAM: 14.6 GB`, not 16. The `Qwen3.5-4B` bf16 checkpoint is **9.32 GB**
on the wire, so weights alone take ~62% of the card before any activations, LoRA state
or optimizer moments. HARDWARE-GUIDE.md says "Colab Free T4 16 GB" and budgets ~10 GB
for 4B bf16 LoRA — the headroom is thinner than documented. Whether it actually fits
is what NB3 decides; the number in the guide should say 14.6 GB either way.

---

## F-07 — the lab hardcoded **bf16**, and its default GPU has none — **FIXED**

**Severity: high.** Found by inspection during the Colab run, then fixed pre-emptively.

`sft_config_kwargs()` emitted `bf16=True` unconditionally and `load_base()` used
`dtype=torch.bfloat16`. The lab's **default tier is a free-Colab T4**, which is Turing
(sm_75) — **bfloat16 requires Ampere (8.0+)**. So the recommended path was configured
for hardware the recommended hardware is not.

This is the standard 2026 tutorial bug: every guide is written on an A100, where
`bf16=True` is correct, and the flag gets copied onto cards that cannot do it.

fp16 is not a drop-in swap either: its exponent range is far smaller, so training needs
**gradient scaling** to avoid underflow — which trainers enable only when told
`fp16=True`. The precision decision therefore has to reach the *training arguments*,
not just the model load.

**Fix.** New `labkit/device.py`: `describe()` / `precision()` / `torch_dtype()` /
`banner()` pick bf16 → fp16 → fp32 from the actual device capability, and
`sft_config_kwargs(precision=...)` allows an explicit override. Both flags are always
set, never both true. 7 new tests (63 → 70), including one asserting the flags are
never hardcoded and one that a T4-shaped device produces the explanatory note.

---

## F-08 — NB2/NB5 print nothing for tens of minutes — **FIXED**

**Severity: medium (usability, but it makes students kill good runs).**

`score_run()` prints only after a whole baseline finishes. On the free T4 the observed
gap between "Loading weights: 100%" and the first number was **>15 minutes** with zero
output. That is indistinguishable from a hang, and the documented remedy for a hung
Colab cell is to interrupt it.

**Fix.** `generate_batch()` now prints a per-batch line with elapsed time and ETA,
labelled by which pass is running (`(a) base + naive prompt/target`, `ft/regression`,
…). NB2 and NB5 pass labels through.

## F-09 — the published time budget is optimistic — **FIXED**

README claims NB2 ≈ 10 min and the core ≈ 80 min on a T4. Measured on free Colab:

| Stage | Claimed | Observed |
|---|---|---|
| NB1 | 2 min | **26 s** ✅ |
| model download (first run only) | not mentioned | **~70 s** for 9.32 GB |
| weight load | not mentioned | ~30 s |
| NB2 baseline (a) alone | — | **>15 min** (pre-fp16) |

**Measured after the fp16 fix**, from the new progress output:

```
[(a) base + naive prompt/target] batch 1/13    44s elapsed  ~523s left
[(a) base + naive prompt/target] batch 2/13    88s elapsed  ~483s left
```

**44 s per batch of 4 prompts ≈ 11 s/prompt** at `max_new_tokens=160`, 4B on a T4.
Extrapolating:

| Stage | README claim | Projected from measurement |
|---|---|---|
| NB1 | 2 min | **14–26 s** ✅ |
| NB2 (two baselines × 65 prompts) | 10 min | **≈ 23 min** |
| NB5 (one scoring pass) | 10 min | **≈ 12 min** |
| **core NB1–NB5** | **80 min** | **≈ 95–110 min** |

The structural cause is that the eval set is generated **three times** across the lab
(baseline a, baseline b, fine-tune). That is inherent to the three-baseline design and
is the right trade — but the README must say so, and `EVAL_LIMIT` should be presented
as the normal way to iterate rather than a hidden knob.

**Resolved 2026-08-20 with end-to-end measurements** (`docs/MEASURED-T4-2026-08-20.md`).
Final numbers, now published as **ranges** in README and HARDWARE-GUIDE:

| stage | was published | measured |
|---|---|---|
| NB2 | 10 ph | **17–23 ph** |
| NB3 | 25 ph | **15–25 ph** |
| NB4 | 35 ph | **45–60 ph** |
| NB5 | 10 ph | **21 ph** (now scores the contrasts too — F-22) |
| core | 80 ph | **100–130 ph** |

Ranges, not point estimates: the *same* 30-step config took **1456 s and then 1021 s**
on identical code, because a free T4 is shared and throttled. And the earlier projection
here was wrong in both directions — fp16 (F-15) sped up **generation** ~3.5× but did not
speed up training at all (48.5 s/step on both paths), while NB4 got *slower* once F-17
stopped the contrasts from being under-counted. Do not describe the fp16 fix as a
training speedup.

`EPOCHS` and `EVAL_LIMIT` are both documented in README as the supported time-box levers.

---

## F-10 — `assistant_only_loss=True` supervises **ZERO tokens** on the default model — **CRITICAL**

The single most damaging finding, and a **silent** one.

NB3 configured `assistant_only_loss=True` and handed training to TRL. TRL derives that
mask from `{% generation %}` markers in the chat template. **Qwen3.5's template has
none.** Result, measured by `scripts/check_mask_agreement.py`:

```
chat template exposes {% generation %} markers: False
labkit assistant-only : 11/31 tokens (35.5%)   '</think>\n\n{"intent": "doi_tra"}<|im_end|>\n'
TRL  assistant_masks  :  0/31 tokens ( 0.0%)   ''
VERDICT: FAIL — TRL would supervise NOTHING.
```

transformers emits a **warning, not an error**. Any pipeline that builds its batches from
that mask trains on nothing, and the loss curve still looks plausible.

> **Correction (2026-10-06, measured with TRL 1.10.0).** The 0/31 above is the
> *tokenizer* path (`apply_chat_template(return_assistant_tokens_mask=True)`), which is
> what this script measured — not an `SFTTrainer` run. `SFTTrainer(assistant_only_loss=True)`
> in TRL >= 1.10 first calls `get_training_chat_template()`:
>
> | base model | labkit (NB1) | tokenizer path | SFTTrainer path |
> |---|---|---|---|
> | `unsloth/Qwen3.5-4B` (lab default) | 11/31 | 0/31, warning only | **raises `ValueError`** (template not recognised) |
> | `Qwen/Qwen3.5-2B` (official template) | 9/31 | 0/31, warning only | **patches the template → 13/31**, incl. the empty `<think>\n\n</think>\n\n` |
>
> So the trainer is loud, not silent — the original "training completes" sentence was an
> inference that did not hold. The fix below still stands, for a sharper reason: the flag
> either crashes or supervises a mask that is not the one NB1 proved.
> `scripts/check_mask_agreement.py` now prints both paths.

This is precisely the class of bug the deck spends §17.2 and §22 on — *"no error, a
plausible loss curve, and a broken model"* — reproduced by the lab's own default
configuration. NB1 proves the mask is correct and then NB3 threw that proof away and
trusted a library flag.

**Fix.** Stop trusting the flag. NB3 now trains on the **exact mask NB1 verified**:
`data.to_training_dataset()` pre-tokenizes with `build_example()`, so `input_ids` and
`labels` are the ones the student decoded and asserted on. `assistant_only_loss` is not
set at all.

Consequence, stated honestly on the slide-facing side: pre-tokenized labels are
incompatible with `packing`, so packing is off for this path. Deck §17.3's point
(packing is free only when boundaries are respected) still stands — here the *mask's
correctness* outranks the throughput, and the lab says so rather than quietly keeping a
flag that does nothing.

`scripts/check_mask_agreement.py` ships with the lab so students can run this check
against any base model they swap in.

---

## F-11 — the format scorer was stricter than the target scorer — **FIXED**

`triage_field_accuracy()` recovered a `{...}` block embedded in prose;
`has_required_keys()` accepted only bare or fenced JSON. A model answering
`"Day la ket qua: {...}"` therefore scored on **target** but **0.000 on format** — a
formatting failure that did not happen. Two scorers disagreeing about what counts as
JSON makes both numbers untrustworthy, and `format` is one of the four graded groups.

Both now share `_parse_json_loose()`. +2 tests (76 total).

Surfaced by the first real T4 measurement: `(a) base + naive prompt  target=0.000
format=0.000`. Those particular zeros turned out to be genuine — a naive prompt with no
schema produces prose, not JSON — but checking *why* they were zero exposed the
inconsistency.

## F-12 — observation: the optimized prompt is ~3× faster, not just more accurate

Measured on the T4: baseline (a) ran at **44 s/batch**, baseline (b) at **15 s/batch**.
Same model, same prompts, same decode settings. The optimized prompt tells the model to
emit only JSON, so it emits ~20 tokens and stops; the naive prompt lets it ramble to the
160-token cap.

Worth teaching: prompt engineering bought a **3× latency win before any fine-tuning**,
which sharpens deck §21's point that baseline (b) is a real bar — it is better on the
target metric *and* cheaper to serve.

---

## F-13 — Colab's preinstalled **torchao 0.10.0** blocks NB3 entirely — **FIXED**

**Severity: high — this is a hard stop on the documented default path.**

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

Raised inside `get_peft_model()` → `_create_new_module()`. Colab preinstalls
torchao 0.10.0; peft 0.20 / transformers 5.15 require >0.16. **Nothing in
`requirements.txt` pulled torchao in**, so pip never upgraded it and the stale
preinstalled copy won. NB3 failed after 51 s.

This is the failure mode a locally-tested lab cannot catch: the machine that broke it
is the one with *extra* packages already installed, not missing ones. `pip install -r`
succeeds; the conflict only appears at import time inside a third-party call.

**Fix.** Explicit `torchao>=0.16` in `requirements.txt` and in both Colab bootstraps,
with a comment saying why a package nothing imports directly is pinned.

### Also confirmed on Colab: F-10, independently

The same cell ran `scripts/check_mask_agreement.py` on the real T4:

```
labkit assistant-only : 11/31 tokens (35.5%)
TRL  assistant_masks  :  0/31 tokens ( 0.0%)
VERDICT: FAIL — TRL would supervise NOTHING.
```

(That verdict line is the old wording; it describes the tokenizer path only — see the
correction under F-10.) Identical to the local reproduction — the F-10 fix is aimed at a real defect on the
real platform, not an artifact of the Mac.

---

## F-14 — `padding_free=True` was unconditional; on the default tier it is both unsafe and useless — **FIXED**

NB3 died at trainer construction:

```
ValueError: When `padding_free=True` without packing, `max_length` is not enforced.
```

preceded by two warnings that matter more than the error:

```
Padding-free training is enabled, but the attention implementation is not set to a
supported Flash Attention variant ... only the following are known to reliably support
this: flash_attention_2, flash_attention_3, ...
Using a batch size of 1 annihilates the benefits of padding-free training.
```

Three separate problems in one flag:

1. **Unsafe here.** Padding-free flattens a batch into one sequence. Without a kernel
   that understands the boundaries, attention can run *across* them — literally deck
   §17.3's warning ("packing is free only when sequence boundaries are respected")
   applied to packing's sibling flag. **FlashAttention-2 needs Ampere (sm_80+), so a T4
   cannot have it at all.**
2. **Useless here.** The T4 tier uses `per_device_train_batch_size=1`. There is no
   inter-sequence padding to remove in a batch of one.
3. **Incompatible with the F-10 fix.** Pre-tokenized labels force `packing=False`, and
   TRL rejects `padding_free` + `max_length` without packing.

**Fix.** `device.supports_padding_free(batch)` requires an importable FlashAttention
kernel **and** batch ≥ 2. When it *is* available, `max_length=None` is passed —
honest, because `build_example()` already truncates — rather than silencing the check.
+3 tests (79 total).

**Meta-point worth keeping:** the deck teaches `packing` + `padding_free` as the §17.3
recommendation. On the hardware the lab actually recommends, neither is available. The
lab now says that out loud instead of setting flags that do not apply.

---

## F-15 — my own F-07 fix was wrong: `torch.cuda.is_bf16_supported()` returns **True on a T4** — **FIXED**

Caught by reading NB3's config dump during the training run: `"bf16": "True",
"fp16": "False"` — on a T4, after supposedly fixing exactly this.

The trap is in torch's signature:

```python
def is_bf16_supported(including_emulation: bool = True):
    ...
    if torch.cuda.get_device_properties(device).major >= 8:
        return True
    if not including_emulation:
        return False
    return _check_bf16_tensor_supported(device)     # <- Turing passes this
```

**The default is `including_emulation=True`**, and it only checks that a bf16 tensor can
be *created*. Turing can, by emulation. So the "is bf16 supported?" API answers **yes**
on hardware with no bf16 units, and training proceeds emulated at a large speed
penalty while truthfully reporting `bf16=True`.

This very likely explains F-09's slow generation: the whole first NB2 run was emulated.

**Fix.** Compute capability ≥ 8.0 is the real test, with
`is_bf16_supported(including_emulation=False)` as a secondary check where the kwarg
exists. +1 test that fakes a T4 (`capability 7.5`, `is_bf16_supported() → True`) and
asserts we still choose fp16. **80 tests.**

**The lesson worth carrying:** a capability API that answers "yes, by emulation" is
worse than no API. Three of this lab's eighteen findings (F-10, F-15, F-16) are the same
shape — a library saying *yes*, or saying nothing much, about something it is not
really doing.

---

## F-16 — `warmup_ratio` no longer exists, so every run trained with **no warmup** — **FIXED**

Visible in NB3's own config dump, one line above the hyper-parameters:

```
⚠ TRL không nhận: ['warmup_ratio']
```

`filter_kwargs` did its job and said so. Nobody read it — including me, across a full
NB3 run.

transformers v5 / TRL 1.10 removed the field. Measured on the Colab VM:

```python
[f.name for f in dataclasses.fields(SFTConfig) if "warm" in f.name]
# -> ['warmup_steps']          # no ratio field at all
```

So `sft_config_kwargs` asked for a knob that does not exist, the filter dropped it with a
warning, and training ran with a cosine schedule and **zero warmup** while printing a
config the reader would assume included it.

**Fix.** Convert the deck's 10% into an absolute step count via a new
`train.planned_steps()`, and emit `warmup_steps`. +3 tests.

**The lesson.** F-16 is the *cost* of the version-defensive design that F-10 justified.
Dropping unknown kwargs converts a loud `TypeError` into a quiet behaviour change; the
warning is only a safeguard if something fails when it fires. A stricter contract —
"warn, and fail if the dropped key is one the recipe depends on" — would have caught
this at step 0 instead of after a 25-minute run.

---

## F-17 — the contrasts were trained **twice as long** as the baseline they are compared against — **FIXED**

`config.py` stated the requirement and then broke it:

```python
# Every contrast run gets the SAME number of optimizer steps as the NB3 baseline
# slice, otherwise the comparison measures wall-clock, not configuration.
CONTRAST_MAX_STEPS = 60
```

NB3 does not run 60 steps. It runs an *epoch* budget — 2 epochs × ⌈225/16⌉ — which the
real T4 run printed as `100% 30/30`. So each of NB4's three contrasts got **2× the
training** of the `correct` run it is measured against, and NB4's own prose had already
absorbed the bug as a workaround:

> `correct` từ NB3 chạy nhiều step hơn — **đừng so loss trực tiếp với nó** … hãy chạy
> lại `correct` với `max_steps=CONTRAST_MAX_STEPS` (một dòng, **~10 phút**)

Two things wrong there. The comparison the notebook is *for* was being deferred to an
optional manual re-run; and "~10 phút" is the estimate the constant was sized against —
measured at 48.5 s/step, 60 steps is **48 minutes** per contrast, so NB4 alone was a
**~145 minute** stage inside a lab advertised at ~80.

**Fix.** Both sides derive the budget from the same recipe
(`train.planned_steps(len(train_ds), TIER, CONTRAST_EPOCHS)`), so the autopsy varies one
variable instead of two, the manual fix-up disappears, and NB4 drops to ~73 min. +2 tests.

**Caught by arithmetic, not by running it** — NB4 had never executed. The measured
48.5 s/step from the one completed NB3 run is what made the 60 visibly wrong.

---

## F-18 — the generated Colab notebooks were stale and still shipped the F-13 crash — **FIXED**

`colab/*.ipynb` is generated from `notebooks/*.py` by `scripts/build_colab.py`. The F-13
fix added `torchao>=0.16` to that script's BOOTSTRAP — but the notebooks were never
regenerated and committed, so all six still carried the old bootstrap.

Consequence: the RUN_ALL path was fixed, while the **per-notebook Colab badges in the
README** — the entry point a student following the lab notebook-by-notebook actually
uses — still walked into the torchao 0.10 `ImportError` at `get_peft_model()`.

Found only because F-17 forced a regeneration and the diff showed an unrelated line
changing. **Generated artifacts that are committed need a build step in the gate**, or
they drift silently from their source.

---

## F-19 — Colab never re-reads the notebook, so a long-lived tab runs *old* code — **ENVIRONMENT, not the repo**

Cost me an 8-minute pipeline run and looked exactly like a regression.

The restarted run died at NB3 with the F-13 error — `Found an incompatible version of
torchao. Found version 0.10.0` — a bug fixed two days earlier, in a VM where cell 1 had
just run green.

Cell 1 was **stale**. Colab fetches notebook source from GitHub *once*, when the URL is
opened, and never again: not on reconnect, not on a new runtime, not when the repo moves.
This tab was opened in the previous session, before `82bda58` added the pin, so the cell
it ran was the pre-fix list. It installed the seven packages it knew about, reported
success in 4 s, and left `torchao 0.10.0` in place.

Three symptoms that make this hard to read correctly:

* the cell **succeeds** — nothing in its output hints the source is old
* `git pull` inside the cell updates the *repo*, which makes the environment look fresh
  while the cell doing the pulling is itself out of date
* the printed `commit :` line reports the freshly pulled HEAD — **a stale cell can print
  a current commit hash**, which is actively misleading

**Not a repo defect** — the committed notebook is correct, and any student opening the
badge gets it. Worth a README line anyway, because "reconnect and re-run" is the natural
reaction to a disconnect and it silently preserves the stale source. The reliable move
after any repo change is to reload the browser tab, not just the runtime.

---

## F-20 — the dependency list existed in **three** hand-synced copies — **FIXED**

`requirements.txt`, `scripts/build_colab.py`'s BOOTSTRAP, and `colab/Lab21_RUN_ALL.ipynb`
cell 1 each carried their own copy of the same pins. Keeping them in sync was manual, and
it had already failed twice:

* **F-18** — BOOTSTRAP got `torchao>=0.16`, the generated notebooks did not
* **F-13's recurrence** — the pin reached `requirements.txt` and the two bootstraps on
  different days, which is what made F-19 possible at all

The failure mode is nasty because a bootstrap missing a pin **does not fail at install
time**. It exits 0, and the run dies ten minutes later inside `get_peft_model()`, with a
traceback pointing at peft rather than at the install cell that actually caused it.

**Fix.** Both bootstraps now `pip install -q -r requirements.txt`. The repo is cloned
before the install, so the file is available; torch is preinstalled on Colab and
`requirements.txt` pins it compatibly, so that line is a no-op. One source of truth.

---

## F-21 — the README's Quick Start pointed at a notebook that does not exist — **FIXED**

```
### Colab (khuyến nghị)
Mở `colab/Lab21_T4.ipynb` → Runtime → Change runtime type → T4 GPU → Run all.
```

There is no `colab/Lab21_T4.ipynb`. The directory holds `Lab21_01`..`Lab21_06` and
`Lab21_RUN_ALL`. The **first instruction in the lab** named a file that was never
generated — a leftover from an earlier naming scheme that no test covers, because nothing
verifies that documentation references resolve.

Found by grepping the README for "colab" while fixing something else, not by any check.

**Fix.** Point at `Lab21_RUN_ALL.ipynb` as a clickable Colab link, and fold the F-19
reload warning in next to it — that is where a student is standing when it bites them.

---

## F-22 — the misconfig autopsy was decided by **training loss**, the thing the lab calls Lỗi #3 — **FIXED**

NB4 is the notebook the lab titles *phần quan trọng nhất*. It trains three deliberately
wrong adapters, saves them, prints a table, and asks:

> `attn_only` có **cùng số tham số huấn luyện** với `correct`. Nó thắng hay thua?

The only number in that table capable of answering is `final_loss` — a **training**
loss. Nothing ever scored those adapters on the target task: NB5 evaluated
`adapters/correct` and stopped. So the lab whose own NB4 header names *"chấm bằng
perplexity"* as **Lỗi #3** was settling its central question with a proxy metric.

It can also point the wrong way. 225 examples × 30 steps, and `attn_only` runs at
r=283 to match the parameter budget. A memorising adapter can drive train loss *below*
`correct` while being worse at the task — in which case the table hands the student the
opposite of the lesson, with the lab's authority behind it.

**Fix.** NB5 §4 scores all three contrasts on target + format (same scale as `correct`)
and writes `results/autopsy.json`; `make verify` requires it. Contrasts skip the
regression sweep — the graded four-group gate stays `correct`-only, so verdict semantics
do not move. NB4 relabels the column as a training loss and points at NB5 for the
ranking; the REPORT template gains a target column; rubric 2.5 says so explicitly.

One trap avoided in the same change: `qlora` learned against a 4-bit base, so
`score_adapter` takes `load_in_4bit` from the run's own spec. Scoring it on the fp16
base would have measured a base/adapter mismatch and labelled the damage *"what QLoRA
costs you"*.

**Caught by reading NB5 while NB4 was still running.** The bug is invisible from NB4
alone — it lives in what the *next* notebook does not do.

---

## F-23 — TRL casts LoRA weights to **bf16** on a GPU that has none — **FIXED**

The pipeline died 55 minutes in, at step 0 of the `qlora` contrast:

```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
                     not implemented for 'BFloat16'
```

raised from fp16's `GradScaler`. Every knob that could plausibly cause it was already
correct — `bnb_4bit_compute_dtype=device.torch_dtype()`, `fp16=True`, `bf16=False`.
Hand-replicating the PEFT setup produced **fp32** trainables and no bf16 at all, which
disproved the obvious hypothesis rather than confirming it.

So `scripts/probe_precision.py --trainer qlora` builds NB4's actual `SFTTrainer` and
asks it:

```
model handed TO SFTTrainer : float16=178  uint8=248     <- zero bf16
model handed BACK by it    : bfloat16=496               <- every LoRA weight
fp16 = True   bf16 = False   precision() = fp16   scaler = GradScaler
```

**TRL casts the adapters to bf16 regardless of the device and regardless of the flags
in the `SFTConfig` it was handed.** A T4 is Turing; it has no bf16 hardware at all, so
the cast is not merely unsupported, it is meaningless. `GradScaler.unscale_` then hits a
CUDA kernel with no `BFloat16` overload and the run is over.

This is precisely the failure `labkit/device.py` was written about — *"tutorials hardcode
bf16 because every 2026 tutorial is written on an A100"* — except the hardcoding is
**inside the training library**, downstream of both the quantization config and our own
flags. It cannot be configured away.

**Fix.** `train.align_trainable_precision()` corrects the model TRL returns: trainable
bf16 → fp32, which is what mixed precision wants anyway and what
`prepare_model_for_kbit_training` produces unaided. Called in NB3 and NB4 after
constructing the Trainer and before `.train()` (the optimizer is not built until then).
No-op on real bf16 hardware. NB4 prints when it fires. +3 tests; the notebook gate is
verified to fail when the call is removed.

**Only reachable through the 4-bit path**, which is why NB3, `attn_only` and `wrong_lr`
all trained clean and this waited until the last of four runs to appear.

---

## F-24 — NB4 restarted from zero after a crash in its third run — **FIXED**

F-23 destroyed 55 minutes of finished work: `attn_only` and `wrong_lr` had trained and
saved, and the only way to reach `qlora` again was to retrain both. On Colab — where
runtimes are dropped for idling, for tab closes, and for nothing in particular — a
75-minute notebook with no resume is a notebook students will not finish.

**Fix.** An existing `adapters/<key>/adapter_model.safetensors` counts as done.
`FORCE_RETRAIN=1` redoes everything, `ONLY=qlora` redoes one, deleting a directory redoes
that one. The contrast table reads `runs.csv` rather than this session's rows, so a
resumed run still prints all four. Verified on the VM: both finished contrasts skipped,
`qlora` alone re-ran.

---

## F-25 — `.env` was never read by anything — **FIXED**

**Severity: high.** README, `HARDWARE-GUIDE.md`, `.env.example` and `config.py`'s own
docstring all state that editing `.env` selects the tier. Nothing parsed the file.
`get_tier()` consulted `os.environ` only, and no dependency or code path populated it:

```
$ grep -c dotenv requirements*.txt src/labkit/*.py     # 0
$ echo COMPUTE_TIER=LAPTOP > .env && python notebooks/01_data_and_mask.py
tier=T4  model=unsloth/Qwen3.5-4B          # <- not LAPTOP
```

`MASK_MODE` and `EPOCHS` were ignored the same way — including the `EPOCHS` lever F-24
documents as *the* supported way to time-box a run. A student on an 8-12 GB laptop who
follows the documented instruction gets the T4 tier's 4B model and OOMs ten minutes
later, having done exactly what they were told. Invisible on Colab, because there the
documented default and the real default happen to agree.

**Fix.** `labkit/env.py` — a dependency-free `.env` parser loaded from
`labkit/__init__.py` before any submodule reads the environment. `requirements-cpu.txt`
is the slice every student can install with no GPU, and `python-dotenv` is not worth a
line in it for thirty lines of parsing. An already-set variable always wins, so
`EVAL_LIMIT=8 make pipeline` and CI still override the file. +5 tests.

---

## F-26 — `colab_run.py` block-buffered the child's stdout, defeating F-08 — **FIXED**

**Severity: medium.** The docstring claims "Output is NOT captured, so Colab streams it
live and a long training run does not look like a hang." The children were spawned with
neither `-u` nor `PYTHONUNBUFFERED`, so CPython block-buffers stdout whenever it is a
**pipe** — which is what Colab, `tee`, and every redirect hand the child.

Measured through a pipe, first line out of a 3-line script printing once per 1.2 s:

```
before fix: first line reached the pipe after 3.61s   (i.e. only at process exit)
after  fix: first line reached the pipe after 0.01s
```

Observed live: >3 minutes of total stdout silence during NB2 while stderr flowed
normally — the asymmetry is the signature of block buffering rather than a hang. The
casualties are F-08's per-batch ETA lines, added specifically so students stop killing
healthy runs, and they never reached the student. **Fix.** `-u` plus
`PYTHONUNBUFFERED=1` on the child.

---

## F-27 — `verify.py` green-lit smoke runs — **FIXED**

**Severity: medium.** NB2 already writes `smoke_mode` and `eval_limit` into
`baselines_frozen.json`, and `.env.example` states that a submitted run must leave
`EVAL_LIMIT` unset. `verify.py` read neither key, so an 8-item run printed
`Ready to submit.` **Fix.** A `full eval set used` check that FAILs on `smoke_mode`,
naming the item count and how to re-run.

> Follow-up for the maintainer: `colab/Lab21_RUN_ALL.ipynb` defaults its widget to
> `EVAL_LIMIT = "8"`, so the documented one-click path now produces a run this gate
> correctly rejects. Left as-is deliberately — whether the default Colab run should be
> submittable (~95-110 min) or fast (~15 min) is a product call, not a bug fix.

---

## F-28 — NB6 scored on 20 items in a full run — **FIXED**

**Severity: low.** `notebooks/06_merge_and_serve.py` defaulted `EVAL_LIMIT` to `"20"`
while every other notebook uses `0` = full set, so the merge no-regression assert
silently ran on 20 of 50 items even in an unabridged run — and the assert is the whole
point of the notebook. **Fix.** Default `0`, slice only when set, print the count.

---

## F-29 — the graded verdict was decided by string-matching its own prose — **FIXED**

**Severity: low (correct today, one reworded sentence from being wrong).**

```python
passed=not any(r.startswith(("target", "general")) for r in reasons)
```

`regression_gate` computes both numeric conditions, throws them away, formats them into
human-readable sentences, and then recovers the verdict by inspecting the first word of
those sentences. Rewording a message — or translating it, in a lab written in
Vietnamese — silently flips a pass to a fail. **Fix.** Derive `passed` from the
booleans that were already computed. +2 tests.

---

## F-30 — `MASK_MODE` is inert on the shipped corpus, and a comment said otherwise — **FIXED (documented)**

**Severity: medium (pedagogical, not behavioural).**

`masked-think` and `response-only` produce a mask **identical** to `assistant-only` on
the shipped data — 37/188 supervised tokens for all three, on the real 0.8B tokenizer.
Two independent reasons:

1. All 250 training answers are bare JSON; `<think>` appears zero times across all four
   `data/*.jsonl` files.
2. More subtly, `add_generation_prompt=True` already emits the *complete* empty block:

```
prefix: '...<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

so the supervised span starts *past* `</think>` and `_skip_reasoning_chars` has nothing
left in `[start, end)` to skip. Its docstring claimed the opposite — "Qwen3.5 emits
`<think>\n\n</think>\n\n` even for a non-reasoning answer, so this fires on ordinary
data too". It does emit it, but into the *prefix*, so the skip never fires on ordinary
data. The second reason is the load-bearing one: even a corpus with traces in `output`
needs them inside the assistant content, not merely present in the file.

Knock-on: NB5's `valid_trace_rate` is structurally 0.0 for every run — the model is
never trained on traces and generation runs `enable_thinking=False`.

**Fix.** Corrected the docstring, documented the caveat in `.env.example`, and made
`to_training_dataset()` warn when a think-mode is selected against a corpus that cannot
exercise it. Deliberately **not** fixed by adding traces to the corpus: that would move
`data/checksums.json` and trip the eval-drift gate for everyone. +3 tests.

---

## F-31 — the lab trained on one prompt and scored on another; **every adapter got 0.000** — **FIXED**

The first end-to-end NB1→NB5 run. All four adapters scored `target=0.000 format=0.000`
— identical to the **untrained** baseline (a) — while every training loss curve looked
healthy:

| run | final loss | target | format |
|---|---|---|---|
| `correct` | 0.0549 | **0.000** | 0.000 |
| `attn_only` | **0.0531** | 0.000 | 0.000 |
| `wrong_lr` | 0.0903 | 0.000 | 0.000 |
| `qlora` | 0.0670 | 0.000 | 0.000 |

`scripts/diagnose_generation.py` printed the two renders side by side:

```
TRAINED ON : user: "<full schema + enum list>\n\n<ticket>"   -> JSON     (no system msg)
EVALUATED  : system: "Phân loại ticket sau."  +  user: "<ticket>"
```

`to_messages()` folded the entire instruction — the four key names, every enum value,
*"chỉ trả về JSON"* — into the **user** turn, and emitted no system message at all.
Evaluation then asked for the same task from a prompt containing none of that, in a role
structure the model had never seen. So the fine-tune answered the question it was
*actually* asked — a vague instruction with no schema — in fluent Vietnamese prose:

> `'Dựa trên nội dung của ticket, đây là một yêu cầu liên quan đến quy trình hậu mãi
> cụ thể. Dưới đây là phân loại chi tiết: * **Loại ticket:** ...'`

**Diagnosed, not guessed.** Latency said it first: 10418 ms/sample against baseline (a)'s
3215 ms. A model that learned nothing is *fast* and wrong; an off-distribution model is
*slow* and wrong, because it rambles to the token cap. The diagnostic then separated the
three candidate explanations — inert adapter (`disable_adapter()` comparison), think-
scaffold mismatch (`enable_thinking` sweep), failed transfer — and only one survived.

NB5's own comment states the intent correctly:

> *the fine-tune is evaluated WITHOUT the long optimized prompt — that is the point of
> fine-tuning: the behaviour moved into the weights, so the prompt can shrink*

That ambition is right. The prompt was shrunk at **evaluation** time without ever
training the shrunk form.

**Fix.** `to_messages(record, system=NAIVE_PROMPT)` now emits the shape evaluation sends:
short system prompt, bare ticket in the user turn, JSON answer. What the model has to
internalise is the label space — precisely what fine-tuning is supposed to buy here.
`system=None` reproduces the old shape so NB1 can show both. `NAIVE_PROMPT` and
`OPTIMIZED_PROMPT` move to `config.py` and `generate.py` imports them: two copies of a
prompt shared by training and eval is how this got in.

`data.prompt_alignment()` returns both renders and whether the eval prompt is a **prefix**
of the training text. NB1 proves the *mask*; this proves what the model **conditions on**.

**A correct mask cannot catch this.** The mask was right the entire time — F-10's proof
still passes. The lab had a rigorous, verified answer to "is the right span supervised?"
and no answer at all to "is this the prompt we will actually send?" Both notebooks that
followed inherited the gap, and the four-group gate dutifully reported FAILED for a
reason no student could have diagnosed from the artifacts.

**Why nothing caught it sooner:** NB5 had never completed a single run before today.

### Verified on GPU — smoke configuration

Re-ran `EPOCHS=1 EVAL_LIMIT=8 FORCE_RETRAIN=1` (the lab's own documented fast-iteration
mode) on the same T4:

| | target | regression | format | latency |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.750 | 0.000 | 3280 ms |
| (b) base + optimized prompt | 0.688 | 0.750 | 1.000 | 1042 ms |
| **(c) LoRA fine-tune** | **0.500** | 0.750 | **1.000** | **1566 ms** |

* **format 0.000 → 1.000** — the fine-tune emits parseable 4-key JSON. That is the claim.
* **latency 10418 → 1566 ms/sample**, a 6.6× drop, because the model now produces JSON
  and stops instead of rambling to the 160-token cap. The tell arrived before the score:
  `[ft/target] done: 8 prompts in 13s`, against ~83 s for the same sweep while broken.
* **regression unchanged at 0.750** — no catastrophic forgetting.

The gate still reports **FAILED**: 0.500 < (b)'s 0.688. That is a *legitimate* outcome at
this budget, not a residual bug — 15 optimizer steps instead of 30, scored on 8 items
where one item is worth 0.125. **Whether a correctly-configured fine-tune can actually
beat baseline (b) on this task is still unmeasured**, and it needs the full run
(`EPOCHS=2`, 50 items, all four adapters retrained). Do not quote 0.500 as the lab's
result; quote it as proof that the pipeline transmits learning to the eval at all.

---

## Verified working

| Check | Where | Result |
|---|---|---|
| `git clone` + `pip install` bootstrap | Colab T4 | ✅ `GPU: Tesla T4`, ~30 s |
| GitHub → Colab notebook launch | Colab | ✅ renders, Vietnamese intact |
| Tokenizer download + chat template | both | ✅ |
| `thinking_survives()` on real template | both | ✅ "reasoning preserved" |
| NB1 end-to-end, real 4B tokenizer | local | ✅ 39/188 supervised, both asserts green |
| Unit tests | both | ✅ 63 passed |
| `requirements.txt` resolution | local (py3.14) | ✅ dry-run clean |
| `verify.py` smoke + full | **Colab T4** | ✅ incl. integrity checks on real artifacts: `baseline (b) beats (a) 0.000 -> 0.760`, prompt SHA unmodified, eval checksums intact, unfilled REPORT template correctly FAILED |
| **NB2 end-to-end, real 4B on T4** | Colab T4 | ✅ **1006 s**, `baselines_frozen.json` written |
| **NB3 config + dataset build** | Colab T4 | ✅ 12 target modules resolved, 32.46 M trainable, matched rank r=283 (budgets within 0.03%), 225 rows, **9014/42101 tokens supervised (21.4%)**, assert passed |
| **NB3 training started** | Colab T4 | ✅ **30 steps, 48 s/step** — reached `1/30 [00:48<23:18]` before the browser extension disconnected |

---

## Measured results (free Colab T4, `unsloth/Qwen3.5-4B`, full 50-item eval)

| Run | target | regression | format | latency |
|---|---|---|---|---|
| **(a)** base + naive prompt | 0.000 | 0.724 | 0.000 | 11331 ms |
| **(b)** base + optimized prompt | **0.760** | 0.724 | **1.000** | **3775 ms** |
| (c) LoRA fine-tune | *not reached* | | | |

**The lab's central design validated itself empirically.** Baseline (b) is a genuinely
hard bar — 0.760 target with perfect JSON compliance and 3× lower latency than (a).
A fine-tune has to beat *that*, which is exactly the discipline deck §21 argues for and
the opposite of the old lab's perplexity-vs-nothing comparison.

Deck §7.4 also became a lab artifact: NB3 printed the real model's
`layer_types: {linear_attention: 24, full_attention: 8}` — the 3:1 hybrid interleave,
read off the checkpoint the student is fine-tuning.

---

## First complete NB1 → NB5 execution (RTX 3060 12 GB, `Qwen/Qwen3.5-0.8B`)

NB3 completion, NB4 and NB5 had never run to the end — the Colab extension dropped
mid-training every time. On a local Ampere card, on the merged tree (F-22…F-31), all
five stages completed **exit 0 in 33.9 minutes**, full 50-item eval, `EVAL_LIMIT` unset:

```
nb1   9s     nb2  126s     nb3  446s     nb4  1308s     nb5  147s     total 2036s
```

This is the CPU tier's 0.8B model run on a GPU, so absolute scores are not comparable
to the 4B Colab numbers above. The *orderings* are the result.

### The three-baseline table

| run | target | regression | format | latency |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.644 | 0.000 | 1471 ms |
| (b) base + optimized prompt | 0.495 | 0.644 | 1.000 | 404 ms |
| **(c) LoRA fine-tune** | **0.990** | **0.067** | **1.000** | 626 ms |

**Verdict: FAILED — on catastrophic forgetting alone.** `target_delta +0.495`,
`regression_delta −0.578` against a 0.020 tolerance.

This is deck §14.3 in its most vivid form. Training on `bare ticket → JSON` with no
instruction taught the model that *any* input means "emit triage JSON", so it now
answers general-knowledge questions with JSON too. The better the task fit, the more
total the collapse. The fix is the one §14.3 names: 1–5% replay data.

Worth recording alongside it — the same adapter measured before the F-31 prompt change,
i.e. with the instruction still in the eval prompt, scored **target 1.000 / regression
0.522**. Nearly the same task score, an order of magnitude more generality retained.
The shrunk prompt gives the model no signal that this is *a* task rather than *the*
task. Anyone tempted to close F-31 by shrinking the prompt further should read that
pair of numbers first.

### The autopsy, on the target metric (F-22)

Four runs, one shared step budget (58), one shared parameter budget (10,822,656):

| run | r | final_loss | **target** | format | VRAM |
|---|---|---|---|---|---|
| correct | 16 | 0.3944 | **0.990** | 1.000 | 3.07 GB |
| attn_only | 271 | 0.4340 | 0.935 | 1.000 | 3.08 GB |
| qlora | 16 | 0.4243 | 0.930 | 0.980 | **2.29 GB** |
| wrong_lr | 16 | 1.5426 | **0.325** | 0.995 | 3.08 GB |

Both deck claims reproduce. §10.2: `correct` beats `attn_only` at a matched parameter
budget, so placement beats rank. §10.3: a full-fine-tune learning rate is the single
most destructive knob — `wrong_lr` collapses to 0.325 with 4x the training loss.
§12: 4-bit buys 25% less VRAM for a small but real accuracy cost.

**F-31 was masking F-22's evidence.** Before the prompt fix, `attn_only` had the *lower*
training loss (0.0741 vs `correct`'s 0.0827) — judged on loss you would have shipped
Mistake #1. With the schema in the prompt the task was near-copying, and a rank-271
adapter concentrated in 6 layers memorised it best. Once the model had to internalise
the label space, the ordering flipped and now agrees with the target metric. Two
defects were interacting.

### Hybrid attention sharpens §10.2 beyond what the deck claims

On this base, `q_proj`/`v_proj` exist only on the 6 `full_attention` layers; the other
18 are `linear_attention` with `in_proj_*`/`out_proj`. Attention-only placement
therefore reaches **12 modules across 6 of 24 layers**, and `matched_rank()` climbs to
**r=271** — a 17x rank increase buying zero extra layer coverage, at a parameter budget
matched to 0.0000%.

---

### The regression collapse is real forgetting, not prompt-shape sensitivity

Worth ruling out explicitly, because it decides whether §14.3's remedy is the right
one. Training renders a system turn (`NAIVE_PROMPT`) + a bare-ticket user turn; NB5's
regression probe sends `system=None` + the question, a shape the fine-tune never saw.
Base and fine-tune are measured identically so the comparison is fair either way — but
if the drop were shape-driven, replay data would be the wrong fix.

| model | `system=None` (what NB5 sends) | `system=NAIVE` (training shape) |
|---|---|---|
| base | 0.6444 | 0.5333 |
| fine-tune | **0.0667** | **0.0000** |

Matching the training shape makes the fine-tune *worse*, not better. It answers "what is
the capital of Vietnam?" with `{"intent": "hoi_thong_tin", "urgency": "thap", ...}`. The
collapse is total and unconditional, so **§14.3's 1-5% replay data is the correct
prescription** and the alternative explanation is ruled out rather than assumed away.

Caveat on the bar itself: the 0.8B base answers that same question with *"Thành phố thủ
đô của Việt Nam là **Hàn Quốc**"* — "the capital of Vietnam is South Korea". A 0.644
regression baseline on this tier is partial keyword credit on weak output, not a model
worth protecting. On the 4B tier the bar should be meaningfully higher.

---

## NB6 verified (first run)

NB6 had never been executed in any session. On the merged tree, with all four adapters
on disk, it passes end to end:

```
trước merge: 0.9900
sau merge:   0.9900   (Δ +0.0000)
adapter đang nạp: ['correct', 'attn_only', 'qlora']
```

`results/merge_check.json`: `{"before_merge": 0.99, "after_merge": 0.99, "delta": 0.0,
"tolerance": 0.01, "n": 50}`. The merge is numerically exact — `W = W₀ + (α/r)·BA` in
bf16 costs nothing measurable here — and `set_adapter()` hot-swaps all three adapters on
one loaded base, each still emitting well-formed JSON.

`n: 50` rather than 20 is F-28: this assert used to run on 20 of 50 items even in a full
run, so a post-merge regression confined to the tail would have gone unseen.

---

## Not verified

* The **T4 / fp16 path** end-to-end. The run above is Ampere (sm_86) using native bf16,
  so `device.precision()`'s fp16 branch remains unit-tested but not exercised on real
  Turing hardware.
* `unsloth/Qwen3.5-4B` end-to-end. NB2 completed on it (see above); NB3→NB5 on the 4B
  model have still not run to completion on a T4.
