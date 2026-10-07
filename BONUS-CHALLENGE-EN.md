# Bonus Challenges — Lab 21 (English)

> Do these after `make verify` is green. Each one connects to a **new** section of the
> 2026 deck (Part A: pretrain → mid-train → post-train → optimizers → architecture).
> Record results in the Appendix of `submission/REPORT.md`.

---

## B1 — Merge & multi-adapter serving (+3) · deck §23

Run `make nb6`. Required: `results/merge_check.json` showing post-merge score does not
drop by more than 0.01, and ≥2 adapters hot-swapped on one loaded base.

**Question:** merging gives zero inference overhead — what do you give up? When would you
keep adapters separate anyway?

---

## B2 — Your own domain dataset (+3) · deck §17

≥200 high-quality examples. Requires `data/CUSTOM_DATASET.md`: source · collection method
· decontamination · **why this data is distributionally new** relative to what the base
model already saw (deck §3.3).

200 careful examples usually beat 2,000 scraped ones — 2026 bases are saturated on
generic web text.

---

## B3 — Reasoning-trace collapse (+4) · deck §17.5 ⭐ hardest

Fine-tuning a reasoning model on ordinary Q→A data **destroys its reasoning while task
accuracy keeps rising**. No familiar metric catches it.

```bash
MASK_MODE=assistant-only make nb3 && make nb5   # record valid_trace_rate
MASK_MODE=response-only  make nb3 && make nb5   # record valid_trace_rate
```

| MASK_MODE | target | **valid_trace_rate** | regression |
|---|---|---|---|
| assistant-only | | | |
| response-only | | | |

**Question:** did `target` rise while `valid_trace_rate` fell? If you had only looked at
`target`, would you have noticed?

> The direction is **model-dependent** — the source study found empty-think catastrophic
> for Qwen3-8B but protective for Llama-R1-8B. Do not generalize from one model.

---

## B4 — A *controlled* rank sweep (+3) · deck §11

The old lab's centrepiece, done properly: **hold** `target_modules="text-linear"` fixed,
sweep only `r ∈ {8, 16, 64}`, same LR and step budget.

**Question:** rank the three knobs — rank, placement, learning rate — by effect size,
with numbers. Does your 250-sample dataset carry enough information for r=64 to use?

---

## B5 — HuggingFace Hub (+2)

`model.push_to_hub("<user>/lab21-qwen35-triage-vi")`, link in the report.

---

## B6 — Ungraded: optimizer mismatch · deck §7.3

Switching to **Muon** to fine-tune an **Adam-pretrained** model *degrades* quality —
"optimizer mismatch" — with severity proportional to update magnitude, which is why
**LoRA makes it survivable**. Try it. Do not carry the Adam learning rate over.

Write your prediction first. A wrong prediction you can explain beats a lucky guess.

---

## B7 — Ungraded: MoE route-aware LoRA · deck §7.5

On an MoE base, expert routing is heavily skewed, so adapting every expert wastes most of
the adapter. Profile routing counts on a small calibration set, adapt only the **top 25%
routed experts per layer**, compare against full LoRA.

Published result: within ±1pp of full LoRA at 70–73% fewer trainable parameters; random
expert selection at the same budget is ~2.5pp worse — the *routing signal* is what works.
And **do not train the router** — vendors disable it by default for stability.
