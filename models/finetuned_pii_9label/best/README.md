---
base_model: fastino/gliner2-privacy-filter-PII-multi
library_name: peft
tags:
- base_model:adapter:fastino/gliner2-privacy-filter-PII-multi
- lora
- transformers
---

# PII redaction adapter — Singapore call-centre transcripts (9 labels)

A LoRA adapter that fine-tunes `fastino/gliner2-privacy-filter-PII-multi`
(GLiNER2, mDeBERTa-v3-base encoder) to detect personal identifiers in Singapore
utility call-centre transcripts. This is the shipped "keeper" adapter; the
inference pipeline (`inference/redact.py`) loads it by default.

## What it does

Detects nine PII types as `(text, label)` spans:

`sg_phone_number`, `sg_nric_fin`, `sg_address`, `sg_postal_code`,
`sg_address_unit_number`, `sg_address_block_number`, `email_address`,
`account_number`, `full_name`.

Seven are inherited from the base model; `account_number` and `full_name` were
added by this project. Addresses are tagged compositionally (street, block, unit,
postal code as separate labels).

## Intended use

Redacting PII from transcribed SP Group call-centre conversations (speech-to-text
output, so numbers arrive spoken, fragmented, and read back for confirmation). Not
a general-purpose PII model — it is tuned for this domain and Singapore formats.

```python
from inference.redact import load_finetuned, redact
model = load_finetuned()
print(redact(model, transcript, fmt="tagged"))
```

The adapter is applied through the full pipeline (spoken-number normalization,
overlapping-window inference, precision filters, recall boosters, value
propagation) — see `PIPELINE_OVERVIEW.md`. Used raw, without that pipeline, recall
is materially lower.

## Training

- **Data:** mixed synthetic + authentic SP Group transcripts — 1,150 train
  (1,000 synthetic + 150 authentic) and 66 val (30 synthetic + 36 authentic).
  Gold follows the "gold invariant": each value is listed once per occurrence, so
  read-backs are labelled. See `DATA.md`.
- **Method:** LoRA (r=16, α=32, dropout=0) on the encoder + task head; encoder LR
  1e-5, task-head LR 5e-4; max sequence length 512; seed 42; up to 15 epochs with
  early stopping (patience 3 on `eval_loss`). See `finetuning/scripts/train.py`.

## Evaluation

Scored on a frozen, held-out set of 419 authentic transcripts against a zero-shot
GLiNER baseline and a rule-based Presidio baseline. The headline metric is **F2**
(recall weighted 2× precision, because a missed identifier is a leak while an
over-redaction is a minor inconvenience). The fine-tuned model leads on recall and
F2, and keeps full-NRIC protection at 100% on the frozen set. Full, current numbers
live in `evaluation/results/frozen_comparison.txt` (regenerate with
`evaluation/run_benchmark.py`).

## Limitations

- **Domain-specific:** Singapore formats and SP Group call flows; not validated
  elsewhere.
- **512-token encoder ceiling:** mitigated at inference by overlapping windows;
  the raw adapter still only sees 512 tokens per pass.
- **Synthetic→authentic gap:** training is ~87% synthetic, so recall on real calls
  is the main headroom; more authentic data is the next lever.
- **PII data is never shipped:** all training/eval transcripts are kept offline.

## License

Internal SP Group project. Inherits the base model's license
(`fastino/gliner2-privacy-filter-PII-multi`); confirm terms before external use.

### Framework versions

- PEFT 0.19.1
