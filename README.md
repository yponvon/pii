# PII Redaction

Detect and redact personal identifiers in Singapore call-centre transcripts.
A GLiNER2 model is LoRA fine-tuned on nine PII types, then applied through an
inference pipeline that normalizes spoken numbers, reads long transcripts in
overlapping windows, and applies precision filters and recall boosters.

The nine labels are `sg_phone_number`, `sg_nric_fin`, `sg_address`,
`sg_postal_code`, `sg_address_unit_number`, `sg_address_block_number`,
`email_address`, `account_number`, and `full_name`.

## Results

Scored on 419 held-out authentic transcripts (precision / recall / F1 / F2,
where F2 weights recall twice as heavily as precision):

| Method       | All 9 labels          | Base 7 labels         |
|--------------|-----------------------|-----------------------|
| Baseline     | 0.68 / 0.73 / 0.70 / 0.72 | 0.67 / 0.69 / 0.68 / 0.68 |
| Rule-based   | 0.76 / 0.61 / 0.68 / 0.64 | 0.68 / 0.50 / 0.58 / 0.53 |
| **Fine-tuned** | **0.77 / 0.87 / 0.82 / 0.85** | **0.64 / 0.84 / 0.72 / 0.79** |

Recall matters more than precision here: a missed identifier is a leak, while an
over-redaction is a minor inconvenience. The fine-tuned model leads on recall and
on the recall-weighted F2 score.

## Setup

Developed on Python 3.11.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The spaCy model and base GLiNER2 weights install through `requirements.txt` and a
first-run download from Hugging Face.

## Reproduce the results

```
python inference/harness/run_frozen_comparison.py
```

This scores the baseline, rule-based, and fine-tuned methods against the frozen
gold set and writes a report to `inference/results/`.

## Redact a transcript

```python
from inference.harness.redact_output import load_finetuned, redact

model = load_finetuned()
print(redact(model, transcript, fmt="tagged"))
```

## Pipeline

A transcript flows through spoken-number normalization, overlapping-window
inference (1800-character windows with 400 overlap, to cover text beyond the
encoder's 512-token limit), the model, per-label precision filters, regex recall
boosters, cross-window reconciliation, and value propagation that redacts every
occurrence of a detected value.

The model call reads its raw predictions rather than the GLiNER2 library's
formatted output. The library's formatter deduplicates entities by value, which
drops repeated occurrences ("read-backs") that the model detects correctly but
that must still be redacted; reading the raw predictions keeps every occurrence.
See `PIPELINE_OVERVIEW.md` for the full description.

## Repository layout

```
inference/
  harness/        Detection pipeline, redaction entry point, and benchmark.
  leak_tests/     Residual-leak and account-redaction test scripts.
  results/        Benchmark metric reports.
finetuning/
  scripts/        LoRA training scripts.
  data_prep/      Training-split builders.
external/         Rule-based Presidio system and evaluation helpers.
models/           The fine-tuned keeper adapter.
```

## Data and privacy

Customer transcripts contain real PII and are not included in this repository.
The `.gitignore` excludes all data, logs, HTML reports, and per-file failure
documentation. Only the fine-tuned adapter and code are tracked.
