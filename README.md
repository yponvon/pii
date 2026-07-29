# PII Redaction

Detect and redact personal identifiers in Singapore call-centre transcripts.
A GLiNER2 model is LoRA fine-tuned on nine PII types, then applied through an
inference pipeline that normalizes spoken numbers, reads long transcripts in
overlapping windows, and applies precision filters and recall boosters.

The nine labels are `sg_phone_number`, `sg_nric_fin`, `sg_address`,
`sg_postal_code`, `sg_address_unit_number`, `sg_address_block_number`,
`email_address`, `account_number`, and `full_name`.

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
python evaluation/run_benchmark.py
```

This scores the baseline, rule-based, and fine-tuned methods against the frozen
gold set and writes a report to `evaluation/results/`.

## Redact a transcript

```python
from inference.redact import load_finetuned, redact

model = load_finetuned()
print(redact(model, transcript, fmt="tagged"))
```

## Retrain the model

The trained adapter is already included, so retraining is optional. To retrain:

1. Drop your training transcripts into `data/train/` and validation transcripts
   into `data/val/` (one `.json` per transcript; these hold real PII, so the
   folders are kept offline — see `DATA.md`).
2. Build the splits: `python finetuning/data_prep/build_splits.py` → writes
   `data/train.jsonl` / `data/val.jsonl`.
3. Run `python finetuning/scripts/train.py [run_name]` (seed 42). Each run writes
   to its own folder `models/runs/<run_name>/` (auto-named by timestamp if you
   omit `run_name`), so retraining never overwrites the shipped keeper.

This produces a *candidate* adapter at `models/runs/<run_name>/best`. Benchmark
it, and only if it beats the current keeper promote it — copy it over
`models/finetuned_pii_9label/best`, which is the checkpoint the inference scripts
load by default.

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

## End-to-end flow

```
1. finetuning/data_prep/build_splits.py  → data/train.jsonl / data/val.jsonl   (offline; real PII)
2. finetuning/scripts/train.py [run_name]             → models/runs/<run_name>/best/   (candidate LoRA adapter + loss.png)
3. evaluation/run_benchmark.py         → evaluation/results/frozen_comparison.txt
        the 3-way benchmark: baseline vs rule-based vs fine-tuned, 419 frozen set, 9- and 7-label prompts
4. evaluation/benchmark_per_label.py          → per-label P/R/F1 on the same 419 set
5. evaluation/leak_tests/                               → residual-leak + account-redaction business tests
```

Steps 1–2 are optional (the trained adapter ships). Step 3 writes the benchmark
report to `evaluation/results/frozen_comparison.txt`.

## Repository layout

```
finetuning/          Make the model (training)
  data_prep/         build_splits.py + generate_synthetic_data (.py demo, .ipynb live generator)
  scripts/           train.py, plot_loss.py
inference/           Redact (the pipeline)
  pipeline.py        Orchestration (run_windowed / run_fulltext)
  preprocessing.py   Spoken-number normalization
  postprocessing.py  Precision filters + recall boosters
  labels.py          Label lists + the NORMALIZED_LABEL name map
  redact.py          Production entry point
evaluation/          Measure the model
  run_benchmark.py   The 3-way benchmark
  benchmark_per_label.py
  matcher.py         Pred-vs-gold matcher
  metrics.py         P/R/F1 + label-group helpers
  results/           frozen_comparison.txt
  leak_tests/        Residual-leak + account-redaction tests
models/
  finetuned_pii_9label/best/   The fine-tuned LoRA adapter (the keeper)
  rule-based-gliner/           The rule-based Presidio baseline (redaction.py)
data/
  train/ val/ test/            Folder = split; one fake example.json each (real data offline)
```

See `DATA.md` for the data shape and how the synthetic corpus was generated.

## Data and privacy

Customer transcripts contain real PII and are not included in this repository.
The `.gitignore` excludes all data, logs, HTML reports, and per-file failure
documentation. Only the fine-tuned adapter and code are tracked.
