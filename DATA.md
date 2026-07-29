# Data

This document describes the **shape** of the data the model consumes, **where**
it lives, and **how** the synthetic training corpus was generated.

Real transcripts contain customer PII, so no actual data is committed. Each data
folder ships with a single synthetic `example.json` that shows the exact shape;
every real file stays local and is gitignored.

---

## 1. Data shape (model input / output)

One JSON object per record, in the model-native GLiNER2 format:

```json
{"input": "<full transcript text>",
 "output": {"entities": {"sg_phone_number": ["9123 4567", "9123 4567"],
                          "sg_nric_fin": ["S1234567A"],
                          "account_number": ["8221 618442"]}}}
```

- **`input`** — the raw transcript string (speaker-labelled turns).
- **`output.entities`** — a map from each of the 9 labels to the list of exact
  surface strings that are PII of that type.
- **Duplicates are meaningful (the "gold invariant"):** a value that occurs *N*
  times in the transcript is listed *N* times. This is how read-backs (a number
  repeated for confirmation) are represented, and the pipeline is expected to
  redact every occurrence.

The nine labels: `sg_phone_number`, `sg_nric_fin`, `sg_address`,
`sg_postal_code`, `sg_address_unit_number`, `sg_address_block_number`,
`email_address`, `account_number`, `full_name`.

Source transcripts are one record per `.json`; the built training/validation
splits are one record per line (`.jsonl`). See `data/train/example.json`,
`data/val/example.json`, and `data/test/example.json` for concrete committed
examples.

At **inference** the model takes just the `input` string and returns predicted
`(text, label)` spans; `inference/redact.py` turns those into a redacted string.

---

## 2. Where the data lives

Everything lives under one folder, `data/`. **The folder *is* the split** — put a
transcript in `data/train/` and it is training data; put it in `data/val/` and it
is validation. There are no hardcoded counts or sampling; you reshape the splits
just by adding or removing files.

| Path | Contents | Committed? |
|---|---|---|
| `data/train/` | training transcripts (one `.json` each) | only `example.json` |
| `data/val/` | validation transcripts | only `example.json` |
| `data/test/` | the frozen held-out benchmark gold (`test_gold_419.jsonl`) | only `example.json` |
| `data/train.jsonl`, `data/val.jsonl` | the built splits (produced by `build_splits.py`) | no (real PII) |

Each split folder may hold provenance subfolders — the shipped layout keeps
`synthetic/` and `authentic/` apart (`data/train/synthetic/`,
`data/train/authentic/`, and the same under `data/val/`) — and the builder globs
them recursively, so the structure is optional.

The consuming code points at these paths:
- `finetuning/data_prep/build_splits.py` reads `data/train/` and `data/val/` and
  writes `data/train.jsonl` / `data/val.jsonl`. The test set is frozen and never
  rebuilt.
- `finetuning/scripts/train.py` reads `data/train.jsonl` and `data/val.jsonl`.
- `evaluation/run_benchmark.py` and `evaluation/benchmark_per_label.py` read
  `data/test/test_gold_419.jsonl`.

---

## 3. How the synthetic corpus was generated

The training corpus is ~1,000 **hand-authored** synthetic SP Group call-centre
transcripts (plus 150 annotated authentic calls). The synthetic transcripts were
written in batch scripts, each producing a handful of records. The method:

1. **Scenario design.** Each transcript covers a realistic scenario (address
   change, bill dispute, account update, refund, sensitive-customer handling,
   and so on) at a difficulty tier — *normal*, *medium*, *hard* — including
   *negative* calls that contain no PII at all. This variety is deliberate: it
   teaches the model the many ways PII is spoken (whole, in fragments,
   digit-by-digit, spelled out as words, and read back for confirmation).
2. **Hand-labelling.** Each transcript is annotated by hand into the `entities`
   dict, listing the exact surface strings per label, with the gold invariant
   (occurrences counted, so read-backs appear as repeated values).
3. **Integrity check.** Each listed gold value is verified to actually occur in
   the transcript (an under-count/phantom check), so the gold never drifts from
   the text.
4. **Write.** Each record is saved as `{"input", "output"}` JSON.

Two files support this, and both write to a **`data/generated/` staging folder**.
Generation is a candidate step: review what lands there, then move approved files
into `data/train/synthetic/` (or `data/val/synthetic/`) before running
`build_splits.py`.

- **`finetuning/data_prep/generate_synthetic_data.ipynb`** — the **live generator**.
  Azure `o4-mini` drafts a long scenario/tier-specific transcript with its gold
  labels, the gold-integrity checker verifies the occurrence counts (with an
  automatic repair pass), and files are saved as `NNN_tier_scenario.json` (anything
  the checker still doubts is marked `_REVIEW`). This reproduces the authoring loop
  above; it needs `pip install -r requirements.txt` and the `AZURE_OPENAI_*`
  environment variables.
- **`finetuning/data_prep/generate_synthetic_data.py`** — a static, dependency-free
  **shape demo**: three hand-written records (normal address update, medium phone
  read-back, negative) showing the exact output format. Run with
  `python finetuning/data_prep/generate_synthetic_data.py`.

The authentic calls were hand-annotated for the two new labels (`account_number`,
`full_name`); their original seven labels are byte-identical to the pre-annotation
backups.
