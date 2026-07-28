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

Source corpus files are one record per `.json`; the built splits are one record
per line (`.jsonl`). See `training_data/example.json`, `val_data/example.json`,
and `test_data/example.json` for concrete committed examples.

At **inference** the model takes just the `input` string and returns predicted
`(text, label)` spans; `inference/redact.py` turns those into a redacted string.

---

## 2. Where the data lives

| Folder | Contents | Committed? |
|---|---|---|
| `training_data/` | training split (`train_mixed2.jsonl`) | only `example.json` |
| `val_data/` | validation split (`val_mixed2.jsonl`) | only `example.json` |
| `test_data/` | frozen held-out test set (`test_gold_419.jsonl`) | only `example.json` |
| `data/` | offline source corpora (`synthetic/`, `authentic_val/`, `authentic_test/`) that the splits are built from | nothing (fully gitignored) |

The consuming code points at these folders:
- `finetuning/scripts/train.py` reads `training_data/` and `val_data/`.
- `evaluation/run_benchmark.py` and `evaluation/benchmark_per_label.py` read `test_data/`.
- `finetuning/data_prep/build_mixed_training_data.py` reads the `data/` source
  corpora and writes the splits into `training_data/` / `val_data/` / `test_data/`.

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

`finetuning/data_prep/generate_synthetic_data.py` is a cleaned, runnable example
batch demonstrating exactly this method (three records: a normal address update,
a medium phone read-back, and a negative). Run it with:

```
python finetuning/data_prep/generate_synthetic_data.py
```

The authentic calls were hand-annotated for the two new labels (`account_number`,
`full_name`); their original seven labels are byte-identical to the pre-annotation
backups.
