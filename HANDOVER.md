# PII Redaction — Handover

A record of the whole project: what was built, how it works, why the key
decisions were made, how to run and retrain it, and where the real improvement
levers are. Read this first; then `README.md` for the quick start and
`PIPELINE_OVERVIEW.md` for the technical detail and data tables.

---

## 1. What this is

Detect and redact personal identifiers in Singapore call-centre transcripts.
The deliverable is a **GLiNER2 model, LoRA fine-tuned on 9 PII types**, plus a
**rule-based (Presidio) baseline** for comparison. The trained adapter is
included, so the model runs out of the box.

**Nine labels:** phone, NRIC/FIN, address, postal code, address unit number,
address block number, email, account number, full name. The base GLiNER2 model
natively supported 7; this project added the last two (**account number** and
**full name**), hand-annotated on the authentic data.

**Status:** complete and evaluated. The fine-tuned model is the production model
("the keeper"). Headline result on 419 held-out authentic calls (all 9 labels,
precision / recall / F1 / F2): **0.77 / 0.87 / 0.82 / 0.85**.

---

## 2. Repository map

```
README.md              Quick start / front door
PIPELINE_OVERVIEW.md   Technical how-it-works, with data + balance tables
HANDOVER.md            This document
requirements.txt       Pinned dependencies (Python 3.11)

finetuning/            Make the model (training)
  data_prep/           build_splits.py + generate_synthetic_data (.py demo, .ipynb live generator)
  scripts/             train.py (the keeper trainer), plot_loss.py
inference/             Redact (the pipeline)
  pipeline.py          Orchestration; + preprocessing.py, postprocessing.py, labels.py, redact.py
evaluation/            Measure the model
  run_benchmark.py     The 3-way benchmark; + benchmark_per_label.py, matcher.py, metrics.py
  results/             frozen_comparison.txt (the authoritative benchmark)
  leak_tests/          Business-facing residual-leak and account-redaction tests
models/finetuned_pii_9label/best/   The trained LoRA adapter (~13 MB)
models/rule-based-gliner/redaction.py   Rule-based Presidio baseline (the 3rd benchmark arm)
data/train/ data/val/ data/test/   Folder = split; one fake example.json each (real data offline; see DATA.md)
```

**The three scripts that matter:**
| Purpose | Script |
|---|---|
| Train the model | `finetuning/scripts/train.py` |
| Redact a transcript (production entry point) | `inference/redact.py` |
| Score on the frozen benchmark | `evaluation/run_benchmark.py` |

---

## 3. Quick start — use the model today (no training)

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
```python
from inference.redact import load_finetuned, redact
model = load_finetuned()
print(redact(model, transcript, fmt="tagged"))
```
The base GLiNER2 weights download from Hugging Face on first run; the fine-tuned
adapter is already in the repo.

---

## 4. Current results (frozen 419-transcript benchmark)

Full table in `evaluation/results/frozen_comparison.txt`. Overall
precision / recall / F1 / **F2** (F2 weights recall 2× — a miss is a leak):

| Method | all 9 labels |
|---|---|
| baseline (zero-shot GLiNER) | 0.67 / 0.74 / 0.70 / 0.72 |
| rule-based (Presidio + spaCy) | 0.76 / 0.61 / 0.68 / 0.64 |
| **fine-tuned (keeper)** | **0.77 / 0.87 / 0.82 / 0.85** |

The fine-tuned model leads on recall and on the recall-weighted F2. It also keeps
**full-NRIC protection at 100%** (no reconstructable NRIC leaks on the frozen set).

**9-label vs 7-label:** the benchmark reports both, and each column is a *genuine*
run at that label count (the models are promptable, so the label set they are
asked for slightly changes their predictions). A 7-label deployment prompts with
only the 7 base labels; those numbers are the "base 7" column.

---

## 5. How the model was built (the process)

### 5a. Data
- **Synthetic corpus** (~1,050 generated SG call transcripts) + **186 authentic**
  transcripts (`data/authentic_val/`), plus **419 authentic** held out as the
  frozen test set (`data/authentic_test/`). All of this is **real-PII-adjacent
  and kept offline** — it is not in the repository.
- The keeper trains on the **mixed** split: **1,000 synthetic + 150 authentic**
  (train), 30 synthetic + 36 authentic (val). All 186 authentic transcripts are
  used (150 train + 36 val); the 419 test set is fully disjoint.
- **Label balance:** across full transcripts every label is well represented
  (63–91% of files, mention counts within a 3.1× range). See
  `PIPELINE_OVERVIEW.md` for the tables.

### 5b. Training
- **Base model** `fastino/gliner2-privacy-filter-PII-multi` (mDeBERTa-v3-base
  encoder), **LoRA** r=16 / α=32, encoder LR 1e-5 / task LR 5e-4, batch 4,
  `max_len=512`, up to 15 epochs with **early stopping** (patience 3 on
  eval_loss), seed 42. ~6–7h on Apple MPS.
- **Important subtlety — training truncation.** With `max_len=512`, only the
  first ~512 tokens of each call reach the model — about **30% of gold entity
  occurrences**; the rest sit later in the call and are truncated away. The model
  therefore trains on a self-consistent 30% subset. This does **not** hurt final
  recall (see 5c) because inference reads the whole call.

### 5c. Inference pipeline
A transcript flows through: **spoken-number normalization** (words → digits) →
**overlapping 1,800-char windows** (400 overlap, to cover text beyond the 512-token
encoder limit) → the **model** → per-label **precision filters** → regex **recall
boosters** → cross-window **reconciliation** → **value propagation** → map spans
back to the original text.

- **Windowing is why recall does not depend on training truncation.** The model
  trained on ~30% of entities but scores 0.87 recall, because at inference it sees
  every part of the call.
- **The read-back fix (biggest single win).** The GLiNER2 library's output
  formatter deduplicates entities by value, silently dropping repeated
  occurrences ("read-backs" — a value read back for confirmation). The model
  detects these correctly at full confidence, but the library discarded them.
  Reading the **raw** predictions (`format_results=False` in `run_fulltext`)
  keeps every occurrence. This lifted all-9 recall from **0.67 → 0.87** at no
  precision cost. Value propagation is the backstop for the normalized /
  cross-window cases.

### 5d. Evaluation
- **Frozen 419** authentic transcripts, held out from training.
- **Lenient matcher** (`evaluation/matcher.py`): exact and partial-containment matches
  count; address subtypes share a label group. It does the pred↔gold matching and
  produces TP/FP/FN; the P/R/F1/F2 formulas live in `evaluation/run_benchmark.py`.
- **F2** is the headline metric (recall-weighted) because a missed identifier is a
  leak. A **full-NRIC protection** roll-up complements it: a full NRIC only
  "leaks" if every piece stays exposed, so catching any one piece is safe.
- Two **business-facing tests** in `evaluation/leak_tests/`: a residual-leak test
  (does a full identifier survive as plain text?) and an account-redaction test.

---

## 6. Key findings and decisions

1. **Read-back deduplication was the biggest lever** — a vendor-library quirk, not
   a model or data problem. Fixing it (5c) took recall 0.67 → 0.87. It is
   undocumented in the GLiNER2 README/model card; found by reading their source.
2. **Chunking the training data does NOT help (tested, dead end).** The idea was
   to recover the truncated 70% by splitting calls into ≤512-token chunks. Two
   runs (384- and 460-token chunks) both regressed: recall stayed flat (inference
   windowing already delivers it) while precision fell (short chunks strip the
   context the model needs to reject look-alikes). **Keep the keeper; do not retry
   chunking as a recall lever.**
3. **Training is not proportional to the test distribution.** Real calls are
   **name-saturated** (54.5% of test mentions are full names) and almost never
   state a full NRIC (0.1%); the synthetic training under-samples names and
   over-samples NRIC/phone. This synthetic→authentic **domain gap** is the main
   remaining ceiling. (Tables in `PIPELINE_OVERVIEW.md`.)
4. **Inference-side precision filters were measured and mostly not worth it.** A
   phone shape-gate, for example, removed 124 false positives but dropped 151 real
   ones (callers dictate phones in fragments), so it was rejected. The keeper is
   near the ceiling of what inference tuning can add.

---

## 7. Retraining

The adapter is included, so retraining is optional.
1. Drop training/validation transcripts into `data/train/` and `data/val/` (one
   `.json` each; kept offline — real PII; see DATA.md).
2. Build the splits: `python finetuning/data_prep/build_splits.py` →
   `data/train.jsonl` / `data/val.jsonl`.
3. `python finetuning/scripts/train.py [run_name]` (seed 42). Each run writes to
   its own `models/runs/<run_name>/` (timestamped if unnamed), so the shipped
   keeper is never overwritten.

This yields a *candidate* at `models/runs/<run_name>/best`. Benchmark it; only if
it beats the keeper, promote it by copying over `models/finetuned_pii_9label/best`
— the checkpoint inference loads by default.

---

## 8. Reproducing the benchmark

```
python evaluation/run_benchmark.py
```
Scores baseline, rule-based, and fine-tuned on the frozen 419 and writes
`evaluation/results/frozen_comparison.txt`. Requires the offline gold set at
`data/test/test_gold_419.jsonl`.

---

## 9. Data and privacy

Customer transcripts contain real PII and are **not** in this repository. The
`.gitignore` excludes all data, splits, logs, HTML reports, and per-file failure
documentation. Only the fine-tuned adapter and code are tracked. When handing
this to another repository, copy only the tracked files (`git ls-files`) — never
the working tree, which contains the offline data.

---

## 10. Next steps (highest-value first)

1. **More authentic training data** — the one lever grounded in the actual
   bottleneck (the domain gap in §6.3). Real, name-heavy calls with real negatives
   (the cue words and look-alikes the model over-tags) would help both recall on
   authentic phrasings and precision. This is untested and the most promising.
2. **Precision gating for specific confusions** — e.g. a phone-vs-account-vs-postal
   shape/context gate, done carefully so it does not cost recall on fragmented
   values.
3. **Re-instate the model's own read-back spans at the source** (already done via
   `format_results=False`) — keep this; do not revert to the library's formatted
   output.

---

## 11. Known limitations / gotchas

- **Domain gap:** synthetic training ≠ authentic call distribution (§6.3). The
  model still performs, but this caps further gains without new data.
- **512-token training window:** the model only trains on call openings; recall is
  carried by inference-time windowing, not training coverage. Do not "fix" this by
  chunking (tested, §6.2).
- **Vendor library dedup:** if you ever call the GLiNER2 model directly, remember
  its default output formatter drops repeated values — use `format_results=False`.
- **Training hardware:** on Apple MPS a full run is ~6–7h. A CUDA GPU cuts this to
  well under an hour.
- **Rule-based baseline** (`models/rule-based-gliner/redaction.py`) is the
  company's Presidio-based reference redactor, kept here as the third benchmark
  arm; treat it as read-only.
