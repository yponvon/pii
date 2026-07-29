# PII Redaction Pipeline — End-to-End Overview

Traced from the actual code, data, and training logs (2026-07-22). This is the
"how the whole thing works" reference: training → diagnostics → inference.

Keeper model: **`pii/models/finetuned_pii_9label/best`** (a LoRA adapter).

---

## 0. One-paragraph summary

We LoRA-finetune a GLiNER2 privacy model to tag 9 PII types in Singapore
call-centre transcripts. It trains on **1,150 transcripts** (1,000 synthetic +
150 authentic), early-stops at the best-generalizing epoch, and at inference time
each held-out transcript is **normalized (spoken words → digits) → sliced into
overlapping windows → tagged → precision-filtered → recall-boosted**, then spans
map back to the original text for redaction.

---

## 1. What we want the model to learn

Tag these **9 labels** wherever they appear, however they're spoken (whole,
in fragments, digit-by-digit, or as words):

`sg_phone_number` · `sg_nric_fin` · `sg_address` · `sg_postal_code` ·
`sg_address_unit_number` · `sg_address_block_number` · `email_address` ·
**`account_number`** · **`full_name`**

The last two are the new ones this project added (base model had 7). Core intent:
**recall matters more than precision** — a missed identifier is a leak (real harm);
an over-redaction is mostly a nuisance. That's why the eval reports **F2** (recall
weighted 2× precision) alongside F1.

---

## 2. Training data

Format — one JSON object per line (`.jsonl`), model-native GLiNER2 shape:

```json
{"input": "<full transcript text>",
 "output": {"entities": {"sg_phone_number": ["9123 4567", "9123 4567", ...],
                          "account_number": ["8221 618442"], ...}}}
```

- Value = the **surface string** as spoken. **Duplicates are meaningful**: a value
  listed N times must occur N times in the text (the "gold invariant"). Fragments
  are listed as separate values (`"S117"`, `"5-8-4-2-H"`).

### The splits actually used by the keeper (`data/train.jsonl` / `data/val.jsonl`)

| split | lines | contents |
|---|--:|---|
| **TRAIN** | 1,150 | 1,000 synthetic + 150 authentic transcripts |
| **VAL** | 66 | authentic (early-stopping signal) |
| **TEST** | 419 | authentic, fully held out → the frozen benchmark |

All three verified disjoint by full-text hash. Only **3 / 1,150** train examples
are negatives (no entities) — the corpus is almost all PII-bearing.

Train-set label balance — every label is well represented: each appears in
63–91% of the 1,150 transcripts, and mention counts span a modest 3.1× range
(no label dominates or is starved).

| Label | Files with label | % of files | Mentions |
|---|--:|--:|--:|
| Phone | 936 | 81% | 6,266 |
| Account | 1,050 | 91% | 4,073 |
| NRIC/FIN | 737 | 64% | 3,847 |
| Address | 903 | 79% | 3,833 |
| Postal code | 721 | 63% | 3,198 |
| Unit number | 846 | 74% | 2,998 |
| Block number | 783 | 68% | 2,943 |
| Email | 803 | 70% | 2,160 |
| Full name | 816 | 71% | 2,038 |

**What the model actually trains on (the 512-token cut).** Training uses
`max_len=512`, so only the first ~512 tokens of each call reach the model — about
**30%** of all gold entity occurrences; the rest sit later in the call and are
truncated away. Because call openings are verification-heavy (name and address are
read out first), this makes the *training-visible* balance skewed (an 8× range vs
the 3.1× above):

| Label | Seen in training | % of label seen |
|---|--:|--:|
| Full name | 1,274 / 2,038 | 63% |
| Unit number | 1,397 / 3,024 | 46% |
| Address | 1,647 / 3,847 | 43% |
| Block number | 1,211 / 2,950 | 41% |
| Postal code | 1,280 / 3,213 | 40% |
| Phone | 1,646 / 6,282 | 26% |
| Account | 461 / 4,073 | 11% |
| NRIC/FIN | 300 / 3,847 | 8% |
| Email | 165 / 2,163 | 8% |
| **Total** | **9,381 / 31,437** | **30%** |

Recall does not depend on this, though: overlapping-window inference reads the whole
call at prediction time, and regex boosters backstop the structured labels
(NRIC, email, postal) that training under-samples — so final recall stays high on
every label (NRIC 1.00, email 0.97) despite the low training exposure.

### Label set: 7 base + 2 new = 9

The base GLiNER2 privacy model natively recognises **7 labels** (phone, address,
block number, unit number, postal code, email, NRIC/FIN). This project **added 2** —
**account number** and **full name** — which the authentic transcripts were
hand-annotated for. There is **one** fine-tuned model, trained on all **9** (not a
separate 7-label model); the benchmark reports **base-7** (comparable to the base
model and the rule-based system) and **all-9** (with the 2 new labels).

### Is training proportional to the test set? (composition, base-7 vs new-2)

Comparing what the model trains on (the 512-cut view) to the authentic test set, as
each label's share of all PII mentions:

| Label | Set | Train (512-cut) | Test | Gap |
|---|---|--:|--:|--:|
| Address | base | 17.6% | 8.9% | −8.6 |
| Phone | base | 17.5% | 7.8% | −9.8 |
| Unit | base | 14.9% | 5.6% | −9.3 |
| Postal | base | 13.6% | 5.1% | −8.6 |
| Block | base | 12.9% | 2.6% | −10.3 |
| NRIC/FIN | base | 3.2% | 0.1% | −3.1 |
| Email | base | 1.8% | 0.9% | −0.9 |
| **Full name** | **new** | **13.6%** | **54.5%** | **+40.9** |
| **Account** | **new** | **4.9%** | **14.6%** | **+9.6** |

Training is **not** proportional to the test distribution, and the gap is
concentrated in the **2 new labels**: real calls are name-saturated (54.5%) and
account-heavy (14.6%), whereas synthetic training under-samples both; the 7 base
labels are over-represented in training relative to real calls. Note the tension —
a corpus *balanced across labels* is by definition not *proportional* to a
name-dominated test set. Closing the gap needs more authentic, name-heavy data.

**Synthetic corpus** (1,050 files, `generated_data_14jul/`) was gold-integrity
repaired: under-count 0, phantom 0. Bugs fixed included 2,049 under-counted tags,
temperature "23" mis-tagged as phone, 259 "Ya" particles mis-tagged as names.
**Authentic** (data_185 val-source + data_all test-source) was hand-annotated for
the 2 new labels; the original 7 labels are byte-identical to pre-annotation backups.

---

## 3. Training run (the keeper)

Script: `finetuning/scripts/train.py` (sweep-winner "D" config, run full-length).

| setting | value |
|---|---|
| base model | `fastino/gliner2-privacy-filter-PII-multi` (mDeBERTa-v3-base encoder) |
| method | **LoRA** r=16, α=32, dropout=0 (adapter only, ~13 MB) |
| encoder LR / task LR | 1e-5 / 5e-4 |
| batch size | 4 |
| **max_len** | **512 tokens** (hard encoder ceiling ≈ 1,800 chars) |
| epochs | 15 requested, **early stopping** patience 3 on `eval_loss` |
| seed | 42 |
| wall time | ~3h39m on MPS (single GPU, ~0.6 samples/s) |

### Is it learning? Is it overfitting? — YES to both, and early stopping handled it

Per-epoch loss from `history.json`:

| epoch | mean train loss | val loss (eval_loss) |
|--:|--:|--:|
| 0 | 48.45 | 59.00 |
| 1 | 23.37 | 41.05 |
| 2 | 22.03 | 53.06 |
| **3** | **16.33** | **36.53  ← BEST, saved as `best/`** |
| 4 | 16.29 | 58.94 |
| 5 | 13.14 | 51.01 |
| 6 | 12.63 | 46.16 |

Reading it:
- **Learning:** train loss falls monotonically 48 → 13. The model is clearly fitting
  the task (not stuck / not noise).
- **Generalizing to epoch 3:** val loss drops to a clear minimum at epoch 3 (36.5,
  well below epoch 0's 59).
- **Overfitting after epoch 3:** past ep3, train loss keeps falling but val loss
  **rises and oscillates** (58.9, 51.0, 46.2) — the textbook divergence signature.
  Patience-3 early stopping stopped at epoch 6 and **kept epoch 3** as `best`.
- **Train↔val gap** (16 vs 36) is sizeable but expected: train is 87% synthetic
  while val is authentic → a domain gap, not pure memorization. This is the main
  reason recall isn't higher, and points at "more authentic data" as the next lever.

Loss curve rendered at `models/finetuned_pii_9label/loss.png`.

---

## 4. What a TEST transcript goes through at inference

Canonical path = `run_windowed()` in `inference/pipeline.py`.
Order matters:

```
original transcript
      │
 (A) NORMALIZE spoken numbers → digits        normalize_numbers()
      │   "eight nine five" → "895"; keep a map back to original spans
      │
 (B) WINDOW: 1800-char windows, 400 overlap    (only if text > 1800 chars)
      │   each entity < 400 chars ⇒ fully inside ≥1 window (coverage guarantee)
      │
 (C) MODEL tags each window                     model.extract_entities(thr=0.35)
      │
 (D) PRECISION FILTERS (drop false positives)   run_fulltext()
      │
 (E) RECALL BOOSTERS (add missed entities)      regex, context-gated, conf=1.0
      │
 (F) CROSS-WINDOW RECONCILIATION                keep one detection per entity
      │
 (G) MAP spans back to ORIGINAL text            redact real words, not digits
```

### (A) Spoken-number normalization
Numbers dictated as words are near-invisible to a digit-trained model (measured
recall **0.04** on spelled-out PII). We collapse `zero/one/.../nine/oh/double/triple`
runs to digits **before** inference, tag on the digit form, then map the accepted
span **back to the original words** so the redaction covers the real text.
word→digit is unambiguous ("eight" is always 8) → no rigidity risk; a no-op on
digit-only transcripts.

### (B)+(F) Windowing & foolproof reconciliation
The encoder only sees 512 tokens (~1,800 chars); authentic calls average ~4,600
chars, so a single pass truncates most of the call. We slide 1,800-char windows
with 400-char overlap. Because every PII entity is far shorter than the 400-char
overlap, each is fully contained in ≥1 window — nothing is lost at a boundary.

When two overlapping windows see the same entity (possibly with **different
labels** near a truncated edge), we keep exactly one, chosen deterministically by:
1. **margin** — distance to the nearest *truncating* window edge (the window that
   saw the entity most interior wins),
2. confidence, 3. span length, 4. position/label.
A later candidate is dropped as a duplicate only if it's the same entity
(same-label overlap, or different-label with IoU ≥ 0.5). Genuinely **nested**
different-label spans (a block inside an address) are preserved.

### (D) Precision filters — per label, shape + context gated
Each was diagnosed from a specific false-positive cluster:
- **EMAIL** — must have email shape (`@`/`at`/`.com`…); denylist the utility's own
  mis-transcribed domain ("SP Group" → sbgroup/htgroup).
- **PHONE** — reject money (`$`, `.dd`) and NRIC/unit-shaped `A746` / `502P`.
- **POSTAL** — must be exactly 6 digits **and** have address context nearby (real
  postal codes aren't bare coincidental 6-digit numbers).
- **UNIT** — trust clean `##-####` shape; else require `unit`/`#`/address context;
  reject `Nth floor`, "reference/account number", inbound/outbound labels.
- **BLOCK** — no hyphen (that's a unit), not "Nth floor", require `block`/`blk` or
  address proximity.
- **NRIC** — accept a full NRIC **or a fragment** (`S117`, `5842H`), because an NRIC
  dictated in pieces would otherwise be silently dropped = a full-NRIC leak.
- **ACCOUNT / NAME** — no format filter (every grouping counts / any name).

### (E) Recall boosters — deterministic regex, context-gated, confidence 1.0
- **POSTAL** — 6-digit runs with address/"postal code" context nearby. *Guard:*
  skips runs flanked by other digits (`372717` inside `896 372717 1`) — those are
  account-number slices, not postal codes. (This fixed the account over-redaction.)
- **NRIC** — `[STFG]\d{7}[A-Z]` near nric/ic/fin keywords.
- **EMAIL** — domain-shaped tails the model missed.

Threshold used in the frozen comparison: **0.35** for GLiNER (windowed 1800/400).

---

## 5. How the three methods differ (the benchmark)

| method | what it is |
|---|---|
| **baseline** | base GLiNER2, no finetune, same windowing + filters/boosters |
| **rulebased** | the legacy regex/Presidio system (no account/name recognizer) |
| **finetuned** | the keeper LoRA, same harness |

Identical gold, matcher (`match_entities_fixed`), filters, boosters, and windowing
across all three — the **only** variable is the detector. (Windowing is applied to
the two GLiNER methods; rule-based runs full-text as it has no context ceiling.)

**Metric:** per-label and overall **P / R / F1 / F2**, plus a **full-NRIC
protection** roll-up (safe if ≥1 fragment is caught, since that breaks
reconstruction). F2 is the headline because recall (leak-avoidance) is the goal.

**Current results** (`evaluation/results/frozen_comparison.txt`),
overall P/R/F1/**F2**:

| method | all 9 (9-label prompt) | base 7 (7-label prompt) |
|---|---|---|
| baseline | 0.67/0.74/0.70/**0.72** | 0.63/0.69/0.66/**0.68** |
| rulebased | 0.76/0.61/0.68/**0.64** | 0.68/0.50/0.58/**0.53** |
| **finetuned** | **0.77/0.87/0.82/0.85** | **0.61/0.84/0.71/0.78** |

Each method is run as it would be deployed: prompted with all 9 labels, and
prompted with only the 7 base labels. The base-7 column is a **genuine 7-label
run**, not a subset of the 9-label run — the GLiNER models are promptable, so
dropping account/name from the prompt slightly lowers their base-7 precision
(those entities leak into the nearest base label). The rule-based system is
unchanged between the two (its recognizers are independent).

Finetuned wins on F1/F2/recall across all 9 labels, tying rule-based on precision
(0.77 vs 0.76) while leading recall by 26 points. The gap is widest on
**F2/recall**, the headline metric. Full-NRIC protection: finetuned 2/2 safe,
rule-based leaks 1/2.

The rule-based method is scored with its account (bare 10-digit regex) and name
(spaCy PERSON) recognizers enabled, so the comparison is fair.

---

## 6. The two business-facing tests (complement the P/R/F1/F2 metric)

See `pii/evaluation/leak_tests/README.md`. Both run the keeper on the frozen 419:
1. **Residual-leak test** — does a *full* identifier survive as plain text? (lenient)
2. **Account-redaction test** — in 7-label mode, how often is an account number
   redacted so completely the BU can't recover the customer?

---

## 7. Open levers (not yet done)

- **More authentic training data** — the train(synthetic)↔val(authentic) domain gap
  is the main recall ceiling; biggest untested lever.
- **Phone-vs-account gate** — reduce account digit-runs mis-tagged as phone.

### Done

- **Read-back redaction** — repeated occurrences of a detected value are now
  redacted. Two mechanisms cover this: (1) the model call reads raw predictions
  instead of the GLiNER2 library's formatted output, whose value-level dedup was
  dropping read-backs the model detects correctly; and (2) value propagation
  re-applies a confirmed value to occurrences the raw predictions miss (e.g. a
  value spoken as words in one place and digits in another, or repeats split
  across windows). Together these raised all-9 recall from 0.67 to 0.87 at no
  precision cost.
