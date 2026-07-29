# Leak & Account-Redaction Tests

Two holistic, business-facing tests that complement the per-span P/R/F1/F2 metrics
in `run_benchmark.py`:

1. **PII residual-leak test** — after redaction, does a *full* direct identifier
   (mobile / NRIC / address) still sit in the transcript as plain text?
2. **Account-redaction test** — in **7-label mode** (account_number NOT a label),
   how often does the model still redact an account number so completely that the
   Business Unit can no longer identify the customer?

Both run against the frozen set (`pii/data/test/test_gold_419.jsonl`). Scripts
use repo-relative paths. **HTML reports** are written to `pii/evaluation/results/reports/`;
**intermediate data** (redacted transcripts, judge results, account detail) goes to
`pii/evaluation/results/leak_tests/`.

Run everything with the project venv:
`venv/bin/python3`

### Choosing the method (`--method`)

Every step takes `--method {finetuned,baseline,rulebased}` so all three
benchmarked detectors can be tested, not just the keeper:

- `finetuned` (default) — the shipped LoRA keeper (`models/finetuned_pii_9label/best`).
- `baseline` — the zero-shot GLiNER base model (no adapter).
- `rulebased` — the Presidio + spaCy system (`models/rule-based-gliner/redaction.py`).

Output files are **suffixed by method** so runs never overwrite each other:
`""` for finetuned (e.g. `redacted_all.jsonl`), `_baseline` / `_rulebased`
otherwise (e.g. `acct_detail_rulebased.json`,
`reports/account_unrecoverable_rulebased.html`). All the per-method plumbing
lives in `methods.py`. Pass the **same** `--method` to every step of a test
(and set `METHOD` to match in the judge notebook).

---

## Test 1 — PII residual-leak test

**Definition (lenient, per business rule):** a transcript LEAKS only if a
**complete** direct identifier appears as plain, untagged text:
- full mobile (8 SG digits, or full international)
- full NRIC (letter + 7 digits + letter)
- full address (block/unit AND street AND postal)

Partials, fragments, isolated components, weak identifiers (name / account /
company / title), and anything already inside a `<TAG>` do NOT count.

### Steps
(shown for the default `finetuned`; add `--method baseline` / `--method rulebased`
to each command to test the other detectors)
1. **Redact** all 419 → tagged transcripts (this is the inference pass):
   `python redact_transcripts.py [--method ...]`  → `results/leak_tests/redacted_all<suffix>.jsonl`
   (uses `methods.leak_tagged`, i.e. the full 9-label pipeline / Presidio tags).
2. **Judge** each transcript with an LLM (blind — no gold shown):
   - Prompt: `leak_judge_prompt.md` (the lenient definition).
   - Judge model: Azure `o4-mini`, driven by `residual_pii_analysis.ipynb`
     (the reproducible notebook in this folder). **Set `METHOD` in the notebook's
     config cell to match** the `--method` you redacted with.
   - Each shard writes `judge_result<suffix>_K.json`:
     `{"assessed": int, "leaked_lines": [..], "details": [{line,type,value}]}`
3. **Aggregate + HTML**:
   `python make_leak_report.py [--method ...]`  → `results/reports/leaked_transcripts<suffix>.html`
   (leaked plain PII in red, correct tags in blue) + prints file- and
   customer-based leak rates.

**Customer dedup:** frozen lines carry no filename, so we match `input` text back
to the per-file transcripts in `pii/data/test/*.json` (the committed `example.json`
is skipped); customer_id = `filename.split("_")[-2]`.

**Last result (keeper model):** 28/419 files (6.7%), 26/258 customers (10.1%).
26 = mobile read-backs (model tags first mention, misses the agent repeating it),
2 = one customer's full address, 0 = NRIC. Value-propagation (redact every
occurrence of a detected value) would cut it to ~1 customer.

---

## Test 2 — Account-redaction test (7-label collateral)

**Question:** with `account_number` NOT queried (7-label mode), account numbers
still get mis-tagged (mostly as `SG_PHONE_NUMBER`) and redacted. On how many calls
is the account number redacted so thoroughly the BU can't recover it?

**Definition:** BU can identify ⇔ a **complete** account number is recoverable
from unredacted text — either a full occurrence (>= COMPLETE_DIGITS) survives, OR
every piece survives (reassemblable). A surviving *fragment* alone is NOT enough.
BAD = no complete account obtainable.

### Steps
(add `--method baseline` / `--method rulebased` to both commands to test the
other detectors; the same `--method` must be passed to both)
1. **Measure** (7-label inference, saves per-value survival — run once):
   `python account_test.py [--method ...]`  → `results/leak_tests/acct_detail<suffix>.json`
   (per account value: digits, total occurrences, redacted occurrences).
2. **Apply rule + HTML** (offline, instant — tweak the rule freely, no re-run):
   `python account_report.py [--method ...]`  → prints bad-transcript count +
   `results/reports/account_unrecoverable<suffix>.html`
   (account pieces color-coded: red = fully redacted/lost, green = survived).

**Results (account-bearing calls that lose the account number, lower is better):**

| Method | Unrecoverable | Rate | Dominant cause |
|--------|---------------|------|----------------|
| finetuned (keeper) | 21 / 176 | 11.9% | digit-runs mis-tagged as `SG_PHONE_NUMBER` |
| rule-based (Presidio) | 33 / 176 | 18.8% | 6-digit runs caught by the `SG_POSTAL_CODE` regex |

Same 176 calls and `COMPLETE_DIGITS=8` rule for both; each method uses its own
benchmark threshold (keeper 0.35, rule-based 0.5). A phone/postal-vs-account
shape gate (like the existing postal gate) would reduce both.

---

## Notes
- The GLiNER methods (`finetuned`, `baseline`) need a fresh inference pass on MPS
  (~30–50 min for redaction, ~15–25 min for the 7-label account pass); keep runs
  serial (single GPU). The `rulebased` method is CPU-only and far faster.
- The account test saves survival data so definition tweaks recompute instantly.
- All per-method logic lives in `methods.py`; the report/notebook steps are
  method-agnostic and only change which suffixed files they read/write.
- HTML reports contain REAL PII — they are written locally, never published.
