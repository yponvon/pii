# Leak & Account-Redaction Tests

Two holistic, business-facing tests that complement the per-span P/R/F1/F2 metrics
in `run_frozen_comparison.py`:

1. **PII residual-leak test** — after redaction, does a *full* direct identifier
   (mobile / NRIC / address) still sit in the transcript as plain text?
2. **Account-redaction test** — in **7-label mode** (account_number NOT a label),
   how often does the model still redact an account number so completely that the
   Business Unit can no longer identify the customer?

Both run against the frozen set (`pii/data/frozen/test_gold_419.jsonl`) and the
fine-tuned keeper model (`pii/models/finetuned_pii_9label/best`). Scripts
use repo-relative paths. **HTML reports** are written to `pii/inference/results/`
(`leaked_transcripts.html`, `account_unrecoverable.html`); **intermediate data**
(redacted transcripts, judge results, account detail) goes to
`pii/inference/results/leak_tests/`.

Run everything with the project venv:
`venv/bin/python3`

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
1. **Redact** all 419 → tagged transcripts:
   `python redact_transcripts.py`  → writes `results/leak_tests/redacted_all.jsonl`
   (uses `redact_output.redact(model, text, fmt="tagged")`).
2. **Judge** each transcript with an LLM (blind — no gold shown):
   - Prompt: `leak_judge_prompt.md` (the lenient definition).
   - Judge model: Azure `o4-mini`, driven by `residual_pii_analysis.ipynb`
     (the reproducible notebook in this folder). Optionally shard the input
     first with `python split_for_judges.py N`  → `judge_chunk_*.jsonl`.
   - Each shard writes `judge_result_K.json`:
     `{"assessed": int, "leaked_lines": [..], "details": [{line,type,value}]}`
3. **Aggregate + HTML**:
   `python make_leak_report.py`  → `results/leak_tests/leaked_transcripts.html`
   (leaked plain PII in red, correct tags in blue) + prints file- and
   customer-based leak rates.

**Customer dedup:** frozen lines carry no filename, so we match `input` text back
to `pii/data/authentic_test/*.json`; customer_id = `filename.split("_")[-2]`.

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
1. **Measure** (7-label inference, saves per-value survival — run once):
   `python account_test.py`  → `results/leak_tests/acct_detail.json`
   (per account value: digits, total occurrences, redacted occurrences).
2. **Apply rule + HTML** (offline, instant — tweak the rule freely, no re-run):
   `python account_report.py`  → prints bad-transcript count +
   `results/leak_tests/account_unrecoverable.html`
   (account pieces color-coded: red = fully redacted/lost, green = survived).

**Last result (keeper model):** 21/176 account-bearing calls (11.9%) lose the
account number. Almost all: account digit-runs mis-tagged as `SG_PHONE_NUMBER`.
A phone-vs-account context/shape gate (like the postal gate) would reduce it.

---

## Notes
- Both tests need a fresh model inference pass (~30–50 min for redaction, ~15–25
  min for the 7-label account pass) on MPS; keep runs serial (single GPU).
- The account test saves survival data so definition tweaks recompute instantly.
- HTML reports contain REAL PII — they are written locally, never published.
