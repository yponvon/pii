"""Redact every frozen transcript with the keeper model for the leak test.

First step of the residual-PII leak test. Loads the finetuned model, redacts
each transcript into tagged form, and writes the blind versions that the leak
judges later review. Output: results/leak_tests/redacted_all.jsonl.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii
sys.path.insert(0, str(ROOT / "inference" / "harness"))
from redact_output import load_finetuned, redact                # noqa: E402

FROZEN = ROOT / "data" / "frozen" / "test_gold_419.jsonl"
OUT = ROOT / "inference" / "results" / "leak_tests" / "redacted_all.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(FROZEN)]
model = load_finetuned()
with open(OUT, "w", encoding="utf-8") as f:
    for i, d in enumerate(rows):
        f.write(json.dumps({"line": i, "redacted": redact(model, d["input"], fmt="tagged")},
                           ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"redacted {i+1}/{len(rows)}", flush=True)
print(f"DONE -> {OUT}")
