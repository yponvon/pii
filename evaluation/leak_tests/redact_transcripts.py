"""Redact every frozen transcript for the leak test, with any method.

First step of the residual-PII leak test. Redacts each transcript into tagged
form with the chosen method (finetuned / baseline / rulebased) and writes the
blind versions the leak judges later review.

    python redact_transcripts.py [--method finetuned|baseline|rulebased]

Output: results/leak_tests/redacted_all<suffix>.jsonl
        (suffix is '' for finetuned, '_baseline' / '_rulebased' otherwise).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import load, leak_tagged, suffix, METHODS   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=METHODS, default="finetuned",
                    help="Detection method to redact with (default: finetuned).")
args = parser.parse_args()

FROZEN = ROOT / "data" / "test" / "test_gold_419.jsonl"
OUT = ROOT / "evaluation" / "results" / "leak_tests" / f"redacted_all{suffix(args.method)}.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(FROZEN)]
handle = load(args.method)
with open(OUT, "w", encoding="utf-8") as f:
    for i, d in enumerate(rows):
        f.write(json.dumps({"line": i, "redacted": leak_tagged(handle, d["input"])},
                           ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"[{args.method}] redacted {i+1}/{len(rows)}", flush=True)
print(f"[{args.method}] DONE -> {OUT}")
