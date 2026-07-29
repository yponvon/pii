"""Record per-account-value survival data under 7-label redaction.

First step of the account-redaction test. Runs the chosen method with its seven
base labels (account_number is NOT queried) and, for each account value, records
how many occurrences were redacted and by which labels. The saved detail lets any
'business user can identify' rule be evaluated offline (see account_report.py).

    python account_test.py [--method finetuned|baseline|rulebased]

Output: results/leak_tests/acct_detail<suffix>.json
        (suffix is '' for finetuned, '_baseline' / '_rulebased' otherwise).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import load, account_spans, suffix, METHODS   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=METHODS, default="finetuned",
                    help="Detection method to test (default: finetuned).")
args = parser.parse_args()

frozen = [json.loads(l) for l in open(ROOT / "data" / "test" / "test_gold_419.jsonl")]
OUT = ROOT / "evaluation" / "results" / "leak_tests" / f"acct_detail{suffix(args.method)}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
handle = load(args.method)
digits = lambda s: re.sub(r"\D", "", s)

out = []
for i, d in enumerate(frozen):
    accts = d["output"]["entities"].get("account_number", [])
    if not accts:
        continue
    text = d["input"]
    spans = account_spans(handle, text)              # (text, label, start, end)

    def hit_labels(a, b):
        return {lab for _t, lab, s, e in spans if not (b <= s or a >= e)}

    vals = []
    for v in set(accts):
        occ = [m.start() for m in re.finditer(re.escape(v), text)]
        if not occ:
            continue
        red_count, labels = 0, {}
        for p in occ:
            labs = hit_labels(p, p + len(v))
            if labs:                        # this occurrence was redacted
                red_count += 1
                for lab in labs:            # which label(s) caused it
                    labels[lab] = labels.get(lab, 0) + 1
        vals.append({"v": v, "digits": len(digits(v)), "total": len(occ),
                     "red": red_count, "labels": labels})
    out.append({"line": i, "vals": vals})
json.dump(out, open(OUT, "w"))
print(f"[{args.method}] saved detail for {len(out)} account-bearing transcripts -> {OUT}")
