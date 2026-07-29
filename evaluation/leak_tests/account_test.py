"""Record per-account-value survival data under the 7-label redaction model.

First step of the account-redaction test. Runs the finetuned model with its
seven base labels (account_number is not queried) and, for each account value,
records how many occurrences were redacted and by which labels. The saved
detail lets any 'business user can identify' rule be evaluated offline.
Output: results/leak_tests/acct_detail.json.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii
sys.path.insert(0, str(ROOT / "inference"))
from pipeline import LABELS_7, run_windowed   # 7 base labels, no account_number
from redact import load_finetuned                        # noqa: E402

frozen = [json.loads(l) for l in open(ROOT / "test_data" / "test_gold_419.jsonl")]
OUT = ROOT / "evaluation" / "results" / "leak_tests" / "acct_detail.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
model = load_finetuned()
digits = lambda s: re.sub(r"\D", "", s)

out = []
for i, d in enumerate(frozen):
    accts = d["output"]["entities"].get("account_number", [])
    if not accts:
        continue
    text = d["input"]
    spans = run_windowed(model, text, LABELS_7, 0.35, return_spans=True)  # (text,label,s,e,conf)

    def hit_labels(a, b):
        return {lab for _t, lab, s, e, _c in spans if not (b <= s or a >= e)}

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
print(f"saved detail for {len(out)} account-bearing transcripts -> {OUT}")
