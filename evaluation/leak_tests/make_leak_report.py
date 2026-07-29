"""Aggregate leak-judge results into leak rates and a highlighted HTML report.

Final step of the residual-PII leak test. Combines the judge_result<suffix>_*.json
files for the chosen --method, computes file-based and customer-based leak rates,
and renders each leaked transcript with the residual PII and correct tags
highlighted.

    python make_leak_report.py [--method finetuned|baseline|rulebased]

Output: results/reports/leaked_transcripts<suffix>.html.
"""
import argparse, json, glob, os, html, re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import suffix, METHODS   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii
LT = ROOT / "evaluation" / "results" / "leak_tests"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=METHODS, default="finetuned",
                    help="Which method's redaction + judge results to aggregate (default: finetuned).")
args = parser.parse_args()
sfx = suffix(args.method)

# Collect this method's judge shards. For finetuned (sfx="") the pattern
# judge_result_*.json would also sweep in the other methods' prefixed files,
# so filter those out explicitly.
if args.method == "finetuned":
    judge_files = [f for f in glob.glob(str(LT / "judge_result_*.json"))
                   if not re.search(r"judge_result_(baseline|rulebased)_", os.path.basename(f))]
else:
    judge_files = glob.glob(str(LT / f"judge_result{sfx}_*.json"))

leaks = {}
for f in judge_files:
    for it in json.load(open(f)).get("details", []):
        leaks.setdefault(it["line"], []).append(it)
red = {json.loads(l)["line"]: json.loads(l)["redacted"] for l in open(LT / f"redacted_all{sfx}.jsonl")}

# customer_id via the per-file test transcript filename (the frozen gold carries
# no filename); the committed example.json is a fake shape demo, so skip it.
cust = {json.load(open(f))["input"]: os.path.basename(f)[:-5]
        for f in glob.glob(str(ROOT / "data" / "test" / "*.json"))
        if os.path.basename(f) != "example.json"}
frozen = [json.loads(l) for l in open(ROOT / "data" / "test" / "test_gold_419.jsonl")]
fname = {i: cust.get(d["input"], "(unmatched)") for i, d in enumerate(frozen)}
cid = lambda i: fname[i].split("_")[-2] if fname[i] != "(unmatched)" else None


def render(text, values):
    esc = html.escape(text)
    terms = set()
    for v in values:
        if not v:
            continue
        terms.add(v.strip())
        for part in re.split(r"[,/]", v):
            if len(part.strip()) >= 3:
                terms.add(part.strip())
    terms = [t for t in sorted((html.escape(t) for t in terms), key=len, reverse=True) if t and t in esc]
    if terms:
        esc = re.compile("|".join(re.escape(t) for t in terms)).sub(
            lambda m: f'<span class="leak">{m.group()}</span>', esc)
    esc = re.sub(r"(&lt;[A-Z_]+(?::[0-9.]+)?&gt;)", r'<span class="tag">\1</span>', esc)
    return esc.replace("\n", "<br>")


cards = []
for n, line in enumerate(sorted(leaks), 1):
    items = leaks[line]
    tags = " ".join(f'<b>{html.escape(it["type"])}</b>: <span class="leak">{html.escape(it["value"])}</span>'
                    for it in items)
    cards.append(f'<div class="card"><div class="hdr">#{n} line {line} <span class="fn">{html.escape(fname[line])}</span></div>'
                 f'<div class="leakinfo">LEAKED &rarr; {tags}</div>'
                 f'<div class="body">{render(red.get(line,""), [it["value"] for it in items])}</div></div>')

doc = ('<!doctype html><meta charset=utf-8><title>Leaked transcripts</title><style>'
       'body{font-family:-apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#faf9f7;color:#222}'
       '.card{background:#fff;border:1px solid #e2e0dc;border-radius:10px;padding:1rem 1.2rem;margin:1.2rem 0}'
       '.hdr{font-weight:600}.fn{color:#888;font-weight:400;font-size:.85rem}'
       '.leakinfo{background:#fff3f0;border:1px solid #f3c9bf;border-radius:6px;padding:.4rem .6rem;font-size:.9rem;margin:.6rem 0}'
       '.body{font-size:.84rem;line-height:1.5}.leak{background:#ffd5cc;color:#9b1c00;font-weight:700;padding:0 2px;border-radius:3px}'
       '.tag{background:#e6f0ff;color:#1c4f9b;padding:0 2px;border-radius:3px;font-weight:600}</style>'
       f'<h2>Residual PII leaks &mdash; {args.method} (lenient rule)</h2>'
       f'<p style="color:#666">{len(leaks)} leaked / 419. Red = leaked plain PII, blue = correct tag.</p>' + "".join(cards))
html_out = LT.parent / "reports" / f"leaked_transcripts{sfx}.html"   # results/reports/
html_out.parent.mkdir(parents=True, exist_ok=True)
html_out.write_text(doc)

total_cust = {c for i in range(len(frozen)) if (c := cid(i))}
leaked_cust = {c for i in leaks if (c := cid(i))}
print(f"[{args.method}] FILE-based leak:     {len(leaks)}/419 = {100*len(leaks)/419:.1f}%")
print(f"[{args.method}] CUSTOMER-based leak: {len(leaked_cust)}/{len(total_cust)} = {100*len(leaked_cust)/max(1,len(total_cust)):.1f}%")
print(f"HTML -> {html_out}")
