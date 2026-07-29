"""Apply the 'business user can identify' rule to the account survival data.

Second step of the account-redaction test. Reads acct_detail.json and, for each
account-bearing transcript, decides whether a complete account remains
recoverable, then builds a review HTML of the transcripts where it does not.

An account is recoverable when a complete account value (at least
COMPLETE_DIGITS digits) survives unredacted, or when every account piece
survives and can be reassembled. A transcript is flagged when no complete
account can be obtained. Because the rule runs on the saved detail,
COMPLETE_DIGITS can be changed without re-running inference.
Output: results/leak_tests/account_unrecoverable.html.
"""
import json, glob, os, html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii
LT = ROOT / "evaluation" / "results" / "leak_tests"
COMPLETE_DIGITS = 8

detail = json.load(open(LT / "acct_detail.json"))
frozen = [json.loads(l) for l in open(ROOT / "data" / "test" / "test_gold_419.jsonl")]
cust = {json.load(open(f))["input"]: os.path.basename(f)[:-5]
        for f in glob.glob(str(ROOT / "data" / "authentic_test" / "*.json"))}

survives = lambda x: x["red"] < x["total"]
bad = []
for d in detail:
    vals = d["vals"]
    if not vals:
        continue
    complete = [x for x in vals if x["digits"] >= COMPLETE_DIGITS]
    recoverable = any(survives(x) for x in complete) or all(survives(x) for x in vals)
    if not recoverable:
        bad.append(d)


def _as(x):
    labs = x.get("labels") or {}
    return (" as " + ", ".join(sorted(labs))) if labs else ""


def describe(x):
    occ = "occurrence" if x["total"] == 1 else "occurrences"
    return (f'{x["v"]} ({x["digits"]} digits, {x["red"]} of {x["total"]} {occ} '
            f'redacted{_as(x)})')


def render(text, vals):
    esc = html.escape(text)
    for x in sorted(vals, key=lambda x: -len(x["v"])):
        cls = "lost" if x["red"] == x["total"] else "kept"
        title = f'redacted{_as(x)}' if x["red"] else "survived"
        esc = esc.replace(html.escape(x["v"]),
                          f'<span class="{cls}" title="{html.escape(title)}">{html.escape(x["v"])}</span>')
    return esc.replace("\n", "<br>")


cards = []
for d in bad:
    line = d["line"]
    fn = cust.get(frozen[line]["input"], "?")
    summ = "  |  ".join(describe(x) for x in d["vals"])
    cards.append(f'<div class="card"><div class="hdr">line {line} <span class="fn">{html.escape(fn)}</span></div>'
                 f'<div class="info">account pieces: {html.escape(summ)}</div>'
                 f'<div class="body">{render(frozen[line]["input"], d["vals"])}</div></div>')

doc = ('<!doctype html><meta charset=utf-8><title>Account unrecoverable</title><style>'
       'body{font-family:-apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#faf9f7}'
       '.card{background:#fff;border:1px solid #e2e0dc;border-radius:10px;padding:1rem;margin:1rem 0}'
       '.hdr{font-weight:600}.fn{color:#888;font-size:.85rem;font-weight:400}'
       '.info{background:#fff8e6;border:1px solid #ecd9a0;border-radius:6px;padding:.4rem;font-size:.85rem;margin:.5rem 0}'
       '.body{font-size:.82rem;line-height:1.5}.lost{background:#ffd5cc;color:#9b1c00;font-weight:700}'
       '.kept{background:#d7f0d7;color:#1c6b1c;font-weight:700}</style>'
       f'<h2>Account unrecoverable review &mdash; {len(bad)} transcripts</h2>'
       '<p style="color:#666">Red = piece fully redacted (lost); green = survived. '
       'Judge whether a COMPLETE account is still recoverable.</p>' + "".join(cards))
html_out = LT.parent / "account_unrecoverable.html"      # results/ (not results/leak_tests/)
html_out.write_text(doc)
print(f"account-bearing: {len(detail)}  BAD (no complete account): {len(bad)} ({100*len(bad)/len(detail):.1f}%)")
print(f"HTML -> {html_out}")
