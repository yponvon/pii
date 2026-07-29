"""Detection-method registry for the leak / account tests.

One place that knows how to run each of the three benchmarked methods so the
test scripts (account_test.py, redact_transcripts.py) can be pointed at any of
them with a --method flag instead of hardcoding the fine-tuned keeper.

Each method exposes exactly the two operations the tests need:
    account_spans(handle, text) -> [(text, label, start, end), ...]   (7-label; account test)
    leak_tagged(handle, text)   -> transcript with <LABEL> tags       (9-label; leak test)

The label sets differ on purpose: the account test runs 7-label (account_number
NOT queried, mirroring the Business-Unit deployment), the leak test runs full
9-label redaction. Both use each method's own benchmark threshold, because the
GLiNER confidence scale and Presidio's score scale are not comparable.

Output-file suffix convention (used by every leak/account script):
    finetuned -> ""            (e.g. redacted_all.jsonl)  -- backward compatible
    baseline  -> "_baseline"   (e.g. redacted_all_baseline.jsonl)
    rulebased -> "_rulebased"
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # .../pii

# NOTE: gliner2 / torch / pipeline are imported lazily inside the functions that
# need a model, so the offline aggregators (account_report.py, make_leak_report.py)
# can `from methods import suffix, METHODS` without paying the heavy import cost.

ADAPTER = ROOT / "models" / "finetuned_pii_9label" / "best"
RULEBASED_PY = ROOT / "models" / "rule-based-gliner" / "redaction.py"
GLINER_THRESHOLD = 0.35
RULEBASED_THRESHOLD = 0.5

METHODS = ("finetuned", "baseline", "rulebased")

# Presidio recognizer name -> canonical label (mirrors run_benchmark.py).
RULEBASED_NORMALIZED_LABEL = {
    "EMAIL_ADDRESS": "EMAIL_ADDRESS", "SG_NRIC_FIN": "SG_NRIC_FIN",
    "SG_PHONE_NUMBER": "SG_PHONE_NUMBER", "SG_ADDRESS": "SG_ADDRESS",
    "SG_POSTAL_CODE": "SG_POSTAL_CODE", "SG_UNIT_NUMBER": "SG_ADDRESS_UNIT",
    "SG_ADDRESS_BLOCK_NUMBER": "SG_ADDRESS_BLOCK",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER", "PERSON": "FULL_NAME",
}
# 7-label subset for the account test (account_number + full_name dropped).
RULEBASED_ENTITIES_7 = [e for e in RULEBASED_NORMALIZED_LABEL
                        if RULEBASED_NORMALIZED_LABEL[e] not in ("ACCOUNT_NUMBER", "FULL_NAME")]


def suffix(method: str) -> str:
    """Output-file suffix for a method: '' for finetuned, '_<method>' otherwise."""
    return "" if method == "finetuned" else f"_{method}"


def _load_gliner(adapter=None):
    sys.path.insert(0, str(ROOT / "inference"))
    from pipeline import MODEL_PATH          # noqa: E402
    from gliner2 import GLiNER2              # noqa: E402
    model = GLiNER2.from_pretrained(MODEL_PATH)
    if adapter is not None:
        model.load_adapter(str(adapter))
    return model


def load(method: str):
    """Load a method and return an opaque handle ``(kind, obj)`` for the helpers below."""
    if method == "finetuned":
        return ("gliner", _load_gliner(ADAPTER))
    if method == "baseline":
        return ("gliner", _load_gliner(None))          # zero-shot base model, no adapter
    if method == "rulebased":
        spec = importlib.util.spec_from_file_location("presidio_redaction", RULEBASED_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return ("rulebased", mod.PIIRedactor())
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def account_spans(handle, text):
    """7-label spans as (span_text, canon_label, start, end) for the account test."""
    kind, obj = handle
    if kind == "gliner":
        sys.path.insert(0, str(ROOT / "inference"))
        from pipeline import LABELS_7, run_windowed          # noqa: E402
        return [(t, l, s, e) for t, l, s, e, _c in
                run_windowed(obj, text, LABELS_7, GLINER_THRESHOLD, return_spans=True)]
    results = obj.analyzer.analyze(text=text, entities=RULEBASED_ENTITIES_7,
                                   language="en", score_threshold=RULEBASED_THRESHOLD)
    return [(text[r.start:r.end], RULEBASED_NORMALIZED_LABEL[r.entity_type], r.start, r.end)
            for r in results if r.entity_type in RULEBASED_NORMALIZED_LABEL]


def leak_tagged(handle, text):
    """Full 9-label tagged redaction string for the residual-leak test."""
    kind, obj = handle
    if kind == "gliner":
        sys.path.insert(0, str(ROOT / "inference"))
        from redact import redact                            # noqa: E402  (LABELS_9 internally)
        return redact(obj, text, fmt="tagged")
    return obj.redact(text, use_tags=True)                   # PIIRedactor: default entities = all 9
