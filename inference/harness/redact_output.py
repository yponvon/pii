"""Format fine-tuned GLiNER2 predictions into redacted output.

Converts detected PII spans into one of three output formats:
    raw    : list of {text, label, start, end, confidence} for inspection.
    tagged : the transcript with each span replaced by <LABEL>.
    spans  : list of (text, label) pairs for auditing.

Overlapping predictions (e.g. a block number nested within an address) are
reduced to the outermost span so a region is never redacted twice.

Usage:
    from redact_output import load_finetuned, redact
    model = load_finetuned()
    print(redact(model, transcript, fmt="tagged"))
"""
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from evaluate_finetuned import MODEL_PATH, SYNTHETIC_LABELS, run_windowed  # noqa: E402
from gliner2 import GLiNER2  # noqa: E402

DEFAULT_ADAPTER = HARNESS_DIR.parent.parent / "models" / "finetuned_pii_9label" / "best"
THRESHOLD = 0.35


def load_finetuned(adapter=DEFAULT_ADAPTER):
    model = GLiNER2.from_pretrained(MODEL_PATH)
    model.load_adapter(str(adapter))
    return model


def predict_spans(model, transcript, threshold=THRESHOLD):
    """Return predicted spans as (text, label, start, end, confidence) tuples in original-text coordinates."""
    return run_windowed(model, transcript, SYNTHETIC_LABELS, threshold, return_spans=True)


def _outermost(spans):
    """Keep only outermost spans, dropping any span overlapping a longer kept span.

    This ensures redaction tags the outermost PII only, so that redacting an
    address also hides its nested block.
    """
    kept = []
    for text, label, s, e, conf in sorted(spans, key=lambda x: -(x[3] - x[2])):
        if any(not (e <= ks or s >= ke) for _t, _l, ks, ke, _c in kept):
            continue
        kept.append((text, label, s, e, conf))
    return kept


def to_tagged(transcript, spans, with_confidence=False):
    """Return the transcript with each PII span replaced by its label tag."""
    result = transcript
    for text, label, s, e, conf in sorted(_outermost(spans), key=lambda x: -x[2]):
        tag = f"<{label}:{conf:.2f}>" if with_confidence else f"<{label}>"
        result = result[:s] + tag + result[e:]
    return result


def redact(model, transcript, fmt="tagged", threshold=THRESHOLD, with_confidence=False):
    spans = predict_spans(model, transcript, threshold)
    if fmt == "raw":
        return [{"text": t, "label": l, "start": s, "end": e, "confidence": round(c, 3)}
                for t, l, s, e, c in spans]
    if fmt == "spans":
        return [(t, l) for t, l, _s, _e, _c in spans]
    if fmt == "tagged":
        return to_tagged(transcript, spans, with_confidence)
    raise ValueError(f"unknown fmt {fmt!r}; use 'raw', 'tagged', or 'spans'")


if __name__ == "__main__":
    # Quick demo on a short synthetic transcript (no real data) so the model can
    # be tried end to end without the offline test set:
    #     python inference/harness/redact_output.py
    demo = (
        "SPEAKER_00: Good afternoon, may I have your name please?\n"
        "SPEAKER_01: Hi, it's Jason Lim.\n"
        "SPEAKER_00: Thanks Jason. Can I verify your NRIC and contact number?\n"
        "SPEAKER_01: Sure, S1234567A, and my mobile is 9123 4567.\n"
        "SPEAKER_00: And the address on file?\n"
        "SPEAKER_01: Block 123 Sunrise Street 12, #05-06, Singapore 570123.\n"
        "SPEAKER_00: And the account number on the bill?\n"
        "SPEAKER_01: It's 1234567890, and my email is jason.lim@example.com.\n"
    )
    print("Loading model ...")
    model = load_finetuned()
    print("\n--- REDACTED ---")
    print(redact(model, demo, fmt="tagged"))
