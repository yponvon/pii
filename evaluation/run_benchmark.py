"""Benchmark three PII detection methods against a single frozen gold set.

Scores baseline zero-shot GLiNER, rule-based presidio, and fine-tuned GLiNER
against one frozen gold set so the resulting numbers are directly comparable.
The comparison holds the following constant across all three methods:

  * Gold set : test_data/test_gold_419.jsonl   (the same 419 files for all three)
  * Matcher  : match_entities_fixed              (the same matching logic for all three)
  * NRIC gold: full-format filter disabled       (frozen-path policy; fragments count)
  * Labels   : standardised to the same nine NORMALIZED_LABEL labels for all three
  * Inference: overlapping windows for the two GLiNER methods, which share the
               512-token encoder limit (89% of frozen files exceed it); full
               text for the rule-based method, which has no such limit.

Windowing is applied only to the GLiNER methods. It is a remedy for the encoder's
truncation limit, not a general accuracy improvement, so applying it only where
that limit exists keeps the comparison fair. The rule-based method runs on full
text because windowing would fragment the surrounding context its filters depend
on.

The rule-based method now covers all nine labels, including ACCOUNT_NUMBER and
FULL_NAME, so it is scored fairly across every label.

Requirements:
  Run with the project virtual environment, which provides gliner2, presidio and
  the other dependencies (from the repository root):
      ./venv/bin/python evaluation/run_benchmark.py
  The first run downloads the GLiNER base model from Hugging Face and caches it.

Usage:
  python run_benchmark.py [ADAPTER_DIR] [-o OUTPUT]
  python run_benchmark.py --help
  ADAPTER_DIR defaults to the fine-tuned keeper's best checkpoint; if it does not exist,
  the fine-tuned method is skipped and only baseline and rule-based are scored.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Callable, List, Tuple

# Repository layout.
# All paths are derived from this file's own location so the script is portable:
# it runs from any working directory and on any machine that has the repository
# checked out, with no absolute paths to edit. The directory tree is:
#     pii/
#       evaluation/run_benchmark.py                  this file
#       evaluation/results/                          report is written here
#       evaluation/{matcher,metrics}.py              matcher + scoring helpers
#       test_data/test_gold_419.jsonl                the frozen gold set (offline)
#       inference/                                   the detection pipeline
#       models/finetuned_pii_9label/                 fine-tuned adapter
#       models/rule-based-gliner/redaction.py        rule-based baseline
EVAL_DIR = Path(__file__).resolve().parent
PII_ROOT = EVAL_DIR.parent
INFERENCE_DIR = PII_ROOT / "inference"
RULEBASED_REDACTION_PY = PII_ROOT / "models" / "rule-based-gliner" / "redaction.py"

# The pipeline (inference/) and the sibling matcher/metrics modules are imported
# by name, so their directories must be on sys.path before the imports below.
for _path in (EVAL_DIR, INFERENCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pipeline import (  # noqa: E402
    NORMALIZED_LABEL, MODEL_PATH, BASELINE_LABELS, SYNTHETIC_LABELS, run_windowed,
)
from matcher import match_entities_fixed  # noqa: E402
from metrics import _prf  # noqa: E402

from gliner2 import GLiNER2

# Fixed inputs and outputs.
FROZEN = PII_ROOT / "test_data" / "test_gold_419.jsonl"          # the gold set
DEFAULT_ADAPTER = PII_ROOT / "models" / "finetuned_pii_9label" / "best"
RESULTS = PII_ROOT / "evaluation" / "results" / "frozen_comparison.txt"  # text report

# Reported F-score weighting. For PII redaction a miss (a leak) is worse than an
# over-redaction, so recall is weighted FBETA times as much as precision. F2 is
# the standard recall-favouring choice; raise it for even harsher recall weighting.
FBETA = 2.0

# The two GLiNER methods share a single confidence threshold so that the only
# variable between them is the fine-tuning itself.
GLINER_THRESHOLD = 0.35
# Presidio's production default. Note that presidio detection scores are not on
# the same scale as GLiNER confidence, so applying an identical numeric
# threshold to both would not make them comparable.
RULEBASED_THRESHOLD = 0.5

# Overlapping-window inference parameters (see run_windowed in inference/pipeline.py).
WIN_CHARS, OVERLAP = 1800, 400

# The baseline model is zero-shot promptable, so it is queried with the full
# label set, including the sg_contact_number synonym (which NORMALIZED_LABEL folds into
# SG_PHONE_NUMBER) and the two labels introduced by the synthetic corpus. This
# gives the baseline the same opportunity to detect every label the other
# methods are scored on.
BASELINE_QUERY = BASELINE_LABELS + ["account_number", "full_name"]

# Mapping from the rule-based system's recognizer names to the shared NORMALIZED_LABEL
# labels. ACCOUNT_NUMBER comes from a bare-10-digit PatternRecognizer, and
# FULL_NAME from spaCy's PERSON entity (both defined in redaction.py), so the
# rule-based system is scored fairly on all 9 labels.
RULEBASED_NORMALIZED_LABEL = {
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "SG_NRIC_FIN": "SG_NRIC_FIN",
    "SG_PHONE_NUMBER": "SG_PHONE_NUMBER",
    "SG_ADDRESS": "SG_ADDRESS",
    "SG_POSTAL_CODE": "SG_POSTAL_CODE",
    "SG_UNIT_NUMBER": "SG_ADDRESS_UNIT",
    "SG_ADDRESS_BLOCK_NUMBER": "SG_ADDRESS_BLOCK",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "PERSON": "FULL_NAME",
}
RULEBASED_ENTITIES = list(RULEBASED_NORMALIZED_LABEL.keys())

# Report order. The final two labels are produced only by the fine-tuned model.
REPORT_LABELS = ["EMAIL_ADDRESS", "SG_ADDRESS", "SG_ADDRESS_BLOCK", "SG_ADDRESS_UNIT",
                 "SG_NRIC_FIN", "SG_PHONE_NUMBER", "SG_POSTAL_CODE",
                 "ACCOUNT_NUMBER", "FULL_NAME"]
BASE7 = REPORT_LABELS[:7]  # the seven labels reported by the historical benchmark

# 7-label query variants: the same per-method queries with the two new labels
# (account_number, full_name) dropped. A 7-label deployment prompts each method
# with only those seven, so the "base 7 labels" column is scored from a genuine
# 7-label run rather than a subset of the 9-label run. This matters because the
# GLiNER models are promptable: the label set they are asked for changes what
# they predict for the labels that remain.
BASELINE_QUERY_7 = [l for l in BASELINE_QUERY if NORMALIZED_LABEL.get(l) not in ("ACCOUNT_NUMBER", "FULL_NAME")]
SYNTHETIC_LABELS_7 = [l for l in SYNTHETIC_LABELS if NORMALIZED_LABEL.get(l) not in ("ACCOUNT_NUMBER", "FULL_NAME")]
RULEBASED_ENTITIES_7 = [e for e in RULEBASED_ENTITIES if RULEBASED_NORMALIZED_LABEL[e] not in ("ACCOUNT_NUMBER", "FULL_NAME")]

# Aggregated results for one method: true/false positives and false negatives
# bucketed per label and corpus-wide, plus the NRIC leak-prevention roll-up
# (nric_protected of nric_total calls containing a full NRIC had at least one
# NRIC piece caught; see the note in score()).
Scores = namedtuple(
    "Scores",
    "tp_by_label fp_by_label fn_by_label total_tp total_fp total_fn "
    "nric_protected nric_total"
)

NRIC_LABEL = "SG_NRIC_FIN"


def load_pii_redactor():
    """Load and return the PIIRedactor class from models/rule-based-gliner/redaction.py.

    Loaded by explicit file location rather than a bare ``import redaction`` so
    the reference is always this single file, independent of sys.path import
    order. redaction.py imports only installed packages (Presidio + GLiNER2), so
    it loads correctly in isolation.
    """
    redaction_py = RULEBASED_REDACTION_PY
    spec = importlib.util.spec_from_file_location("presidio_redaction", redaction_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PIIRedactor


def load_frozen() -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Load the frozen gold set.

    Returns:
        A list of (transcript, gold) cases, where gold is a list of
        (entity_text, STD_LABEL) pairs.
    """
    cases = []
    with open(FROZEN, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            gold = [(text, NORMALIZED_LABEL[label])
                    for label, values in record["output"]["entities"].items()
                    for text in values]
            cases.append((record["input"], gold))
    return cases


# Predictors.
# Each predictor takes the full transcript and returns a list of
# (entity_text, STD_LABEL) predictions, so all three feed the shared matcher
# in an identical format.

def predict_baseline(model, text: str, labels) -> List[Tuple[str, str]]:
    """Zero-shot GLiNER, run with overlapping-window inference on `labels`."""
    return run_windowed(model, text, labels, GLINER_THRESHOLD,
                        win_chars=WIN_CHARS, overlap=OVERLAP)


def predict_finetuned(model, text: str, labels) -> List[Tuple[str, str]]:
    """Fine-tuned GLiNER, run with overlapping-window inference on `labels`."""
    return run_windowed(model, text, labels, GLINER_THRESHOLD,
                        win_chars=WIN_CHARS, overlap=OVERLAP)


_SPEAKER_LABEL_RE = re.compile(r'^(?:speaker[_\s]*\d+|unknown|caller|agent|customer)$',
                               re.IGNORECASE)


def predict_rulebased(redactor, text: str, entities) -> List[Tuple[str, str]]:
    """Rule-based presidio system, run on the full transcript for `entities`.

    Detections are mapped from presidio recognizer names to the shared NORMALIZED_LABEL
    labels.
    """
    results = redactor.analyzer.analyze(
        text=text, entities=entities, language="en",
        score_threshold=RULEBASED_THRESHOLD,
    )
    predictions = []
    for result in results:
        canon = RULEBASED_NORMALIZED_LABEL.get(result.entity_type)
        if canon is None:
            continue
        span = text[result.start:result.end].strip()
        if not span:
            continue
        # spaCy tags transcript speaker labels ("SPEAKER_01", "Unknown") as PERSON.
        # These are formatting, not names, so drop them so name precision reflects
        # real content. The GLiNER models do not make this mistake.
        if canon == "FULL_NAME" and _SPEAKER_LABEL_RE.match(span):
            continue
        predictions.append((span, canon))
    return predictions


def score(cases, predict_fn: Callable[[str], List[Tuple[str, str]]]) -> Scores:
    """Run one predictor over every frozen case and aggregate the results.

    Scores each case with the shared matcher and returns per-label and
    corpus-wide tallies plus the NRIC roll-up.

    NRIC roll-up: a full NRIC only leaks when it can be reconstructed in full,
    that is, when every one of its pieces is left exposed. If the model redacts
    even one piece, the complete NRIC can no longer be assembled and what remains
    exposed is just a non-identifying fragment, so that call is safe. A call with
    NRIC gold is therefore counted as protected when at least one of its NRIC
    pieces was caught; it is a leak only when all pieces were missed.
    nric_protected of nric_total is the number of full NRICs kept safe.
    """
    tp_by_label = defaultdict(int)
    fp_by_label = defaultdict(int)
    fn_by_label = defaultdict(int)
    total_tp = total_fp = total_fn = 0
    nric_protected = nric_total = 0
    for text, gold in cases:
        pred = predict_fn(text)
        tp, fp, fn, errors_fp, errors_fn, tp_per_label = match_entities_fixed(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for label, count in tp_per_label.items():
            tp_by_label[label] += count
        for label, _text in errors_fp:
            fp_by_label[label] += 1
        for label, _text in errors_fn:
            fn_by_label[label] += 1
        # NRIC leak-prevention roll-up (one call is one full NRIC in the frozen
        # set). Safe when at least one NRIC piece was caught, which breaks the
        # full reconstruction; a leak only when every piece was missed.
        nric_gold = sum(1 for _text, label in gold if label == NRIC_LABEL)
        if nric_gold:
            nric_total += 1
            nric_missed = sum(1 for label, _text in errors_fn if label == NRIC_LABEL)
            if nric_missed < nric_gold:   # at least one piece caught, so not reconstructable
                nric_protected += 1
    return Scores(tp_by_label, fp_by_label, fn_by_label,
                  total_tp, total_fp, total_fn, nric_protected, nric_total)


def _fbeta(precision, recall, beta=FBETA):
    """Compute the F-beta score, weighting recall `beta` times as much as precision.

    beta=1 is F1, beta=2 is F2. Returns zero when both precision and recall are
    zero.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def overall_prf(labels, s: Scores):
    """Aggregate precision, recall and F1 over the given subset of labels.

    The F1 element is ignored by the report, which recomputes F-beta.
    """
    tp = sum(s.tp_by_label[label] for label in labels)
    fp = sum(s.fp_by_label[label] for label in labels)
    fn = sum(s.fn_by_label[label] for label in labels)
    return _prf(tp, fp, fn)


def evaluate_methods(cases, adapter: Path, log: Callable[[str], None]) -> dict:
    """Run all three methods over the frozen cases and return the scores.

    Each method is run twice: once prompted with all nine labels, and once with
    only the seven base labels, so the two report columns reflect how the method
    is actually called in a 9-label and a 7-label deployment. The result is
    ``{name: {"nine": Scores, "seven": Scores}}``. The fine-tuned method is
    skipped if its adapter directory does not exist, so the script still produces
    a baseline versus rule-based comparison before any model has been trained.
    """
    methods = {}

    log("[1/3] baseline (zero-shot GLiNER, windowed) -- 9-label then 7-label ...")
    base_model = GLiNER2.from_pretrained(MODEL_PATH)
    methods["baseline"] = {
        "nine": score(cases, lambda text: predict_baseline(base_model, text, BASELINE_QUERY)),
        "seven": score(cases, lambda text: predict_baseline(base_model, text, BASELINE_QUERY_7)),
    }

    log("[2/3] rule-based (presidio, full text) -- 9-label then 7-label ...")
    redactor_class = load_pii_redactor()
    redactor = redactor_class()
    methods["rulebased"] = {
        "nine": score(cases, lambda text: predict_rulebased(redactor, text, RULEBASED_ENTITIES)),
        "seven": score(cases, lambda text: predict_rulebased(redactor, text, RULEBASED_ENTITIES_7)),
    }

    log("[3/3] fine-tuned (GLiNER + adapter, windowed) -- 9-label then 7-label ...")
    if adapter.exists():
        ft_model = GLiNER2.from_pretrained(MODEL_PATH)
        ft_model.load_adapter(str(adapter))
        methods["finetuned"] = {
            "nine": score(cases, lambda text: predict_finetuned(ft_model, text, SYNTHETIC_LABELS)),
            "seven": score(cases, lambda text: predict_finetuned(ft_model, text, SYNTHETIC_LABELS_7)),
        }
    else:
        log(f"      adapter not found at {adapter} -- fine-tuned method skipped")

    return methods


def format_report(methods: dict, num_files: int) -> List[str]:
    """Build the full text report from the scored methods.

    Returns the report as a list of lines: a header, a per-label table, and an
    overall table. Methods appear as columns in a fixed order; any that were
    skipped are omitted.
    """
    columns = [name for name in ("baseline", "rulebased", "finetuned") if name in methods]
    lines = []

    def cell(prf) -> str:
        # Report both F1 (balanced) and F-beta (recall-weighted). For PII
        # redaction a miss (a leak) is worse than an over-redaction, so F-beta
        # weights recall FBETA times as much as precision. F1 is kept alongside
        # it for reference.
        precision, recall, f1 = prf
        return f"{precision:.2f}/{recall:.2f}/{f1:.2f}/{_fbeta(precision, recall):.2f}"

    fscore = f"F1 / F{FBETA:g}   (F{FBETA:g} = recall-weighted, beta={FBETA:g})"

    # Header: what was run and under what settings.
    lines.append("=" * 78)
    lines.append(f"FROZEN 3-WAY COMPARISON  --  {num_files} files ({FROZEN.name})")
    lines.append(f"matcher=match_entities_fixed  NRIC-filter=OFF  "
                 f"GLiNER thr={GLINER_THRESHOLD} (windowed {WIN_CHARS}/{OVERLAP})  "
                 f"rule-based thr={RULEBASED_THRESHOLD} (full text)")
    lines.append("=" * 78)

    def per_label_table(title, prompt_key, label_list):
        lines.append("")
        lines.append(f"PER-LABEL — {title}  (precision / recall / {fscore})")
        lines.append("-" * 78)
        lines.append(f"{'label':<20}" + "".join(f"{name:>23}" for name in columns))
        for label in label_list:
            row = f"{label:<20}"
            for name in columns:
                s = methods[name][prompt_key]
                row += "  " + cell(_prf(s.tp_by_label[label], s.fp_by_label[label], s.fn_by_label[label]))
            lines.append(row)

    # Two per-label tables: one for each deployment configuration. The 9-label
    # prompt covers all nine labels; the 7-label prompt is a genuine 7-label run
    # (the model is only asked for those seven), scored on the base seven.
    per_label_table("9-label prompt (all 9 labels)", "nine", REPORT_LABELS)
    per_label_table("7-label prompt (base 7 labels)", "seven", BASE7)

    # Overall table: all nine (from the 9-label run) and base seven (from the
    # 7-label run), so each column matches how the method would actually be called.
    lines.append("")
    lines.append(f"OVERALL  (precision / recall / {fscore})")
    lines.append("-" * 78)
    lines.append(f"{'method':<20}{'all 9 (9-label prompt)':>26}{'base 7 (7-label prompt)':>26}")
    for name in columns:
        s9, s7 = methods[name]["nine"], methods[name]["seven"]
        lines.append(f"{name:<20}{cell(overall_prf(REPORT_LABELS, s9)):>26}"
                     f"{cell(overall_prf(BASE7, s7)):>26}")

    # Full-NRIC roll-up (from the 9-label run): a full NRIC leaks only when every
    # piece is left exposed. Catching even one piece breaks reconstruction, so the
    # call is safe.
    lines.append("")
    lines.append("FULL-NRIC PROTECTION  (9-label prompt; safe = at least one piece caught, breaking reconstruction)")
    lines.append("-" * 78)
    for name in columns:
        s = methods[name]["nine"]
        leaked = s.nric_total - s.nric_protected
        rate = f"rate {s.nric_protected / s.nric_total:.2f}" if s.nric_total else "n/a"
        lines.append(f"{name:<20}{s.nric_protected}/{s.nric_total} safe, "
                     f"{leaked} leaked   ({rate})")

    lines.append("")
    lines.append("Note: rule-based ACCOUNT_NUMBER = bare-10-digit regex; FULL_NAME = spaCy PERSON.")
    lines.append("GLiNER methods use value propagation (read-back repeats redacted).")
    lines.append("base 7 = a genuine 7-label run (each method prompted with only the 7 base labels).")
    return lines


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Score the baseline, rule-based and fine-tuned methods "
                    "against the frozen gold set and print a comparison report.",
    )
    parser.add_argument(
        "adapter", nargs="?", default=str(DEFAULT_ADAPTER),
        help="Fine-tuned LoRA adapter directory to evaluate. If it does not "
             "exist, the fine-tuned method is skipped. Default: %(default)s",
    )
    parser.add_argument(
        "-o", "--output", default=str(RESULTS),
        help="Path to write the text report. Default: %(default)s",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = Path(args.adapter)
    results_path = Path(args.output)

    if not FROZEN.exists():
        sys.exit(f"error: frozen gold set not found at {FROZEN}\n"
                 "Check out the full repository, or update the FROZEN path.")

    # 1. Load the frozen gold set once; every method is scored against it.
    cases = load_frozen()
    print(f"Loaded {len(cases)} frozen files from {FROZEN.name}\n")

    # 2. Run the three methods, printing progress as each one starts.
    methods = evaluate_methods(cases, adapter, log=lambda msg: print(msg, flush=True))

    # 3. Build the report, then both print it and save it to disk.
    report = "\n".join(format_report(methods, len(cases)))
    print("\n" + report)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(report + "\n")
    print(f"\nReport written to {results_path}")


if __name__ == "__main__":
    main()
