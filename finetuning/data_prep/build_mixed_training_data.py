"""Build the train/val/test jsonl splits for the 9-label PII model (the keeper's
"mixed2" recipe).

The split mixes synthetic and authentic transcripts:

  TRAIN : 1000 synthetic + 150 authentic   -> train_mixed2.jsonl   (1150)
  VAL   :   30 synthetic +  36 authentic   -> val_mixed2.jsonl     (66)
  TEST  : all authentic (held out entirely) -> test_mixed2.jsonl

Mixing 150 authentic calls into TRAIN (not synthetic only) is what the shipped
keeper trained on: it narrows the synthetic-to-authentic domain gap. VAL and TEST
are authentic-heavy so they measure real-world performance rather than how well
the model learned the generator's habits; the authentic sets also carry ~20-24%
zero-entity negatives, which the synthetic corpus lacks.

The 186 authentic val-source files are split 150 (train) + 36 (val); all
authentic test-source files are held out as TEST and never seen in training.

The output format is one JSON object per line,
{"input": <transcript>, "output": {"entities": {<label>: [<str>, ...]}}},
written with ensure_ascii=False. All source directories already use that schema,
so records pass through verbatim.

Runs are deterministic under SEED=42. The 30 synthetic val files are drawn
stratified by the difficulty tier encoded in each filename, so val is not
accidentally dominated by one tier.

The authentic val and test sets must carry account_number and full_name gold.
Training a 9-label model against val gold that lacks these two labels would
teach it to suppress them on realistic text, so the script fails loudly if that
annotation is missing.

Usage:
  python3 build_mixed_training_data.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
# Source corpora (offline, real PII): the hand-authored synthetic transcripts and
# the annotated authentic calls. See DATA.md for how the synthetic corpus is made.
SYNTH_DIR = BASE / "data" / "synthetic"
VAL_AUTH_DIR = BASE / "data" / "authentic_val"
TEST_AUTH_DIR = BASE / "data" / "authentic_test"
# Built splits are written to the committed data folders (real files stay local).
TRAIN_DIR = BASE / "training_data"
VAL_DIR = BASE / "val_data"
TEST_DIR = BASE / "test_data"

SEED = 42
N_TRAIN_SYNTH = 1000
N_VAL_SYNTH = 30
N_VAL_AUTH = 36   # of the authentic val-source files; the remainder (150) go to TRAIN

TIERS = ("easy", "medium", "hard", "normal")

LABELS = [
    "sg_phone_number", "sg_nric_fin", "sg_address", "sg_postal_code",
    "sg_address_unit_number", "sg_address_block_number", "email_address",
    "account_number", "full_name",
]
NEW_LABELS = ("account_number", "full_name")


def tier_of(path):
    parts = path.stem.split("_")
    for p in parts:
        if p in TIERS:
            return p
    return "normal"


def load_dir(d, require_output=True):
    """Load JSON records from a directory.

    Args:
        d: Directory to scan for *.json files.
        require_output: When True, records lacking an "output" key are skipped.

    Returns:
        A tuple (records, skipped). Each record is a (path, obj) pair where obj
        is the raw {"input", "output"} dict. Each skipped entry is a
        (filename, reason) pair.
    """
    recs, skipped = [], []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append((p.name, f"unparseable: {exc}"))
            continue
        if "input" not in obj or (require_output and "output" not in obj):
            skipped.append((p.name, "missing input/output key"))
            continue
        recs.append((p, obj))
    return recs, skipped


def label_counts(recs):
    c = Counter()
    files = Counter()
    for _p, obj in recs:
        ents = obj.get("output", {}).get("entities", {})
        for k, v in ents.items():
            if v:
                c[k] += len(v)
                files[k] += 1
    return c, files


def zero_entity(recs):
    return sum(1 for _p, o in recs
               if not any(o.get("output", {}).get("entities", {}).values()))


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for _p, obj in recs:
            fh.write(json.dumps({"input": obj["input"], "output": obj["output"]},
                                ensure_ascii=False) + "\n")


def report(name, recs):
    counts, files = label_counts(recs)
    lens = [len(o["input"]) for _p, o in recs]
    mean = round(sum(lens) / len(lens)) if lens else 0
    print(f"\n--- {name}: {len(recs)} examples (mean {mean} chars, "
          f"{zero_entity(recs)} zero-entity) ---")
    print(f"  {'label':28} {'mentions':>9} {'files':>7}")
    for lab in LABELS:
        print(f"  {lab:28} {counts.get(lab, 0):>9} {files.get(lab, 0):>7}")
    print(f"  {'TOTAL':28} {sum(counts.values()):>9}")


def main():
    rng = random.Random(SEED)

    synth, synth_skipped = load_dir(SYNTH_DIR)
    val_auth, val_skipped = load_dir(VAL_AUTH_DIR)
    test_auth, test_skipped = load_dir(TEST_AUTH_DIR)

    for name, sk in (("synthetic", synth_skipped), ("data_185", val_skipped),
                     ("data_all", test_skipped)):
        for fn, why in sk:
            print(f"SKIPPED [{name}] {fn}: {why}")

    # Guard: the authentic sets must carry the two new labels.
    problems = []
    for name, recs in (("data_185 (val)", val_auth), ("data_all (test)", test_auth)):
        counts, _ = label_counts(recs)
        missing = [l for l in NEW_LABELS if counts.get(l, 0) == 0]
        if missing:
            problems.append(f"{name} has no gold for: {', '.join(missing)}")
    if problems:
        print("\nERROR -- authentic sets are not annotated for the new labels:")
        for p in problems:
            print("  " + p)
        print("\nTraining a 9-label model against val gold that lacks these labels\n"
              "teaches it to suppress them on realistic text. Annotate first, or\n"
              "drop the labels from the schema deliberately. Refusing to build.")
        sys.exit(1)

    if len(synth) < N_TRAIN_SYNTH + N_VAL_SYNTH:
        print(f"ERROR: need {N_TRAIN_SYNTH + N_VAL_SYNTH} synthetic files, "
              f"found {len(synth)}")
        sys.exit(1)
    if len(val_auth) < N_VAL_AUTH:
        print(f"ERROR: need at least {N_VAL_AUTH} authentic val-source files, "
              f"found {len(val_auth)}")
        sys.exit(1)

    # -- synthetic split: draw N_VAL_SYNTH for val, stratified by difficulty tier
    # (from the filename) so val is not dominated by one tier; the rest are the
    # train pool.
    by_tier = defaultdict(list)
    for rec in synth:
        by_tier[tier_of(rec[0])].append(rec)
    for t in by_tier:
        by_tier[t].sort(key=lambda r: r[0].name)
        rng.shuffle(by_tier[t])

    total = sum(len(v) for v in by_tier.values())
    val_synth = []
    for t in sorted(by_tier):
        take = round(N_VAL_SYNTH * len(by_tier[t]) / total)
        val_synth.extend(by_tier[t][:take])
    # Correct any rounding drift deterministically.
    pool = [r for t in sorted(by_tier) for r in by_tier[t] if r not in val_synth]
    pool.sort(key=lambda r: r[0].name)
    while len(val_synth) < N_VAL_SYNTH:
        val_synth.append(pool.pop(0))
    val_synth = val_synth[:N_VAL_SYNTH]

    val_names = {r[0].name for r in val_synth}
    train_synth = [r for r in synth if r[0].name not in val_names]
    train_synth.sort(key=lambda r: r[0].name)
    rng.shuffle(train_synth)
    train_synth = train_synth[:N_TRAIN_SYNTH]

    # -- authentic val-source split: N_VAL_AUTH -> val, the remainder -> train.
    # Mixing these authentic calls into TRAIN is the mixed2 recipe -- it narrows
    # the synthetic-to-authentic domain gap the model sees during training.
    auth = sorted(val_auth, key=lambda r: r[0].name)
    rng.shuffle(auth)
    val_auth_sel = auth[:N_VAL_AUTH]
    train_auth = auth[N_VAL_AUTH:]

    # -- assemble the mixed2 splits.
    train = train_synth + train_auth
    rng.shuffle(train)
    val = val_synth + val_auth_sel

    # Verify the splits are disjoint by full input text.
    import hashlib
    h = lambda recs: {hashlib.md5(o["input"].encode()).hexdigest() for _p, o in recs}
    ht, hv, hs = h(train), h(val), h(test_auth)
    assert not (ht & hv), f"train/val overlap: {len(ht & hv)}"
    assert not (ht & hs), f"train/test overlap: {len(ht & hs)}"
    assert not (hv & hs), f"val/test overlap: {len(hv & hs)}"

    write_jsonl(TRAIN_DIR / "train_mixed2.jsonl", train)
    write_jsonl(VAL_DIR / "val_mixed2.jsonl", val)
    write_jsonl(TEST_DIR / "test_mixed2.jsonl", test_auth)
    (TEST_DIR / "test_mixed2_files.txt").write_text(
        "\n".join(p.name for p, _ in test_auth) + "\n", encoding="utf-8")

    report(f"TRAIN ({N_TRAIN_SYNTH} synthetic + {len(train_auth)} authentic)", train)
    report(f"VAL ({len(val_synth)} synthetic + {len(val_auth_sel)} authentic)", val)
    report("TEST (authentic, held out)", test_auth)

    print("\nWritten:")
    for d, f in ((TRAIN_DIR, "train_mixed2.jsonl"), (VAL_DIR, "val_mixed2.jsonl"),
                 (TEST_DIR, "test_mixed2.jsonl"), (TEST_DIR, "test_mixed2_files.txt")):
        print(f"  {d / f}")
    print("\nSplits verified disjoint by full-text hash.")
    print("TEST is authentic and never passed to the trainer.")


if __name__ == "__main__":
    main()
