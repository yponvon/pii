"""Build the train/val JSONL splits from the data/ folders.

Folder = split. Whatever transcripts you drop into data/train/ and data/val/
(one record per .json, in the {"input", "output"} schema) become the training and
validation sets. There are NO hardcoded counts, seeds, or sampling -- the split
is exactly what is in the folders, so you grow or reshape it just by adding or
removing files.

    data/train/  -> data/train.jsonl      (every *.json under here, recursively)
    data/val/    -> data/val.jsonl
    data/test/   frozen benchmark gold (test_gold_419.jsonl) -- NOT built here.

Subfolders are globbed recursively, so you can keep provenance without changing
anything -- the shipped layout is:

    data/train/synthetic/   data/train/authentic/
    data/val/synthetic/     data/val/authentic/

The committed example.json in each folder is a fake shape demo and is skipped.

Output is one JSON object per line, {"input": ..., "output": {"entities": ...}},
written with ensure_ascii=False; source records pass through verbatim.

Correctness checks (no magic numbers -- just safety):
  * every source file must parse and carry both "input" and "output";
  * train and val must be disjoint by full input text (the same transcript in
    both folders fails the build);
  * warns if the training set has no gold for account_number / full_name, since a
    9-label model trained without them learns to suppress them on real text.

Usage:
  python3 build_splits.py
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
# Only train/val are built here; test is a frozen gold file, never rebuilt.
SPLIT_DIRS = {"train": DATA / "train", "val": DATA / "val"}

LABELS = [
    "sg_phone_number", "sg_nric_fin", "sg_address", "sg_postal_code",
    "sg_address_unit_number", "sg_address_block_number", "email_address",
    "account_number", "full_name",
]
NEW_LABELS = ("account_number", "full_name")


def load_split(folder):
    """Load every *.json under folder (recursively), skipping the committed example."""
    recs, skipped = [], []
    for p in sorted(folder.rglob("*.json")):
        if p.name == "example.json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append((p.name, f"unparseable: {exc}"))
            continue
        if "input" not in obj or "output" not in obj:
            skipped.append((p.name, "missing input/output key"))
            continue
        recs.append((p, obj))
    return recs, skipped


def label_counts(recs):
    mentions, files = Counter(), Counter()
    for _p, obj in recs:
        for lab, vals in obj.get("output", {}).get("entities", {}).items():
            if vals:
                mentions[lab] += len(vals)
                files[lab] += 1
    return mentions, files


def zero_entity(recs):
    return sum(1 for _p, o in recs
               if not any(o.get("output", {}).get("entities", {}).values()))


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for _p, obj in recs:
            fh.write(json.dumps({"input": obj["input"], "output": obj["output"]},
                                ensure_ascii=False) + "\n")


def report(name, recs):
    mentions, files = label_counts(recs)
    lens = [len(o["input"]) for _p, o in recs]
    mean = round(sum(lens) / len(lens)) if lens else 0
    print(f"\n--- {name}: {len(recs)} examples (mean {mean} chars, "
          f"{zero_entity(recs)} zero-entity) ---")
    print(f"  {'label':28} {'mentions':>9} {'files':>7}")
    for lab in LABELS:
        print(f"  {lab:28} {mentions.get(lab, 0):>9} {files.get(lab, 0):>7}")
    print(f"  {'TOTAL':28} {sum(mentions.values()):>9}")


def main():
    built = {}
    for name, folder in SPLIT_DIRS.items():
        if not folder.is_dir():
            print(f"ERROR: {folder} does not exist. Create it and drop .json "
                  f"transcripts inside (see data/{name}/example.json).")
            sys.exit(1)
        recs, skipped = load_split(folder)
        for fn, why in skipped:
            print(f"SKIPPED [{name}] {fn}: {why}")
        if not recs:
            print(f"ERROR: no usable .json files found under {folder}.")
            sys.exit(1)
        built[name] = recs

    # Splits must be disjoint by full input text.
    h = lambda recs: {hashlib.md5(o["input"].encode()).hexdigest() for _p, o in recs}
    dup = h(built["train"]) & h(built["val"])
    if dup:
        print(f"\nERROR: {len(dup)} transcript(s) appear in BOTH data/train/ and "
              f"data/val/. A file must live in exactly one split. Refusing to build.")
        sys.exit(1)

    # A 9-label model needs the two new labels represented in training.
    counts, _ = label_counts(built["train"])
    missing = [l for l in NEW_LABELS if counts.get(l, 0) == 0]
    if missing:
        print(f"\nWARNING: data/train/ has no gold for: {', '.join(missing)}.")
        print("A 9-label model trained without these learns to suppress them on "
              "real text. Add annotated examples, or drop the labels deliberately.")

    for name, recs in built.items():
        write_jsonl(DATA / f"{name}.jsonl", recs)
        report(name, recs)

    print("\nWritten:")
    for name in built:
        print(f"  {DATA / (name + '.jsonl')}")
    print("\nSplits verified disjoint by full-text hash.")
    print("Test set (data/test/test_gold_419.jsonl) is frozen and not built here.")


if __name__ == "__main__":
    main()
