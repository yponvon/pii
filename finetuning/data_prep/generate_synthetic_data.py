"""
generate_synthetic_data.py

A representative, self-contained example of how the synthetic training corpus
was produced. The full corpus (~1,000 transcripts) was authored in batches like
this one; this file is one cleaned batch you can read and run to see the method.

Method (see DATA.md for the full write-up):
  1. Each record is a hand-authored SP Group call-centre transcript covering a
     realistic scenario across a difficulty tier (normal / medium / hard, plus
     "negative" calls that contain no PII at all). Scenario variety is what
     teaches the model the many ways PII is spoken (whole, in fragments,
     digit-by-digit, spelled out, read back for confirmation).
  2. Each transcript is hand-labelled: the `entities` dict lists, per label, the
     exact surface strings that are PII. Duplicates are meaningful -- a value
     that occurs N times in the text is listed N times (the "gold invariant"),
     so read-backs are represented.
  3. Records are written one JSON object per file:
         {"input": <transcript>, "output": {"entities": {<label>: [<str>, ...]}}}
     This is the exact shape the model trains on and the pipeline is scored on
     (see data/train/example.json).

Output: data/generated/<name>.json  (offline; the data/ folder is gitignored).
Like the notebook, this writes to the data/generated/ staging folder. Review the
records, then move approved ones into data/train/synthetic/ (or data/val/synthetic/)
before running build_splits.py.

Usage:
  python finetuning/data_prep/generate_synthetic_data.py
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"

# Each record: (filename, tier, transcript, entities).
# Values are listed once per occurrence so gold counts match the text exactly.
RECORDS = [
    (
        "example_normal_address_update.json", "normal",
        "SPEAKER_00: Good afternoon, SP Group, this is Mei speaking, how can I help you?\n"
        "SPEAKER_01: Hi, I've moved and want to update my address.\n"
        "SPEAKER_00: Sure, what's the new address?\n"
        "SPEAKER_01: Block 45 Riverside Road, #12-08, Singapore 610045.\n"
        "SPEAKER_00: Block 45 Riverside Road, unit 12-08, postal 610045 -- updated.\n"
        "SPEAKER_01: Great, thank you.\n"
        "SPEAKER_00: You're welcome, have a good day.",
        {
            "sg_address": ["Block 45 Riverside Road"],
            "sg_address_block_number": ["45"],
            "sg_address_unit_number": ["#12-08", "12-08"],
            "sg_postal_code": ["610045", "610045"],
            "full_name": ["Mei"],
        },
    ),
    (
        "example_medium_phone_readback.json", "medium",
        "SPEAKER_00: SP Group, Daniel speaking.\n"
        "SPEAKER_01: Hi, please update my contact number.\n"
        "SPEAKER_00: Go ahead.\n"
        "SPEAKER_01: It's nine one two three four five six seven.\n"
        "SPEAKER_00: Let me read that back, 9123 4567?\n"
        "SPEAKER_01: Yes that's right.\n"
        "SPEAKER_00: Done, thank you.",
        {
            # spoken form and the digit read-back both count (gold invariant)
            "sg_phone_number": ["nine one two three four five six seven", "9123 4567"],
            "full_name": ["Daniel"],
        },
    ),
    (
        "example_negative_billing_faq.json", "normal (negative)",
        "SPEAKER_00: Good morning, SP Group, Charmaine here.\n"
        "SPEAKER_01: Hi, does a ceiling fan use much electricity compared to aircon?\n"
        "SPEAKER_00: A fan uses far less than air conditioning, generally.\n"
        "SPEAKER_01: Good to know, thanks.\n"
        "SPEAKER_00: You're welcome.",
        {},  # no PII: negatives teach the model not to over-redact
    ),
]


def _standalone_counts(entities):
    """Total gold mentions in a record (a simple integrity read-out)."""
    return sum(len(v) for v in entities.values())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, tier, text, entities in RECORDS:
        # Integrity check: every listed value must actually occur in the text.
        for label, values in entities.items():
            for v in values:
                assert v in text, f"{fname}: gold value {v!r} ({label}) not found in transcript"
        record = {"input": text, "output": {"entities": entities}}
        with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        print(f"wrote {fname:<45} tier={tier:<20} gold_mentions={_standalone_counts(entities)}")
    print(f"\n{len(RECORDS)} records -> {OUT_DIR}")


if __name__ == "__main__":
    main()
