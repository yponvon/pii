"""
labels.py

Label configuration for the PII pipeline: the query-label lists sent to the
model, the canonical-name map (model label -> reporting label), and the base
model path. Kept separate so both the pipeline and the benchmark import the same
single source of truth.
"""

MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"

# -- query-label lists ----------------------------------------------------
# The zero-shot baseline needs sg_contact_number queried as a synonym of
# sg_phone_number. The fine-tuned model was trained with sg_contact_number
# already merged into sg_phone_number, so it is queried with the 7 base labels
# only. The synthetic corpus adds two more labels (account_number, full_name),
# kept in a separate list so callers that import FINETUNED_LABELS are unaffected.

BASELINE_LABELS = ["sg_phone_number", "sg_contact_number", "sg_address", "sg_address_unit_number",
                   "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]
FINETUNED_LABELS = ["sg_phone_number", "sg_address", "sg_address_unit_number",
                    "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]
SYNTHETIC_EXTRA_LABELS = ["account_number", "full_name"]
SYNTHETIC_LABELS = FINETUNED_LABELS + SYNTHETIC_EXTRA_LABELS

# Model label (lower case) -> canonical reporting label (upper case).
CANON = {
    "sg_phone_number": "SG_PHONE_NUMBER", "sg_contact_number": "SG_PHONE_NUMBER",
    "sg_address": "SG_ADDRESS", "sg_address_unit_number": "SG_ADDRESS_UNIT",
    "sg_address_block_number": "SG_ADDRESS_BLOCK", "sg_postal_code": "SG_POSTAL_CODE",
    "email_address": "EMAIL_ADDRESS", "sg_nric_fin": "SG_NRIC_FIN",
    # Synthetic-corpus labels; additive and harmless for models that never emit them.
    "account_number": "ACCOUNT_NUMBER", "full_name": "FULL_NAME",
}

# Canonical label -> the label name the content filter keys on.
_FILTER_LABEL = {
    "SG_PHONE_NUMBER": "SG_PHONE_NUMBER", "SG_ADDRESS": "SG_ADDRESS",
    "SG_ADDRESS_UNIT": "SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK": "SG_ADDRESS_BLOCK_NUMBER",
    "SG_POSTAL_CODE": "SG_POSTAL_CODE", "EMAIL_ADDRESS": "EMAIL_ADDRESS", "SG_NRIC_FIN": "SG_NRIC_FIN",
    # No format filter for ACCOUNT_NUMBER (per spec, every grouping counts) and
    # none for FULL_NAME. _passes_content_filter only gates the phone, postal,
    # and unit labels, so these pass through untouched. The mapping exists purely
    # so passes_validity() does not raise a KeyError when the model emits them.
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER", "FULL_NAME": "FULL_NAME",
}
