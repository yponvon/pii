"""
labels.py

The single source of truth for label configuration, shared by the inference
pipeline and the benchmark.

There are 7 base label concepts, plus 2 this project added (= 9). 

Both GLiNER2 methods (the zero-shot baseline and the fine-tuned model) are queried with
the SAME label set, so the benchmark's only variable is the fine-tuning itself --
no method is given an extra or synonym label. 

NORMALIZED_LABEL then maps each model label onto its reporting concept (mostly a lower->upper case change because the model reports in lower case, plus
the address subtypes to their reported names), so all three methods are scored on the same concept names.
"""

# The base GLiNER2 model. The fine-tuned adapter is loaded on top of this.
MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"


# -- Query-label lists: what the models are asked for ----------
# LABELS_7 = the 7 base concepts (one string each). 
# LABELS_9 adds the 2 concepts this project introduced (account_number, full_name) for the full 9-label run. 
LABELS_7 = ["sg_phone_number", "sg_address", "sg_address_unit_number",
                    "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]
EXTRA_LABELS = ["account_number", "full_name"]
LABELS_9 = LABELS_7 + EXTRA_LABELS


# -- NORMALIZED_LABEL: model label -> the reporting concept it is scored as ----
# Mostly a lower-case -> upper-case standardisation. Every method's raw output
# passes through this before matching, so all three are scored on the same
# concept names.
NORMALIZED_LABEL = {
    "sg_phone_number": "SG_PHONE_NUMBER",
    "sg_address": "SG_ADDRESS", "sg_address_unit_number": "SG_ADDRESS_UNIT",
    "sg_address_block_number": "SG_ADDRESS_BLOCK", "sg_postal_code": "SG_POSTAL_CODE",
    "email_address": "EMAIL_ADDRESS", "sg_nric_fin": "SG_NRIC_FIN",
    # The two added concepts. Harmless for any model that never emits them.
    "account_number": "ACCOUNT_NUMBER", "full_name": "FULL_NAME",
}


# -- _FILTER_LABEL: internal plumbing for the precision filters ----------------
# Maps each scored concept to the key its content filter looks up. It exists for
# one narrow reason: ACCOUNT_NUMBER and FULL_NAME have no shape filter, so this
# mapping lets passes_validity() look them up without raising a KeyError. Only
# postprocessing.py / pipeline.py use it; the leading underscore marks it private.
_FILTER_LABEL = {
    "SG_PHONE_NUMBER": "SG_PHONE_NUMBER", "SG_ADDRESS": "SG_ADDRESS",
    "SG_ADDRESS_UNIT": "SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK": "SG_ADDRESS_BLOCK_NUMBER",
    "SG_POSTAL_CODE": "SG_POSTAL_CODE", "EMAIL_ADDRESS": "EMAIL_ADDRESS", "SG_NRIC_FIN": "SG_NRIC_FIN",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER", "FULL_NAME": "FULL_NAME",
}
