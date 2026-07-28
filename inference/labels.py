"""
labels.py

The single source of truth for label configuration, shared by the inference
pipeline and the benchmark so they can never drift apart. Pure data -- no logic.

Two ideas to keep separate while reading this file:
  * QUERY STRINGS -- the label names we *ask* the model for. Different models
    have slightly different native vocabularies, so the query lists differ.
  * SCORED CONCEPTS -- the labels we actually *measure*. There are 7 base
    concepts (+2 added by this project = 9). NORMALIZED_LABEL maps every query
    string onto one of these concepts, so the number of strings we ask for is
    NOT the number of things being scored.
"""

# The base GLiNER2 model. The fine-tuned adapter is loaded on top of this.
MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"


# -- query-label lists: what each model is ASKED for ----------------------
#
# BASELINE_LABELS has 8 strings but still only covers the 7 base CONCEPTS. The
# extra string is sg_contact_number, which is just a second name for "phone":
# the un-fine-tuned base model natively tags some phones as sg_contact_number,
# so we must ask for that name too or those phones would score as misses and
# unfairly penalise the baseline. Both names fold into SG_PHONE_NUMBER below.
BASELINE_LABELS = ["sg_phone_number", "sg_contact_number", "sg_address", "sg_address_unit_number",
                   "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]

# FINETUNED_LABELS has exactly the 7 base concepts, one string each. The
# fine-tuned model was TRAINED to put every phone under sg_phone_number, so it
# does not need the contact-number synonym -- that merge is something fine-tuning
# bought us.
FINETUNED_LABELS = ["sg_phone_number", "sg_address", "sg_address_unit_number",
                    "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]

# The two concepts this project added on top of the base 7 (the base model does
# not know them). Kept separate so callers that want only the base 7 are
# unaffected; SYNTHETIC_LABELS is the full 9 the keeper is queried with.
SYNTHETIC_EXTRA_LABELS = ["account_number", "full_name"]
SYNTHETIC_LABELS = FINETUNED_LABELS + SYNTHETIC_EXTRA_LABELS


# -- NORMALIZED_LABEL: query string -> the one concept it is scored as --------
#
# This is what makes "8 query strings, 7 scored concepts" work: sg_phone_number
# and sg_contact_number both normalise to SG_PHONE_NUMBER, so the two names for
# phone are measured as one thing. Every method's raw output passes through this
# map before matching, so all three methods are scored on the same concepts.
NORMALIZED_LABEL = {
    "sg_phone_number": "SG_PHONE_NUMBER", "sg_contact_number": "SG_PHONE_NUMBER",  # two names, one concept
    "sg_address": "SG_ADDRESS", "sg_address_unit_number": "SG_ADDRESS_UNIT",
    "sg_address_block_number": "SG_ADDRESS_BLOCK", "sg_postal_code": "SG_POSTAL_CODE",
    "email_address": "EMAIL_ADDRESS", "sg_nric_fin": "SG_NRIC_FIN",
    # The two added concepts. Harmless for any model that never emits them.
    "account_number": "ACCOUNT_NUMBER", "full_name": "FULL_NAME",
}


# -- _FILTER_LABEL: internal plumbing for the precision filters ----------------
#
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
