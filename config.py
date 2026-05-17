
# ── CLO Model — local HuggingFace text-generation ────────────────────────────
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# ── Skill Classifier — local zero-shot NLI model ─────────────────────────────

CLASSIFIER_MODEL_ID = "facebook/bart-large-mnli"

# ── Bloom's Taxonomy Verbs (used for CLO generation prompt) ──────────────────
BLOOMS_VERBS = ["Explain", "Apply", "Analyze", "Implement", "Create", "Design"]

# ── UI / CLI defaults ─────────────────────────────────────────────────────────
DEFAULT_N_CLOS = 4

# ── HuggingFace pipeline args for CLO generation ─────────────────────────────
GENERATION_DEFAULTS = {
    "temperature"     : 0.1,
    "max_new_tokens"  : 300,
    "do_sample"       : True,
    "return_full_text": False,
}