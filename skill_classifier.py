
from __future__ import annotations
from transformers import pipeline
from config import CLASSIFIER_MODEL_ID



# ── Colours ───────────────────────────────────────────────────────────────────
BLOOMS_COLORS = {
    "Remember"  : "#64748b",
    "Understand": "#0ea5e9",
    "Apply"     : "#22c55e",
    "Analyze"   : "#f59e0b",
    "Evaluate"  : "#f97316",
    "Create"    : "#a855f7",
}

CATEGORY_COLORS = {
    "Theoretical Understanding" : "#38bdf8",
    "Practical Implementation"  : "#34d399",
    "Analysis & Problem Solving": "#f59e0b",
    "Design & Creation"         : "#a855f7",
    "Evaluation & Critique"     : "#fb923c",
}

# ── Candidate labels BART classifies against ──────────────────────────────────
SKILL_LABELS = [
    "theoretical understanding of core concepts and principles",
    "practical implementation and hands-on application",
    "analysis and problem solving",
    "design and creation of new solutions or artifacts",
    "evaluation and critical assessment",
]

# Map each label → display name, Bloom's level, skill template
LABEL_META = {
    "theoretical understanding of core concepts and principles": {
        "category": "Theoretical Understanding",
        "bloom"   : "Understand",
    },
    "practical implementation and hands-on application": {
        "category": "Practical Implementation",
        "bloom"   : "Apply",
    },
    "analysis and problem solving": {
        "category": "Analysis & Problem Solving",
        "bloom"   : "Analyze",
    },
    "design and creation of new solutions or artifacts": {
        "category": "Design & Creation",
        "bloom"   : "Create",
    },
    "evaluation and critical assessment": {
        "category": "Evaluation & Critique",
        "bloom"   : "Evaluate",
    },
}

# Minimum confidence to show a skill (filters out irrelevant ones)
MIN_SCORE = 0.10


# ── Topic extractor ───────────────────────────────────────────────────────────

def _extract_key_topics(course_content: str, max_topics: int = 4) -> str:
    import re

    # First try: split on punctuation and list separators
    raw = re.split(r"[,.\n;:\-–()\[\]]", course_content)
    topics = []
    seen   = set()

    for chunk in raw:
        chunk = chunk.strip()
        words = chunk.split()
        if 2 <= len(words) <= 6 and chunk and chunk not in seen:
            if not re.match(r"^[\d\s]+$", chunk):
                topics.append(chunk)
                seen.add(chunk)
        if len(topics) >= max_topics * 3:
            break

    # Second try: if very few topics found (delimiter-free prose),
    # scan for Title Case or ALLCAPS runs of 2–4 words as topic names
    if len(topics) < 3:
        title_runs = re.findall(
            r"(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})", course_content
        )
        for run in title_runs:
            # Skip runs that are just acronyms glued together (< 3 chars per word avg)
            avg_len = sum(len(w) for w in run.split()) / len(run.split())
            if avg_len >= 3 and run not in seen:
                topics.append(run)
                seen.add(run)
            if len(topics) >= max_topics * 3:
                break

    # Prefer title-case phrases
    title_case = [t for t in topics if t[0].isupper()]
    chosen = (title_case if title_case else topics)[:max_topics]

    if not chosen:
        words = course_content.split()
        chosen = [" ".join(words[:4])]

    if len(chosen) == 1:
        return chosen[0]
    return ", ".join(chosen[:-1]) + " and " + chosen[-1]


# ── Skill statement builder ───────────────────────────────────────────────────

def _build_skill_statement(label: str, topics: str) -> str:
    templates = {
        "theoretical understanding of core concepts and principles":
            f"Understand and explain the foundational concepts of {topics}",

        "practical implementation and hands-on application":
            f"Implement and apply practical techniques related to {topics}",

        "analysis and problem solving":
            f"Analyze problems and apply structured reasoning in the context of {topics}",

        "design and creation of new solutions or artifacts":
            f"Design and develop solutions or systems involving {topics}",

        "evaluation and critical assessment":
            f"Evaluate methods, compare approaches, and justify decisions related to {topics}",
    }
    return templates.get(label, f"Apply knowledge of {topics}")


# ── Model loader ──────────────────────────────────────────────────────────────

def load_classifier(model_id: str = CLASSIFIER_MODEL_ID):
    print(f"[SkillClassifier] Loading: {model_id}")
    return pipeline("zero-shot-classification", model=model_id, device=-1)


# ── Core: ONE call to BART on the whole content ───────────────────────────────

def extract_skill_set(clf, course_content: str) -> list[dict]:
    """
    Classify entire course content in ONE BART call.
    Returns 3–5 subject-specific skill statements sorted by confidence.
    Works for any subject — no hardcoded topic lists.
    """
    text   = course_content.strip()[:1024]
    topics = _extract_key_topics(course_content)

    result = clf(text, SKILL_LABELS, multi_label=True)

    skills = []
    for label, score in zip(result["labels"], result["scores"]):
        if score >= MIN_SCORE:
            meta = LABEL_META[label]
            skills.append({
                "skill"   : _build_skill_statement(label, topics),
                "bloom"   : meta["bloom"],
                "category": meta["category"],
                "score"   : score,
            })

    skills = skills[:5]

    # Guarantee at least 3
    if len(skills) < 3:
        for label, score in zip(result["labels"], result["scores"]):
            if len(skills) >= 3:
                break
            meta  = LABEL_META[label]
            entry = {
                "skill"   : _build_skill_statement(label, topics),
                "bloom"   : meta["bloom"],
                "category": meta["category"],
                "score"   : score,
            }
            if entry not in skills:
                skills.append(entry)

    return skills


# ── Text export ───────────────────────────────────────────────────────────────

def format_skill_set_text(results: list[dict]) -> str:
    lines = ["EXPECTED SKILL SET AFTER COURSE COMPLETION", "=" * 50, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['skill']}")
        lines.append(f"   Bloom's Level : {r['bloom']}")
        lines.append(f"   Area          : {r['category']}")
        lines.append(f"   Confidence    : {r['score']:.0%}")
        lines.append("")
    return "\n".join(lines)