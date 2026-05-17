
from config import BLOOMS_VERBS, GENERATION_DEFAULTS

SYSTEM_PROMPT = (
    "You are an academic assistant who generates clear, measurable "
    "Course Learning Outcomes (CLOs) for any subject."
)

USER_PROMPT_TEMPLATE = """
Generate exactly {no_of_clos} Course Learning Outcomes (CLOs).

Strict Rules:
- Each CLO must start with ONE verb from: {blooms_verbs}
- Each CLO must contain ONLY ONE action verb
- Each CLO must focus on ONE concept only
- Do NOT combine multiple topics
- Do NOT use phrases like 'students will' or 'the course covers'
- Each CLO must be measurable and specific

Each CLO must follow this structure:
[Verb] + [specific concept] + [measurable task]

Bad Example:
Design and implement CNNs, RNNs, GANs, and YOLO models.

Good Example:
Design a convolutional neural network for image classification.

Now generate CLOs from:

Course Content:
{course_content}
"""


def build_messages(
    course_content: str,
    no_of_clos: int,
    blooms_verbs: list = BLOOMS_VERBS,
) -> list[dict]:
    """
    Return a list of chat messages (system + user) ready for the pipeline.

    Parameters
    ----------
    course_content : str   — raw textual course content to analyse.
    no_of_clos     : int   — how many CLOs to generate (from UI slider).
    blooms_verbs   : list  — allowed Bloom's taxonomy verbs.
    """
    user_content = USER_PROMPT_TEMPLATE.format(
        no_of_clos=no_of_clos,
        blooms_verbs=blooms_verbs,
        course_content=course_content.strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]