import os

EMBEDDING_INDEX = os.getenv("ES_EMBEDDING_INDEX", "embedded_index")
ES_EMBEDDING_INDEX_LENGTH = int(os.getenv("ES_EMBEDDING_INDEX_LENGTH", "1000"))
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "model")


def _as_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return normalized == "true"


# --- Optional page scoring (rank pages by more than their single best paragraph) ---
# Off by default. Uses the SAME trained model to add a title match and the
# second-best paragraph to each page's score; only the order of the retrieved
# candidates changes. See page_scoring.py. The default weights are the ones
# chosen on 250 development questions and confirmed on 250 held-out questions
# over the full corpus (rag_eval/results/page_scoring_500).
PAGE_SCORING_ENABLED = _as_bool(os.getenv("PAGE_SCORING_ENABLED", "false"), "PAGE_SCORING_ENABLED")
PAGE_SCORING_TITLE_WEIGHT = float(os.getenv("PAGE_SCORING_TITLE_WEIGHT", "0.25"))
PAGE_SCORING_SECOND_WEIGHT = float(os.getenv("PAGE_SCORING_SECOND_WEIGHT", "0.25"))
_margin = os.getenv("PAGE_SCORING_MARGIN", "0.05").strip().lower()
PAGE_SCORING_MARGIN = None if _margin in ("", "none", "off") else float(_margin)
if PAGE_SCORING_TITLE_WEIGHT < 0 or PAGE_SCORING_SECOND_WEIGHT < 0 \
        or (PAGE_SCORING_MARGIN is not None and PAGE_SCORING_MARGIN < 0):
    raise ValueError("PAGE_SCORING_* weights and margin must be non-negative")

