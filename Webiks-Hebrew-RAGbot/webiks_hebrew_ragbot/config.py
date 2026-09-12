import os

EMBEDDING_INDEX = os.getenv("ES_EMBEDDING_INDEX", "embedded_index")
ES_EMBEDDING_INDEX_LENGTH = int(os.getenv("ES_EMBEDDING_INDEX_LENGTH", "1000"))
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "model")


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError("RERANK_ENABLED must be true or false")
    return normalized == "true"


# --- Optional reranker (a more careful second pass over search results) ---
# Off by default: when RERANK_ENABLED is false the system behaves exactly as
# before. RERANK_MODEL can be a Hugging Face id or a local folder path (use a
# local path to run fully offline).
RERANK_ENABLED = _as_bool(os.getenv("RERANK_ENABLED", "false"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP = int(os.getenv("RERANK_TOP", "20"))
RERANK_MAX_SEQ = int(os.getenv("RERANK_MAX_SEQ", "512"))
RERANK_DTYPE = os.getenv("RERANK_DTYPE", "float32")
if RERANK_DTYPE not in ("float32", "float16"):
    raise ValueError("RERANK_DTYPE must be float32 or float16")
# How the reranker's opinion is used:
#   "blend"   - add it to the search order (both opinions count). Default: it
#               measured better than replacing on the full corpus.
#   "replace" - throw the search order away and use the reranker's order only.
RERANK_MODE = os.getenv("RERANK_MODE", "blend").strip().lower()
if RERANK_MODE not in ("blend", "replace"):
    raise ValueError("RERANK_MODE must be blend or replace")
# Blend strength: small = the top of each list matters most (5 measured best).
RERANK_BLEND_K = int(os.getenv("RERANK_BLEND_K", "5"))
if RERANK_BLEND_K < 1:
    raise ValueError("RERANK_BLEND_K must be positive")

