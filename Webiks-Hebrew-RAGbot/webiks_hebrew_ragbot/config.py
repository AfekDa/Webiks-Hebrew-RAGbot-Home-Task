import os

EMBEDDING_INDEX = os.getenv("ES_EMBEDDING_INDEX", "embedded_index")
ES_EMBEDDING_INDEX_LENGTH = int(os.getenv("ES_EMBEDDING_INDEX_LENGTH", "1000"))
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "model")


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# --- Optional reranker (a more careful second pass over search results) ---
# Off by default: when RERANK_ENABLED is false the system behaves exactly as
# before. RERANK_MODEL can be a Hugging Face id or a local folder path (use a
# local path to run fully offline).
RERANK_ENABLED = _as_bool(os.getenv("RERANK_ENABLED", "false"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP = int(os.getenv("RERANK_TOP", "20"))
RERANK_MAX_SEQ = int(os.getenv("RERANK_MAX_SEQ", "512"))

