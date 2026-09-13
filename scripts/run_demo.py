"""Launch the real Demo API with local Elasticsearch and a key-free mock LLM."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Webiks-Hebrew-RAGbot-Demo/app/src"


def configure():
    defaults = {
        "DOCUMENT_DEFINITION_CONFIG": str(BACKEND / "doc-config.json"),
        "MODEL_LOCATION": str(ROOT),
        "STATIC_DIR": str(BACKEND / "static"),
        "PATH_TO_ES_INITIAL_VALUES": str(ROOT / "Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json"),
        "DOCKER_ES_SCHEME": "http", "DOCKER_ES_HOST": "127.0.0.1", "DOCKER_ES_PORT": "9200",
        "ES_EMBEDDING_INDEX": "hebrew_rag_local_embeddings",
        "ES_EMBEDDING_INDEX_LENGTH": "100000000",
        "CONFIG_INDEX": "hebrew_rag_local_config",
        "CONVERSATIONS_INDEX": "hebrew_rag_local_conversations",
        "UPDATES_INDEX": "hebrew_rag_local_updates",
        "IS_MOCK_GPT_CLIENT": "true",
        # The submitted improvement: page scoring (title match + second paragraph).
        "PAGE_SCORING_ENABLED": "true", "PAGE_SCORING_TITLE_WEIGHT": "0.25",
        "PAGE_SCORING_SECOND_WEIGHT": "0.25", "PAGE_SCORING_MARGIN": "0.05",
        "CODE_VERSION": "local-page-scoring", "LOG_LEVEL": "INFO",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(BACKEND))


if __name__ == "__main__":
    configure()
    import get_es_client
    es = get_es_client.factory()
    es.info()  # Fail before loading either model if Elasticsearch is unavailable.
    if es.count(index=os.environ["ES_EMBEDDING_INDEX"] + "_*")["count"] == 0:
        raise RuntimeError("No paragraphs indexed; run scripts/seed_demo.py first")
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000)
