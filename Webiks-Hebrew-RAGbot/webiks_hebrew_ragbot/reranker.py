"""Optional reranker: a second, more careful pass over the search results.

The normal search compares the question and each paragraph *separately* (by their
pre-computed vectors). That is fast but rough. A reranker (a "cross-encoder")
reads the question and a paragraph *together* and gives one relevance score, which
is more accurate -- so the best paragraph is more likely to end up on top.

This class is only built when reranking is turned on (config.RERANK_ENABLED), so
the base system carries no extra cost or dependency when it is off. To keep a
single query responsive on a CPU, it re-reads only the top `RERANK_TOP` results
and leaves the rest in their original order behind them.
"""
import logging

from . import config
from .document import document_definition_factory

definitions = document_definition_factory()


class Reranker:
    def __init__(self, model_name: str = None, top_rerank: int = None, max_seq: int = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or config.RERANK_MODEL
        self.top_rerank = top_rerank or config.RERANK_TOP
        self.text_field = definitions.field_to_embed          # the paragraph text ("content")
        self.model = CrossEncoder(self.model_name, max_length=max_seq or config.RERANK_MAX_SEQ)
        logging.info(f"reranker loaded: {self.model_name} (re-reads top {self.top_rerank})")

    def rerank(self, query: str, docs: list) -> list:
        """Reorder Elasticsearch hits by real relevance to `query`.

        Re-reads the top `top_rerank` hits with the cross-encoder, sorts those by
        the new score, and keeps the remaining hits unchanged behind them. The hit
        objects themselves are untouched -- only their order changes -- so the rest
        of the pipeline (dedup to pages, answer step) works exactly as before.
        """
        if not docs or len(docs) < 2:
            return docs
        head = docs[:self.top_rerank]
        tail = docs[self.top_rerank:]
        pairs = [[query, d["_source"].get(self.text_field, "")] for d in head]
        scores = self.model.predict(pairs)
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        return [head[i] for i in order] + tail
