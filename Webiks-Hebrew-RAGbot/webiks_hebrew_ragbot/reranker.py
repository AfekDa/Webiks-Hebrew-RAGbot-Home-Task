"""Optional cross-encoder pass over the search results.

The normal search compares the question and each paragraph *separately* (by their
pre-computed vectors). That is fast but rough. A reranker (a "cross-encoder")
reads the question and a paragraph together and gives one relevance score.
The paired evaluation determines whether this ordering improves retrieval.

This class is only built when reranking is turned on (config.RERANK_ENABLED), so
the base system carries no extra model cost when it is off. Candidate count
bounds inference cost: it re-reads the top `RERANK_TOP` results and leaves the
rest in their original order behind them.
"""
import logging
import torch

from . import config
from .document import document_definition_factory

definitions = document_definition_factory()


class Reranker:
    def __init__(self, model_name: str = None, top_rerank: int = None, max_seq: int = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or config.RERANK_MODEL
        self.top_rerank = config.RERANK_TOP if top_rerank is None else top_rerank
        max_seq = config.RERANK_MAX_SEQ if max_seq is None else max_seq
        if self.top_rerank < 1 or max_seq < 1:
            raise ValueError("top_rerank and max_seq must be positive")
        self.text_field = definitions.field_to_embed          # the paragraph text ("content")
        if config.RERANK_DTYPE == "float16" and not torch.cuda.is_available():
            raise RuntimeError("float16 reranking requires CUDA; set RERANK_DTYPE=float32 on CPU")
        self.model = CrossEncoder(self.model_name, max_length=max_seq)
        if config.RERANK_DTYPE == "float16":
            self.model.model.half()
        logging.info(f"reranker loaded: {self.model_name} (re-reads top {self.top_rerank})")

    def rerank(self, query: str, docs: list) -> list:
        """Reorder Elasticsearch hits by cross-encoder relevance to `query`.

        Re-reads the top `top_rerank` hits with the cross-encoder, sorts those by
        the new score, and keeps the remaining hits unchanged behind them. The hit
        objects themselves are untouched -- only their order changes -- so the rest
        of the pipeline (dedup to pages, answer step) works exactly as before.
        """
        if not docs or len(docs) < 2:
            return docs
        head = docs[:self.top_rerank]
        tail = docs[self.top_rerank:]
        pairs = [[query, d["_source"][self.text_field]] for d in head]
        # Sort raw logits: a float16 sigmoid can round distinct high scores to 1.
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False,
                                    activation_fct=torch.nn.Identity())
        order = sorted(range(len(head)), key=lambda i: (
            -float(scores[i]), str(head[i]["_source"]["doc_id"]),
            head[i]["_source"][self.text_field],
        ))
        return [head[i] for i in order] + tail
