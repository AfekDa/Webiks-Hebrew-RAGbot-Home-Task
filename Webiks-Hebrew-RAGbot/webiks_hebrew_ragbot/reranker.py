"""Optional cross-encoder pass over the search results.

The normal search compares the question and each paragraph *separately* (by their
pre-computed vectors). That is fast but rough. A reranker (a "cross-encoder")
reads the question and a paragraph together and gives one relevance score.
The paired evaluation determines whether this ordering improves retrieval.

The reranker's opinion can be used in two ways (config.RERANK_MODE):
  "blend"   - combine it with the search order, so both opinions count. This is
              the default because on the full corpus it measured better than
              replacing: the search model was trained on these very questions,
              so its order is worth keeping.
  "replace" - use the reranker's order alone.

This class is only built when reranking is turned on (config.RERANK_ENABLED), so
the base system carries no extra model cost when it is off. Candidate count
bounds inference cost: it re-reads the top `RERANK_TOP` results and leaves the
rest in their original order behind them.
"""
import logging
import torch

from . import config
from .document import document_definition_factory
from .rank_fusion import fuse_orders

definitions = document_definition_factory()


class Reranker:
    def __init__(self, model_name: str = None, top_rerank: int = None, max_seq: int = None,
                 mode: str = None, blend_k: int = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or config.RERANK_MODEL
        self.top_rerank = config.RERANK_TOP if top_rerank is None else top_rerank
        max_seq = config.RERANK_MAX_SEQ if max_seq is None else max_seq
        self.mode = config.RERANK_MODE if mode is None else mode
        self.blend_k = config.RERANK_BLEND_K if blend_k is None else blend_k
        if self.top_rerank < 1 or max_seq < 1 or self.blend_k < 1:
            raise ValueError("top_rerank, max_seq and blend_k must be positive")
        if self.mode not in ("blend", "replace"):
            raise ValueError("mode must be blend or replace")
        self.text_field = definitions.field_to_embed          # the paragraph text ("content")
        if config.RERANK_DTYPE == "float16" and not torch.cuda.is_available():
            raise RuntimeError("float16 reranking requires CUDA; set RERANK_DTYPE=float32 on CPU")
        self.model = CrossEncoder(self.model_name, max_length=max_seq)
        if config.RERANK_DTYPE == "float16":
            self.model.model.half()
        logging.info(f"reranker loaded: {self.model_name} "
                     f"(re-reads top {self.top_rerank}, mode={self.mode})")

    def rerank(self, query: str, docs: list) -> list:
        """Reorder Elasticsearch hits using the cross-encoder's relevance to `query`.

        Re-reads the top `top_rerank` hits with the cross-encoder and sorts those
        by the new score. In "blend" mode that new order is then merged with the
        original search order (see rank_fusion.py); in "replace" mode it is used
        as-is. The remaining hits stay unchanged behind them. The hit objects
        themselves are untouched -- only their order changes -- so the rest of the
        pipeline (dedup to pages, answer step) works exactly as before.
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
        if self.mode == "blend":
            order = fuse_orders(list(range(len(head))), order, self.blend_k)
        return [head[i] for i in order] + tail
