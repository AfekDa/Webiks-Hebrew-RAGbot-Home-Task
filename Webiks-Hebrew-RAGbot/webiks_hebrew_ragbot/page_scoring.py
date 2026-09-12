"""Page scoring: rank pages by more than their single best paragraph.

Today the engine ranks the top-50 paragraphs and a page's rank is simply the
rank of its best paragraph. That ignores two signals the *same* trained model
gives us for free, and that turn out to separate the right page from a
look-alike:

  title  : how well the question matches the page TITLE. Kol Zchut titles name
           the topic ("child allowance", "mourning days"), so a page that is
           *about* the question scores here, while a page with one look-alike
           paragraph does not.
  second : how well the page's SECOND-best paragraph matches. The right page
           usually has several relevant paragraphs; a distractor has one.

    page score = best paragraph + second_weight * second paragraph
                                + title_weight * title match

An optional margin gate lets only pages whose best paragraph is close to the
top compete on the full score; pages further down keep their original order
behind them. This stops a generic "hub" page with a broad title ("guide to
foster care") from jumping up from far below, which is the main way the title
signal was seen to misfire.

Only the ORDER of the same top-50 candidates changes: no new model, no
re-indexing, recall untouched. The scoring rule itself is `order_pages` in
page_order.py (dependency-free), shared with the offline evaluation
(rag_eval/eval_page_scoring.py) so both behave the same.
"""
import logging

from . import config
from .document import document_definition_factory
from .page_order import order_pages   # re-exported for convenience

definitions = document_definition_factory()
TITLE_FIELD = "title"
_UNSET = object()      # "argument not given" (None is a real value: no margin gate)

__all__ = ["PageScorer", "order_pages"]


class PageScorer:
    """Reorders Elasticsearch hits so that pages come out in `order_pages` order."""

    def __init__(self, retrieval_model, title_weight: float = None, second_weight: float = None,
                 margin=_UNSET):
        self.model = retrieval_model
        self.title_weight = config.PAGE_SCORING_TITLE_WEIGHT if title_weight is None else title_weight
        self.second_weight = config.PAGE_SCORING_SECOND_WEIGHT if second_weight is None else second_weight
        self.margin = config.PAGE_SCORING_MARGIN if margin is _UNSET else margin
        if self.title_weight < 0 or self.second_weight < 0 or (self.margin is not None and self.margin < 0):
            raise ValueError("page scoring weights and margin must be non-negative")
        if self.title_weight and TITLE_FIELD not in definitions.saved_fields:
            raise ValueError(f"page scoring needs the '{TITLE_FIELD}' field to be saved in the index")
        self.id_field = definitions.identifier
        logging.info(f"page scoring on: title x{self.title_weight}, second x{self.second_weight}, "
                     f"margin {self.margin}")

    def reorder(self, query_embedding, hits: list) -> list:
        """Group hits by page, score pages, and return the hits with pages in the
        new order. Within a page the hits keep their original order, so the
        paragraph the pipeline keeps for each page is unchanged."""
        if len(hits) < 2:
            return hits
        groups, scores, first_pos, titles = {}, {}, {}, {}
        for pos, hit in enumerate(hits):
            page = hit["_source"][self.id_field]
            groups.setdefault(page, []).append(hit)
            # Elasticsearch scores are cosine + 1.0; use plain cosine like the eval.
            scores.setdefault(page, []).append(float(hit["_score"]) - 1.0)
            first_pos.setdefault(page, pos)
            titles.setdefault(page, hit["_source"].get(TITLE_FIELD) or "")
        title_sim = {}
        if self.title_weight:
            pages = list(titles)
            t_emb = self.model.encode([titles[p] for p in pages], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
            sims = t_emb @ query_embedding
            title_sim = {p: float(sims[i]) for i, p in enumerate(pages)}
        order = order_pages(scores, title_sim, first_pos,
                            self.second_weight, self.title_weight, self.margin)
        return [hit for page in order for hit in groups[page]]
