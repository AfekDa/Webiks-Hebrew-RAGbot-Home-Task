import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault(
    "DOCUMENT_DEFINITION_CONFIG",
    str(Path(__file__).resolve().parents[2] / "Webiks-Hebrew-RAGbot-Demo/app/src/doc-config.json"),
)

from webiks_hebrew_ragbot.page_scoring import PageScorer, order_pages
from webiks_hebrew_ragbot.engine import Engine


# ---- the pure scoring rule ----

def test_zero_weights_keep_the_original_order():
    scores = {"a": [0.80], "b": [0.79], "c": [0.78, 0.77]}
    first = {"a": 0, "b": 1, "c": 2}
    assert order_pages(scores, {}, first) == ["a", "b", "c"]


def test_title_match_lifts_the_page_about_the_question():
    # "b" has a slightly weaker best paragraph but a much better title.
    scores = {"a": [0.80], "b": [0.79]}
    first = {"a": 0, "b": 1}
    titles = {"a": 0.50, "b": 0.90}
    assert order_pages(scores, titles, first, title_weight=0.5) == ["b", "a"]
    assert order_pages(scores, titles, first, title_weight=0.0) == ["a", "b"]


def test_second_paragraph_rewards_pages_with_several_matches():
    scores = {"one_hit": [0.80], "two_hits": [0.79, 0.78]}
    first = {"one_hit": 0, "two_hits": 1}
    assert order_pages(scores, {}, first, second_weight=0.5) == ["two_hits", "one_hit"]


def test_margin_gate_stops_far_pages_from_jumping():
    # "hub" is far below on paragraphs but has a perfect title; the gate keeps it back.
    scores = {"a": [0.80], "b": [0.79], "hub": [0.60]}
    first = {"a": 0, "b": 1, "hub": 2}
    titles = {"a": 0.5, "b": 0.5, "hub": 1.0}
    assert order_pages(scores, titles, first, title_weight=1.0, margin=None)[0] == "hub"
    assert order_pages(scores, titles, first, title_weight=1.0, margin=0.05) == ["a", "b", "hub"]


def test_ties_keep_original_order():
    scores = {"x": [0.8], "y": [0.8]}
    assert order_pages(scores, {}, {"y": 0, "x": 1}, title_weight=0.5) == ["y", "x"]


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        order_pages({"a": [0.5]}, {}, {"a": 0}, title_weight=-1)


# ---- inside the engine ----

def _fake_model(vectors):
    """A retrieval model whose encode() returns fixed vectors per text."""
    model = Mock()
    model.encode.side_effect = lambda texts, **kw: (
        np.array(vectors[texts]) if isinstance(texts, str)
        else np.array([vectors[t] for t in texts]))
    return model


def test_engine_orders_pages_by_title_and_keeps_best_paragraph():
    # Two pages. Page 2's best paragraph is a touch weaker, but its title matches
    # the question; with page scoring on, page 2 should come out first, and the
    # paragraph kept for each page must still be its first (best) one.
    q = [1.0, 0.0]
    vectors = {"question": q, "title A": [0.0, 1.0], "title B": [1.0, 0.0]}
    hits = [
        {"_score": 1.80, "_source": {"doc_id": 1, "title": "title A", "content": "A best"}},
        {"_score": 1.79, "_source": {"doc_id": 2, "title": "title B", "content": "B best"}},
        {"_score": 1.70, "_source": {"doc_id": 1, "title": "title A", "content": "A second"}},
    ]
    engine = Engine.__new__(Engine)
    engine.retrieval_model = _fake_model(vectors)
    engine.elastic_model = Mock()
    engine.elastic_model.search.return_value = hits
    engine.reranker = None
    engine.page_scorer = PageScorer(engine.retrieval_model, title_weight=0.5, second_weight=0.0, margin=0.05)

    result = engine.search_documents("question", top_k=2)

    assert [d["doc_id"] for d in result] == [2, 1]
    assert result[1]["content"] == "A best"          # not "A second"


def test_engine_page_scoring_off_is_the_original_behaviour():
    hits = [
        {"_score": 1.80, "_source": {"doc_id": 1, "title": "t1", "content": "p1"}},
        {"_score": 1.79, "_source": {"doc_id": 2, "title": "t2", "content": "p2"}},
    ]
    engine = Engine.__new__(Engine)
    engine.retrieval_model = _fake_model({"question": [1.0, 0.0]})
    engine.elastic_model = Mock()
    engine.elastic_model.search.return_value = hits
    engine.reranker = None
    engine.page_scorer = None
    assert [d["doc_id"] for d in engine.search_documents("question", top_k=2)] == [1, 2]


def test_single_hit_is_returned_unchanged():
    scorer = PageScorer(_fake_model({}), title_weight=0.5, second_weight=0.0, margin=0.05)
    hit = [{"_score": 1.8, "_source": {"doc_id": 1, "title": "t", "content": "c"}}]
    assert scorer.reorder(np.array([1.0, 0.0]), hit) == hit
