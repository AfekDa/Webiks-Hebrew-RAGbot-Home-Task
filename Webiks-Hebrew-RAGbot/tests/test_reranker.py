import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

os.environ.setdefault(
    "DOCUMENT_DEFINITION_CONFIG",
    str(Path(__file__).resolve().parents[2] / "Webiks-Hebrew-RAGbot-Demo/app/src/doc-config.json"),
)

from webiks_hebrew_ragbot.reranker import Reranker
from webiks_hebrew_ragbot.engine import Engine


def test_rerank_happens_before_page_deduplication():
    # The second paragraph of page 1 is better than its first paragraph.
    # Selecting pages before reranking would return the wrong excerpt.
    hits = [
        {"_source": {"doc_id": 1, "content": "weak excerpt"}},
        {"_source": {"doc_id": 2, "content": "other page"}},
        {"_source": {"doc_id": 1, "content": "answer excerpt"}},
        {"_source": {"doc_id": 3, "content": "unreranked tail"}},
    ]
    with patch("sentence_transformers.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = [0.1, 0.5, 0.9]
        reranker = Reranker(top_rerank=3)
    engine = Engine.__new__(Engine)
    engine.retrieval_model = Mock()
    engine.elastic_model = Mock()
    engine.elastic_model.search.return_value = hits
    engine.reranker = reranker

    result = engine.search_documents("question", top_k=3)

    assert result == [hits[2]["_source"], hits[1]["_source"], hits[3]["_source"]]
    assert hits[0]["_source"]["content"] == "weak excerpt"
    assert len(encoder.return_value.predict.call_args.args[0]) == 3


def test_missing_paragraph_text_fails():
    with patch("sentence_transformers.CrossEncoder"):
        reranker = Reranker()
    with pytest.raises(KeyError, match="content"):
        reranker.rerank("question", [{"_source": {}}, {"_source": {"content": "text"}}])


def test_high_float16_scores_keep_their_order():
    def predict(pairs, activation_fct=torch.nn.Sigmoid(), **kwargs):
        return activation_fct(torch.tensor([11.0, 12.0], dtype=torch.float16)).numpy()

    with patch("sentence_transformers.CrossEncoder") as encoder:
        encoder.return_value.predict.side_effect = predict
        reranker = Reranker()
    hits = [{"_source": {"doc_id": 1, "content": "first"}}, {"_source": {"doc_id": 2, "content": "second"}}]
    assert reranker.rerank("question", hits) == [hits[1], hits[0]]


def test_score_ties_do_not_depend_on_elasticsearch_order():
    with patch("sentence_transformers.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = [1.0, 1.0]
        reranker = Reranker()
    first = {"_source": {"doc_id": 1, "content": "first"}}
    second = {"_source": {"doc_id": 2, "content": "second"}}
    assert reranker.rerank("question", [second, first]) == [first, second]
    assert reranker.rerank("question", [first, second]) == [first, second]


def test_float16_requires_cuda(monkeypatch):
    from webiks_hebrew_ragbot import config
    monkeypatch.setattr(config, "RERANK_DTYPE", "float16")
    with patch("torch.cuda.is_available", return_value=False), patch("sentence_transformers.CrossEncoder") as encoder:
        with pytest.raises(RuntimeError, match="requires CUDA"):
            Reranker()
        encoder.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"top_rerank": 0}, {"max_seq": 0}, {"top_rerank": -1}])
def test_invalid_limits_fail_before_loading_model(kwargs):
    with patch("sentence_transformers.CrossEncoder") as encoder:
        with pytest.raises(ValueError, match="positive"):
            Reranker(**kwargs)
        encoder.assert_not_called()
