"""Unit tests for the shared rank-fusion helpers (no heavy dependencies)."""
import pytest

from webiks_hebrew_ragbot.rank_fusion import fuse_orders, reciprocal_rank_fusion


def test_rrf_agreeing_lists_keep_order():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["a", "b", "c"]], k=60) == ["a", "b", "c"]


def test_rrf_surfaces_item_both_rank_high():
    # "x" is 2nd in both short lists; "a" and "b" are 1st in one but absent in the other.
    # k=1: x = 1/3 + 1/3 = 0.667 ; a = 1/2 = 0.5 ; b = 1/2 = 0.5. x wins.
    result = reciprocal_rank_fusion([["a", "x"], ["b", "x"]], k=1)
    assert result[0] == "x"
    assert set(result) == {"a", "b", "x"}


def test_rrf_includes_items_from_either_list():
    # BM25 finds a page dense never returned; it must still appear in the output.
    result = reciprocal_rank_fusion([["dense_only"], ["bm25_only"]], k=60)
    assert set(result) == {"dense_only", "bm25_only"}


def test_rrf_weights_favor_the_heavier_list():
    even = reciprocal_rank_fusion([["a"], ["b"]], k=1)              # tie -> first-seen order
    weighted = reciprocal_rank_fusion([["a"], ["b"]], k=1, weights=[3, 1])
    assert even == ["a", "b"]
    assert weighted[0] == "a"


def test_rrf_rejects_bad_arguments():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_fuse_orders_still_requires_same_items():
    with pytest.raises(ValueError, match="same items"):
        fuse_orders([1, 2], [1, 3])
