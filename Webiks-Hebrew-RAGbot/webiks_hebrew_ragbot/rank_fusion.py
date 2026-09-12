"""Blend two orderings of the same items into one ("reciprocal rank fusion").

Why this exists: the fast search and the reranker are two different opinions
about which paragraph fits the question best. Simply *replacing* the search
order with the reranker's order turned out to hurt on this data (the search
model was trained on these exact questions, so it is a very strong opinion to
throw away). Blending keeps both opinions: each item gets points from where it
stands in each list, and the points are added up.

    points(item) = 1 / (k + position in list A) + 1 / (k + position in list B)

Positions start at 1. A small `k` lets the top of each list matter a lot; a big
`k` flattens the difference between positions. Items that both lists like end
up on top; an item only one list likes gets pulled up only part-way.

This module has no dependencies on purpose, so the offline evaluation and the
live engine share exactly the same code.
"""


def fuse_orders(order_a: list, order_b: list, k: int = 5) -> list:
    """Return `order_a`'s items in blended order.

    `order_a` and `order_b` must hold the same items (any hashable, e.g. ints
    or hit dicts wrapped as indices). Ties are broken by the position in
    `order_a`, so the result is deterministic and never depends on `order_b`'s
    tie-order alone.
    """
    if set(order_a) != set(order_b) or len(order_a) != len(order_b):
        raise ValueError("both orders must contain the same items")
    if k < 1:
        raise ValueError("k must be positive")
    points = {}
    for pos, item in enumerate(order_a, start=1):
        points[item] = 1.0 / (k + pos)
    for pos, item in enumerate(order_b, start=1):
        points[item] += 1.0 / (k + pos)
    first_pos = {item: pos for pos, item in enumerate(order_a)}
    return sorted(order_a, key=lambda item: (-points[item], first_pos[item]))
