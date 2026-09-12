"""The page-scoring rule, as one pure function with no dependencies.

    page score = best paragraph + second_weight * second-best paragraph
                                + title_weight  * title match

Kept dependency-free on purpose so the live engine (page_scoring.py) and the
offline evaluation (rag_eval/eval_page_scoring.py) share exactly this code.
See page_scoring.py for the reasoning behind each term.
"""


def order_pages(page_para_scores: dict, page_title_sim: dict, page_first_pos: dict,
                second_weight: float = 0.0, title_weight: float = 0.0, margin=None) -> list:
    """Return page ids best-first.

    page_para_scores : page -> list of its candidate paragraphs' similarity scores
    page_title_sim   : page -> similarity between the question and the page title
    page_first_pos   : page -> position of its first paragraph in the original
                       list (ties keep the original order, so this is deterministic)
    margin           : if set, only pages whose best paragraph is within `margin`
                       of the top score compete on the full score; the rest keep
                       their original order behind them (guards against generic
                       "hub" pages with broad titles jumping up from far below).
    """
    if not page_para_scores:
        return []
    if second_weight < 0 or title_weight < 0 or (margin is not None and margin < 0):
        raise ValueError("weights and margin must be non-negative")
    top = max(max(v) for v in page_para_scores.values())
    scored = []
    for page, vals in page_para_scores.items():
        vals = sorted(vals, reverse=True)
        best = vals[0]
        s = best
        if second_weight and len(vals) > 1:
            s += second_weight * vals[1]
        if title_weight:
            s += title_weight * page_title_sim.get(page, 0.0)
        if margin is not None and best < top - margin:
            s = best - 10.0          # out of the race: stays behind, in original order
        scored.append((s, page))
    scored.sort(key=lambda t: (-t[0], page_first_pos[t[1]]))
    return [page for _, page in scored]
