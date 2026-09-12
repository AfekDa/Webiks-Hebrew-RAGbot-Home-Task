"""Tune the blend WITHOUT re-running the reranker.

We already saved one relevance score per candidate from the reranker run. Those
scores never change, so we can *replay* the "combine the two orders" step with
many different settings in seconds, and pick the best one honestly:

  - pick the setting on the FIRST half of the questions (the "dev" half),
  - then report it once on the SECOND half (the "held-out" half) it never saw.

This is the fair way to choose a knob: if a setting only looks good because it
was hand-picked on the same questions it is scored on, the held-out half exposes
that. Two knobs are swept, both free to replay:

  blend_k : how much the reranker is allowed to override the search order.
            small = reranker matters more; large = search order is kept more.
  depth   : how many of the top candidates get reordered at all; the rest keep
            the search order behind them.

It reads the cache written by eval_baseline.py + eval_reranker.py (candidates +
saved scores). Run it wherever that cache lives (the GPU PC):

    .venv/Scripts/python rag_eval/sweep_blend.py --tag full --dev 100
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Webiks-Hebrew-RAGbot"))
from webiks_hebrew_ragbot.rank_fusion import fuse_orders
import common

KS = (1, 3, 5, 10)


def dedup_to_pages(para_idx_order, para_doc_ids):
    pages, seen = [], set()
    for i in para_idx_order:
        did = para_doc_ids[i]
        if did not in seen:
            seen.add(did); pages.append(did)
    return pages


def load_questions(out):
    """Rebuild, per question: the candidate paragraph indices in search order,
    the reranker score for each, and the accepted (correct) pages."""
    candidates = json.load(open(os.path.join(out, "candidates.json"), encoding="utf-8"))
    scores_by_qi = {}
    for line in open(os.path.join(out, "rerank_progress.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            scores_by_qi[rec["qi"]] = rec["reranker_scores"]
    paras = json.load(open(os.path.join(out, "paras.json"), encoding="utf-8"))
    para_doc_ids = [p["doc_id"] for p in paras]
    rows = []
    for qi, c in enumerate(candidates):
        if qi not in scores_by_qi:
            continue
        scores = scores_by_qi[qi]
        head = c["cand_para_idx"][:len(scores)]      # reranked slots, search order
        tail = c["cand_para_idx"][len(scores):]      # never reranked, kept behind
        # The reranker's own order of the head (matches eval_reranker.py exactly:
        # best score first, ties broken by page id then text so it is stable).
        rerank_order = sorted(range(len(head)), key=lambda j: (
            -float(scores[j]), str(paras[head[j]]["doc_id"]), paras[head[j]]["content"]))
        rows.append({"qi": qi, "head": head, "tail": tail,
                     "rerank_order": rerank_order, "accepted": set(c["accepted"])})
    return rows, para_doc_ids


def pages_for(row, para_doc_ids, mode, k, depth):
    """Rebuild the final page order for one question under a setting.
    `depth` = how many top slots to reorder; the rest stay in search order."""
    n = len(row["head"])
    d = min(depth, n)
    reranked_top = [j for j in row["rerank_order"] if j < d]     # the top d, reranker order
    if mode == "replace":
        new_order = reranked_top + list(range(d, n))
    else:  # blend the search order (0..d-1) with the reranker order of those same d
        new_order = fuse_orders(list(range(d)), reranked_top, k) + list(range(d, n))
    para_order = [row["head"][j] for j in new_order] + row["tail"]
    return dedup_to_pages(para_order, para_doc_ids)


def measure(rows, para_doc_ids, mode, k, depth):
    hit = {kk: 0 for kk in KS}; mrr = 0.0
    for row in rows:
        pages = pages_for(row, para_doc_ids, mode, k, depth)
        hits, rr, _ = common.score_ranking(pages, row["accepted"], KS)
        for kk in KS:
            hit[kk] += hits[kk]
        mrr += rr
    n = len(rows)
    return {"hit@1": hit[1]/n, "hit@3": hit[3]/n, "hit@5": hit[5]/n,
            "hit@10": hit[10]/n, "mrr@10": mrr/n}


def baseline(rows, para_doc_ids):
    hit = {kk: 0 for kk in KS}; mrr = 0.0
    for row in rows:
        pages = dedup_to_pages(row["head"] + row["tail"], para_doc_ids)
        hits, rr, _ = common.score_ranking(pages, row["accepted"], KS)
        for kk in KS:
            hit[kk] += hits[kk]
        mrr += rr
    n = len(rows)
    return {"hit@1": hit[1]/n, "hit@3": hit[3]/n, "hit@5": hit[5]/n,
            "hit@10": hit[10]/n, "mrr@10": mrr/n}


def fmt(m):
    return (f"{m['hit@1']*100:6.1f}{m['hit@3']*100:7.1f}{m['hit@5']*100:7.1f}"
            f"{m['hit@10']*100:7.1f}{m['mrr@10']:8.3f}")


def header(title):
    print(f"\n{title}")
    print(f"  {'setting':<20}{'#1':>6}{'top3':>7}{'top5':>7}{'top10':>7}{'MRR':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dev", type=int, default=100, help="first N questions = dev (pick here); rest = held-out")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 40, 80], help="blend strengths to try")
    ap.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20, 50], help="how many top slots to reorder")
    args = ap.parse_args()

    out = common.cache_dir(args.tag)
    rows, para_doc_ids = load_questions(out)
    if len(rows) <= args.dev:
        raise SystemExit(f"need more than --dev={args.dev} questions; cache has {len(rows)}")
    dev, test = rows[:args.dev], rows[args.dev:]
    depth_full = max(len(r["head"]) for r in rows)
    print(f"cache '{args.tag}': {len(rows)} questions "
          f"(dev = first {len(dev)}, held-out = last {len(test)}), reranked depth = {depth_full}")

    base_dev, base_test = baseline(dev, para_doc_ids), baseline(test, para_doc_ids)

    # ---- 1. blend_k sweep on dev, at full depth (matches production) ----
    header(f"[dev] blend strength sweep (full depth {depth_full})")
    print(f"  {'baseline':<20}{fmt(base_dev)}")
    print(f"  {'replace (k=0)':<20}{fmt(measure(dev, para_doc_ids, 'replace', 0, depth_full))}")
    configs = []
    for k in args.ks:
        m = measure(dev, para_doc_ids, "blend", k, depth_full)
        print(f"  {'blend k='+str(k):<20}{fmt(m)}")
        configs.append(("blend", k, depth_full, m))

    # ---- 2. depth sweep on dev, at blend k=5 (cheap extra lever) ----
    header("[dev] depth sweep (blend k=5, reorder only the top D)")
    print(f"  {'baseline':<20}{fmt(base_dev)}")
    for d in args.depths:
        m = measure(dev, para_doc_ids, "blend", 5, d)
        print(f"  {'blend k=5 depth='+str(d):<20}{fmt(m)}")
        configs.append(("blend", 5, d, m))

    # ---- 3. pick on dev by the stated rule, confirm on held-out ----
    # Rule: no Hit@5 regression vs baseline, and improve Hit@1 or MRR.
    tol = 1e-9
    qualified = [c for c in configs
                 if c[3]["hit@5"] >= base_dev["hit@5"] - tol
                 and (c[3]["hit@1"] > base_dev["hit@1"] + tol
                      or c[3]["mrr@10"] > base_dev["mrr@10"] + tol)]
    print("\n" + "=" * 64)
    if not qualified:
        print("DEV VERDICT: no setting beats baseline (Hit@1 or MRR up, Hit@5 not down).")
        print("Recommendation: do NOT ship BGE. Pivot to BM25 + dense with RRF.")
        return
    best = max(qualified, key=lambda c: (c[3]["mrr@10"], c[3]["hit@1"]))
    mode, k, depth, m_dev = best
    label = f"{mode} k={k} depth={depth}"
    m_test = measure(test, para_doc_ids, mode, k, depth)
    print(f"DEV PICK: {label}")
    header("Held-out confirmation (last questions, never used to pick)")
    print(f"  {'baseline':<20}{fmt(base_test)}")
    print(f"  {label:<20}{fmt(m_test)}")
    d1 = (m_test["hit@1"] - base_test["hit@1"]) * 100
    d5 = (m_test["hit@5"] - base_test["hit@5"]) * 100
    dm = m_test["mrr@10"] - base_test["mrr@10"]
    print(f"\n  held-out change: Hit@1 {d1:+.1f}, Hit@5 {d5:+.1f}, MRR {dm:+.3f}")
    ship = (d5 >= -tol) and (d1 > tol or dm > tol)
    print(f"  SHIP: {'yes' if ship else 'no'} "
          f"(rule: Hit@5 not down AND Hit@1 or MRR up, on held-out)")
    if not ship:
        print("  -> BGE not salvageable on held-out. Pivot to BM25 + dense with RRF.")


if __name__ == "__main__":
    main()
