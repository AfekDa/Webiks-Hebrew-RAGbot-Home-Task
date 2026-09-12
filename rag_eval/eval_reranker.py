"""STEP 3 (the improvement): re-read the top candidates with a smarter model.

This is our one change to the system. It does NOT search again. It takes the
same candidate paragraphs the baseline already found for each question (saved
in candidates.json), and hands the top ones -- together with the question -- to
a second, more careful model called a "reranker" (BGE).

The reranker reads the question and a paragraph AT THE SAME TIME and gives one
number: how well this paragraph answers this question. We re-sort by that
number, collapse to pages the same way the real system does, and measure again.

To keep the comparison airtight, this script also recomputes the BASELINE on the
exact same questions and the exact same candidate lists -- so "before" and
"after" differ only in the ORDER of the candidates, nothing else.

Two knobs keep it fast on a CPU (BGE is heavy):
  --top-rerank N : only re-read the top N candidates; the rest keep their old
                   order behind them (so the page set is identical to baseline).
  --n-questions M: only test the first M questions from the cache.

Run eval_baseline.py first with the same --tag (it saves candidates.json).

Usage:
    .venv/Scripts/python rag_eval/eval_reranker.py --tag main --n-questions 100 --top-rerank 20
"""
import argparse, json, os, time
# Force offline: everything is already on disk, and the network here is flaky.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import common

# Where the downloaded BGE model lives on disk (loaded straight from here).
DEFAULT_MODEL = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "models--BAAI--bge-reranker-v2-m3",
    "snapshots", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
)


def dedup_to_pages(para_idx_order, para_doc_ids):
    """Collapse an ordered list of paragraph indices to unique pages
    (first-seen order) -- exactly the system's rule."""
    pages, seen = [], set()
    for i in para_idx_order:
        did = para_doc_ids[i]
        if did not in seen:
            seen.add(did)
            pages.append(did)
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="reranker model (local folder or hub id)")
    ap.add_argument("--top-rerank", type=int, default=20, dest="top_rerank",
                    help="how many of the top candidates the reranker re-reads")
    ap.add_argument("--n-questions", type=int, default=100, dest="n_questions",
                    help="how many questions to test (from the cache)")
    ap.add_argument("--seq", type=int, default=512, help="most text to read per pair")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU threads (default = physical cores, avoids thrashing)")
    args = ap.parse_args()
    out = common.cache_dir(args.tag)

    paras = json.load(open(os.path.join(out, "paras.json"), encoding="utf-8"))
    candidates = json.load(open(os.path.join(out, "candidates.json"), encoding="utf-8"))
    candidates = candidates[:args.n_questions]
    para_doc_ids = [p["doc_id"] for p in paras]
    print(f"cache '{args.tag}': testing {len(candidates)} questions, "
          f"reranking top {args.top_rerank} of each, {len(paras)} paragraphs", flush=True)

    import torch
    torch.set_num_threads(args.threads)
    from sentence_transformers import CrossEncoder
    print(f"loading reranker (threads={args.threads}) ...", flush=True)
    t_load = time.perf_counter()
    reranker = CrossEncoder(args.model, max_length=args.seq)
    print(f"  loaded in {time.perf_counter()-t_load:.0f}s", flush=True)

    ks = (1, 3, 5, 10)
    base_agg = {k: 0 for k in ks}; base_mrr = 0.0     # baseline on this same subset
    rr_agg = {k: 0 for k in ks};   rr_mrr = 0.0        # reranker
    t0 = time.perf_counter()
    for qi, c in enumerate(candidates):
        cand = c["cand_para_idx"]                       # top-50 in the baseline's order
        accepted = set(c["accepted"])

        # --- baseline: original order, dedup to pages ---
        b_pages = dedup_to_pages(cand, para_doc_ids)
        b_hits, b_rr, _ = common.score_ranking(b_pages, accepted, ks)
        for k in ks: base_agg[k] += int(b_hits[k])
        base_mrr += b_rr

        # --- reranker: re-read the top N, re-sort them, keep the rest behind ---
        head = cand[:args.top_rerank]
        tail = cand[args.top_rerank:]
        pairs = [[c["question"], paras[i]["content"]] for i in head]
        scores = reranker.predict(pairs, batch_size=args.batch, show_progress_bar=False)
        head_sorted = [head[j] for j in sorted(range(len(head)), key=lambda j: -scores[j])]
        r_pages = dedup_to_pages(head_sorted + tail, para_doc_ids)
        r_hits, r_rr, _ = common.score_ranking(r_pages, accepted, ks)
        for k in ks: rr_agg[k] += int(r_hits[k])
        rr_mrr += r_rr

        if (qi + 1) % 5 == 0:
            el = (time.perf_counter() - t0) / 60
            eta = el / (qi + 1) * (len(candidates) - qi - 1)
            print(f"  {qi+1}/{len(candidates)}  ({el:.1f} min, ~{eta:.0f} min left)", flush=True)

    n = len(candidates)
    baseline = {"hit@1": base_agg[1]/n, "hit@3": base_agg[3]/n, "hit@5": base_agg[5]/n,
                "hit@10": base_agg[10]/n, "mrr@10": base_mrr/n}
    result = {"tag": args.tag, "model": args.model, "n_questions": n,
              "top_rerank": args.top_rerank, "n_paragraphs": len(paras),
              "hit@1": rr_agg[1]/n, "hit@3": rr_agg[3]/n, "hit@5": rr_agg[5]/n,
              "hit@10": rr_agg[10]/n, "mrr@10": rr_mrr/n,
              "baseline_same_subset": baseline}
    json.dump(result, open(os.path.join(out, "reranker_results.json"), "w"), indent=2)

    def pct(x): return f"{x*100:5.1f}%"
    rows = [("correct page at #1", "hit@1"), ("correct page in top3", "hit@3"),
            ("correct page in top5", "hit@5"), ("correct page in top10", "hit@10")]
    print(f"\n===== BEFORE vs AFTER  (same {n} questions, top-{args.top_rerank} reranked) =====")
    print(f"  minutes to run : {(time.perf_counter()-t0)/60:.0f}")
    print(f"\n  {'metric':<22}{'before':>9}{'after':>9}{'change':>9}")
    for label, key in rows:
        b, a = baseline[key], result[key]
        print(f"  {label:<22}{pct(b):>9}{pct(a):>9}{(a-b)*100:>+8.1f}")
    b, a = baseline["mrr@10"], result["mrr@10"]
    print(f"  {'MRR@10':<22}{b:>9.3f}{a:>9.3f}{a-b:>+9.3f}")
    print(f"\nsaved -> {out}/reranker_results.json")


if __name__ == "__main__":
    main()
