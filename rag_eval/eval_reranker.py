"""STEP 3 (the improvement): re-read the top 50 with a smarter model, re-sort.

This is our one change to the system. It does NOT search again. It takes the
same 50 candidate paragraphs the baseline already found for each question
(saved in candidates.json), and hands each one -- together with the question --
to a second, more careful model called a "reranker" (BGE).

The reranker reads the question and a paragraph AT THE SAME TIME and gives one
number: how well this paragraph answers this question. We sort the 50 by that
number, collapse to pages the same way the real system does, and measure again.

Because it reorders the exact same 50 candidates, the comparison is clean:
same questions, same starting list -- only the order changes.

Run eval_baseline.py first with the same --tag (it saves candidates.json).

Usage:
    .venv/Scripts/python rag_eval/eval_reranker.py --tag main
"""
import argparse, json, os, time
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3",
                    help="the reranker model to download/use")
    ap.add_argument("--seq", type=int, default=512, help="most text to read per pair")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    out = common.cache_dir(args.tag)

    paras = json.load(open(os.path.join(out, "paras.json"), encoding="utf-8"))
    candidates = json.load(open(os.path.join(out, "candidates.json"), encoding="utf-8"))
    para_doc_ids = [p["doc_id"] for p in paras]
    print(f"loaded cache '{args.tag}': {len(candidates)} questions, "
          f"{len(paras)} paragraphs")

    from sentence_transformers import CrossEncoder
    import torch
    torch.set_num_threads(os.cpu_count())
    print(f"loading the reranker '{args.model}' (first time downloads it) ...")
    reranker = CrossEncoder(args.model, max_length=args.seq)

    ks = (1, 3, 5, 10)
    agg = {k: 0 for k in ks}
    mrr_sum = 0.0
    t0 = time.perf_counter()
    for qi, c in enumerate(candidates):
        cand_idx = c["cand_para_idx"]                 # the same 50 as the baseline
        pairs = [[c["question"], paras[i]["content"]] for i in cand_idx]
        scores = reranker.predict(pairs, batch_size=args.batch,
                                  show_progress_bar=False)
        # re-sort the 50 candidates by the reranker's score, best first
        reordered = [cand_idx[j] for j in sorted(range(len(cand_idx)),
                                                 key=lambda j: -scores[j])]
        # collapse to unique pages in the new order (same rule as the system)
        ranked_pages, seen = [], set()
        for i in reordered:
            did = para_doc_ids[i]
            if did not in seen:
                seen.add(did)
                ranked_pages.append(did)
        hits, rr, rank = common.score_ranking(ranked_pages, set(c["accepted"]), ks)
        for k in ks:
            agg[k] += int(hits[k])
        mrr_sum += rr
        if (qi + 1) % 20 == 0:
            print(f"  {qi+1}/{len(candidates)} questions "
                  f"({(time.perf_counter()-t0)/60:.1f} min)")

    n = len(candidates)
    result = {
        "tag": args.tag, "model": args.model, "n_questions": n,
        "n_paragraphs": len(paras),
        "hit@1": agg[1]/n, "hit@3": agg[3]/n, "hit@5": agg[5]/n, "hit@10": agg[10]/n,
        "mrr@10": mrr_sum/n,
    }
    json.dump(result, open(os.path.join(out, "reranker_results.json"), "w"), indent=2)

    # ---- print a before/after table if the baseline result is available ----
    base_path = os.path.join(out, "baseline_results.json")
    base = json.load(open(base_path)) if os.path.exists(base_path) else None

    print("\n===== RERANKER (after the improvement) =====")
    print(f"  reranker            : {args.model}")
    print(f"  minutes to run      : {(time.perf_counter()-t0)/60:.1f}")

    def pct(x): return f"{x*100:5.1f}%"
    rows = [("correct page at #1", "hit@1"), ("correct page in top3", "hit@3"),
            ("correct page in top5", "hit@5"), ("correct page in top10", "hit@10")]
    if base:
        print(f"\n  {'metric':<22}{'before':>9}{'after':>9}{'change':>9}")
        for label, key in rows:
            b, a = base[key], result[key]
            print(f"  {label:<22}{pct(b):>9}{pct(a):>9}{(a-b)*100:>+8.1f}")
        b, a = base["mrr@10"], result["mrr@10"]
        print(f"  {'MRR@10':<22}{b:>9.3f}{a:>9.3f}{a-b:>+9.3f}")
    else:
        for label, key in rows:
            print(f"  {label:<22}{pct(result[key]):>9}")
        print(f"  {'MRR@10':<22}{result['mrr@10']:>9.3f}")

    print(f"\nsaved -> {out}/reranker_results.json")


if __name__ == "__main__":
    main()
