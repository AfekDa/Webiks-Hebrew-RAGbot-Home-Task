"""STEP 2 (fast): measure the baseline (today's system) on the cached subset.

Run build_subset.py first with the same --tag; this step reads its cache.

Encodes the test questions, scores them against the cached haystack exactly the
way the real system does (top-50 paragraphs -> dedup to pages), and reports how
often the correct page lands at #1 / top-3 / top-5 / top-10, plus MRR@10.

It also saves each question's top-50 paragraph candidates, so the page-scoring
step can reorder the SAME candidates without recomputing this stage.

Usage:
    .venv/Scripts/python rag_eval/eval_baseline.py --tag main
"""
import argparse, json, os
import numpy as np
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = common.cache_dir(args.tag)

    emb = np.load(os.path.join(out, "para_emb.npy"))
    paras = json.load(open(os.path.join(out, "paras.json"), encoding="utf-8"))
    questions = json.load(open(os.path.join(out, "questions.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    para_doc_ids = [p["doc_id"] for p in paras]
    print(f"loaded cache '{args.tag}': {len(paras)} paragraphs, {len(questions)} questions")

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    model = SentenceTransformer(common.MODEL_DIR)
    # Cut questions to the same length limit used when the paragraphs were
    # embedded, so both sides are measured the same way.
    model.max_seq_length = cfg.get("seq", 512)
    model.eval()

    q_texts = [q["question"] for q in questions]
    q_emb = model.encode(q_texts, batch_size=16, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)

    ks = (1, 3, 5, 10)
    agg = {k: 0 for k in ks}
    mrr_sum = 0.0
    candidates = []  # saved for the page-scoring step
    for qi, q in enumerate(questions):
        sims = q_emb[qi] @ emb.T                     # cosine (both normalized)
        ranked_pages, top_para_idx = common.rank_pages(sims, para_doc_ids)
        accepted = set(q["accepted"])
        hits, rr, rank = common.score_ranking(ranked_pages, accepted, ks)
        for k in ks:
            agg[k] += int(hits[k])
        mrr_sum += rr
        candidates.append({
            "question": q["question"],
            "accepted": q["accepted"],
            "cand_para_idx": [int(i) for i in top_para_idx],  # top-50 paragraph indices into paras.json
        })

    n = len(questions)
    result = {
        "tag": args.tag, "n_questions": n, "n_paragraphs": len(paras),
        "hit@1": agg[1]/n, "hit@3": agg[3]/n, "hit@5": agg[5]/n, "hit@10": agg[10]/n,
        "mrr@10": mrr_sum/n,
    }
    json.dump(result, open(os.path.join(out, "baseline_results.json"), "w"), indent=2)
    json.dump(candidates, open(os.path.join(out, "candidates.json"), "w", encoding="utf-8"), ensure_ascii=False)

    print("\n===== BASELINE (today's system) =====")
    print(f"  questions evaluated : {n}")
    print(f"  haystack size       : {len(paras)} paragraphs")
    print(f"  correct page at #1  : {result['hit@1']*100:5.1f}%")
    print(f"  correct page in top3: {result['hit@3']*100:5.1f}%")
    print(f"  correct page in top5: {result['hit@5']*100:5.1f}%")
    print(f"  correct page in top10:{result['hit@10']*100:5.1f}%")
    print(f"  MRR@10              : {result['mrr@10']:.3f}")
    print(f"\nsaved -> {out}/baseline_results.json and candidates.json")


if __name__ == "__main__":
    main()
