"""STEP 1 (slow, one-time): build a test subset and cache its embeddings.

Picks N evaluation questions, builds a "haystack" = every paragraph from those
questions' correct pages + random distractor paragraphs up to --haystack size,
then embeds the haystack ONCE with the Hebrew model and saves it. The baseline
and the reranker both reuse this cache, so the slow encoding is paid only once.

Usage (from the project root, C:\\Users\\GIGABYTE\\Documents\\webiks):
    .venv/Scripts/python rag_eval/build_subset.py --questions 200 --haystack 2000 --tag main

Smoke test (tiny, ~3 min):
    .venv/Scripts/python rag_eval/build_subset.py --questions 10 --haystack 60 --tag smoke
"""
import argparse, json, os, random, time
import numpy as np
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=200, help="how many test questions")
    ap.add_argument("--haystack", type=int, default=2000, help="target number of paragraphs to search")
    ap.add_argument("--seq", type=int, default=512, help="max tokens per paragraph (keep 512 for fidelity)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="main", help="name for this cache (folder under rag_eval/cache)")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    out = common.cache_dir(args.tag)
    os.makedirs(out, exist_ok=True)

    print("loading corpus + QA ...")
    corpus = common.load_corpus()
    q2docs = common.load_qa()

    # keep only questions whose accepted pages all exist in the corpus (all do,
    # but be safe), then sample N of them deterministically.
    corpus_pages = set(r["doc_id"] for r in corpus)
    usable = [q for q, docs in q2docs.items() if docs <= corpus_pages]
    questions = random.sample(usable, min(args.questions, len(usable)))
    gold_pages = set()
    for q in questions:
        gold_pages |= q2docs[q]

    gold_paras = [r for r in corpus if r["doc_id"] in gold_pages]
    distractor_pool = [r for r in corpus if r["doc_id"] not in gold_pages]
    n_distract = max(0, args.haystack - len(gold_paras))
    distractors = random.sample(distractor_pool, min(n_distract, len(distractor_pool)))
    haystack = gold_paras + distractors
    random.shuffle(haystack)

    print(f"  questions: {len(questions)} | gold pages: {len(gold_pages)}")
    print(f"  haystack: {len(haystack)} paragraphs "
          f"({len(gold_paras)} from gold pages + {len(distractors)} distractors)")

    # ---- the slow part: embed the haystack once ----
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    print("loading model ...")
    model = SentenceTransformer(common.MODEL_DIR)
    model.max_seq_length = args.seq
    model.eval()

    texts = [r["content"] for r in haystack]
    print(f"embedding {len(texts)} paragraphs (this is the slow step) ...")
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=args.batch, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    print(f"  done in {(time.perf_counter()-t0)/60:.1f} min | shape {emb.shape}")

    # ---- save cache ----
    np.save(os.path.join(out, "para_emb.npy"), emb.astype(np.float32))
    json.dump([{"doc_id": r["doc_id"], "title": r["title"], "content": r["content"],
                "link": r["link"]} for r in haystack],
              open(os.path.join(out, "paras.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump([{"question": q, "accepted": sorted(q2docs[q])} for q in questions],
              open(os.path.join(out, "questions.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"cache written to {out}/  (para_emb.npy, paras.json, questions.json, config.json)")


if __name__ == "__main__":
    main()
