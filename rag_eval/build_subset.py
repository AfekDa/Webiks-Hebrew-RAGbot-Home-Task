"""STEP 1 (the slow one, run once): get a practice quiz ready.

This sets up a test we can use to check how good the system is at finding the
right page for a question. It does three things:

  1. picks some real questions to test with (we already know the correct page
     for each, so we can grade later);
  2. gathers a set of pages to search through -- the pages that hold the
     answers, plus lots of random other pages mixed in so the test is not too
     easy;
  3. goes through every page in that set and "studies" it once (turns it into
     the number form the system searches with). This is the slow part, so we
     save it to disk and never repeat it -- every later test reuses it.

Usage (from the project root, C:\\Users\\GIGABYTE\\Documents\\webiks):
    .venv/Scripts/python rag_eval/build_subset.py --questions 200 --pages 2000 --tag main

Quick tiny run to check it works (~3 min):
    .venv/Scripts/python rag_eval/build_subset.py --questions 10 --pages 60 --tag smoke
"""
import argparse, json, os, random, time
import numpy as np
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=200, help="how many questions to test with")
    ap.add_argument("--pages", type=int, default=2000, help="how many pages to search through in total")
    ap.add_argument("--seq", type=int, default=512, help="most text to read per page (leave at 512)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="main", help="a name for this saved quiz (folder under rag_eval/cache)")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    out = common.cache_dir(args.tag)
    os.makedirs(out, exist_ok=True)

    print("loading the pages and the questions ...")
    corpus = common.load_corpus()
    q2docs = common.load_qa()

    # Only keep questions whose correct pages are actually in our page set
    # (they all are, but just in case), then pick the questions at random.
    # The random seed is fixed, so the same run always picks the same ones.
    corpus_pages = set(r["doc_id"] for r in corpus)
    usable = [q for q, docs in q2docs.items() if docs <= corpus_pages]
    questions = random.sample(usable, min(args.questions, len(usable)))

    # The pages that hold the answers to the chosen questions.
    answer_pages = set()
    for q in questions:
        answer_pages |= q2docs[q]

    # Build the set of pages to search: all the answer pages, plus random
    # other pages ("extras") mixed in until we reach the size we asked for.
    answer_paras = [r for r in corpus if r["doc_id"] in answer_pages]
    other_paras = [r for r in corpus if r["doc_id"] not in answer_pages]
    n_extras = max(0, args.pages - len(answer_paras))
    extras = random.sample(other_paras, min(n_extras, len(other_paras)))
    search_pages = answer_paras + extras
    random.shuffle(search_pages)

    print(f"  questions: {len(questions)} | pages that hold answers: {len(answer_pages)}")
    print(f"  pages to search: {len(search_pages)} "
          f"({len(answer_paras)} that hold answers + {len(extras)} random extras)")

    # ---- the slow part: "study" every page once ----
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    print("loading the Hebrew model ...")
    model = SentenceTransformer(common.MODEL_DIR)
    model.max_seq_length = args.seq
    model.eval()

    texts = [r["content"] for r in search_pages]
    print(f"studying {len(texts)} pages (this is the slow step) ...")
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=args.batch, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    print(f"  done in {(time.perf_counter()-t0)/60:.1f} min | shape {emb.shape}")

    # ---- save everything so later tests are fast ----
    np.save(os.path.join(out, "para_emb.npy"), emb.astype(np.float32))
    json.dump([{"doc_id": r["doc_id"], "title": r["title"], "content": r["content"],
                "link": r["link"]} for r in search_pages],
              open(os.path.join(out, "paras.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump([{"question": q, "accepted": sorted(q2docs[q])} for q in questions],
              open(os.path.join(out, "questions.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"saved to {out}/  (para_emb.npy, paras.json, questions.json, config.json)")


if __name__ == "__main__":
    main()
