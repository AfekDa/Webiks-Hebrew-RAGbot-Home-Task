"""STEP 1 (the slow one, run once): get a practice quiz ready.

This sets up a test we can use to check how good the system is at finding the
right page for a question. It does three things:

  1. picks some real questions to test with (we already know the correct page
     for each, so we can grade later);
  2. gathers a set of pages to search through -- the pages that hold the
     answers, plus lots of random other pages mixed in so the test is not too
     easy;
  3. goes through every page in that set and "studies" it once (turns it into
     the number form the system searches with). This is the slow part.

The slow "studying" is saved in CHUNKS as it goes, so if the run is stopped it
resumes where it left off instead of starting over. Re-running with a tag that
is already finished does nothing.

Usage (from the project root):
    .venv/Scripts/python rag_eval/build_subset.py --questions 200 --pages 2000 --tag main

Quick tiny run to check it works (~3 min):
    .venv/Scripts/python rag_eval/build_subset.py --questions 10 --pages 60 --tag smoke
"""
import argparse, json, math, os, random, shutil, time
import numpy as np
import common


def select_pages(args, out, reused_paras=None, excluded_questions=None):
    """Pick the questions and the pages to search, and save them. If a previous
    run already saved them for this tag, reuse them exactly (so the page set and
    its order never change between resumes)."""
    paras_path = os.path.join(out, "paras.json")
    questions_path = os.path.join(out, "questions.json")
    if os.path.exists(paras_path) and os.path.exists(questions_path):
        print("reusing existing page/question selection", flush=True)
        return json.load(open(paras_path, encoding="utf-8"))

    print("loading the pages and the questions ...", flush=True)
    corpus = common.load_corpus()
    q2docs = common.load_qa()

    # Only keep questions whose correct pages are in our corpus, then pick at
    # random with a fixed seed (so the same run always picks the same ones).
    corpus_pages = set(r["doc_id"] for r in corpus)
    excluded_questions = excluded_questions or set()
    usable = [q for q, docs in q2docs.items()
              if docs <= corpus_pages and q not in excluded_questions]
    if len(usable) < args.questions:
        raise ValueError(f"only {len(usable)} usable questions remain; requested {args.questions}")
    questions = random.sample(usable, min(args.questions, len(usable)))

    if reused_paras is not None:
        # The source cache has already embedded this exact full corpus.  Keep
        # its order: para_emb.npy rows correspond to it position-for-position.
        json.dump([{"question": q, "accepted": sorted(q2docs[q])} for q in questions],
                  open(questions_path, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(vars(args), open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=2)
        print(f"  questions: {len(questions)} | reusing {len(reused_paras)} embedded paragraphs", flush=True)
        return reused_paras

    answer_pages = set()
    for q in questions:
        answer_pages |= q2docs[q]

    # All answer pages, plus random other pages ("extras") up to the target size.
    answer_paras = [r for r in corpus if r["doc_id"] in answer_pages]
    other_paras = [r for r in corpus if r["doc_id"] not in answer_pages]
    n_extras = max(0, args.pages - len(answer_paras))
    extras = random.sample(other_paras, min(n_extras, len(other_paras)))
    search_pages = answer_paras + extras
    random.shuffle(search_pages)

    print(f"  questions: {len(questions)} | pages that hold answers: {len(answer_pages)}", flush=True)
    print(f"  pages to search: {len(search_pages)} "
          f"({len(answer_paras)} that hold answers + {len(extras)} random extras)", flush=True)

    json.dump([{"doc_id": r["doc_id"], "title": r["title"], "content": r["content"],
                "link": r["link"]} for r in search_pages],
              open(paras_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump([{"question": q, "accepted": sorted(q2docs[q])} for q in questions],
              open(questions_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)
    return search_pages


def prepare_reused_embeddings(args, out):
    """Copy a finished full-corpus cache without recomputing paragraph vectors.

    The cached vectors are valid only for its exact paragraph order.  Validate
    that source cache before copying so a partial or differently ordered cache
    cannot silently corrupt an evaluation.
    """
    if not args.reuse_paragraph_cache:
        return None
    if os.path.exists(os.path.join(out, "paras.json")) or os.path.exists(os.path.join(out, "para_emb.npy")):
        raise FileExistsError(f"destination cache '{args.tag}' already contains paragraph data")
    source = common.cache_dir(args.reuse_paragraph_cache)
    source_paras = os.path.join(source, "paras.json")
    source_emb = os.path.join(source, "para_emb.npy")
    if not os.path.isfile(source_paras) or not os.path.isfile(source_emb):
        raise FileNotFoundError(f"reuse cache '{args.reuse_paragraph_cache}' needs paras.json and para_emb.npy")
    paras = json.load(open(source_paras, encoding="utf-8"))
    emb = np.load(source_emb, mmap_mode="r")
    if emb.ndim != 2 or emb.shape[0] != len(paras):
        raise ValueError("reuse cache embeddings do not match its paragraph list")
    corpus = common.load_corpus()
    if len(paras) != len(corpus) or {p["doc_id"] for p in paras} != {p["doc_id"] for p in corpus}:
        raise ValueError("reuse cache does not contain the complete current corpus")
    shutil.copy2(source_paras, os.path.join(out, "paras.json"))
    shutil.copy2(source_emb, os.path.join(out, "para_emb.npy"))
    print(f"reused embeddings from cache '{args.reuse_paragraph_cache}'", flush=True)
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=200, help="how many questions to test with")
    ap.add_argument("--pages", type=int, default=2000, help="how many pages to search through in total")
    ap.add_argument("--seq", type=int, default=512, help="most text to read per page (leave at 512)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256, help="pages per saved chunk (resume granularity)")
    ap.add_argument("--tag", default="main", help="a name for this saved quiz (folder under rag_eval/cache)")
    ap.add_argument("--reuse-paragraph-cache", default=None,
                    help="finished full-corpus cache whose paragraph embeddings to copy")
    ap.add_argument("--exclude-questions-from", default=None,
                    help="cache tag whose selected questions must not be sampled again")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    out = common.cache_dir(args.tag)
    os.makedirs(out, exist_ok=True)

    excluded_questions = set()
    if args.exclude_questions_from:
        old_questions = os.path.join(common.cache_dir(args.exclude_questions_from), "questions.json")
        if not os.path.isfile(old_questions):
            raise FileNotFoundError(f"exclusion cache '{args.exclude_questions_from}' has no questions.json")
        excluded_questions = {q["question"] for q in json.load(open(old_questions, encoding="utf-8"))}
    reused_paras = prepare_reused_embeddings(args, out)
    search_pages = select_pages(args, out, reused_paras, excluded_questions)

    emb_path = os.path.join(out, "para_emb.npy")
    if os.path.exists(emb_path):
        print(f"embeddings already complete -> {emb_path}", flush=True)
        return

    texts = [r["content"] for r in search_pages]
    n = len(texts)
    n_chunks = math.ceil(n / args.chunk)
    chunk_dir = os.path.join(out, "emb_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    done = [os.path.exists(os.path.join(chunk_dir, f"chunk_{c:05d}.npy")) for c in range(n_chunks)]
    print(f"studying {n} pages in {n_chunks} chunks of {args.chunk} "
          f"({sum(done)} already done)", flush=True)

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    print("loading the Hebrew model ...", flush=True)
    model = SentenceTransformer(common.MODEL_DIR)
    model.max_seq_length = args.seq
    model.eval()

    t0 = time.perf_counter(); processed = 0
    for c in range(n_chunks):
        cpath = os.path.join(chunk_dir, f"chunk_{c:05d}.npy")
        if os.path.exists(cpath):
            continue
        block = texts[c * args.chunk:(c + 1) * args.chunk]
        emb = model.encode(block, batch_size=args.batch, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
        tmp = cpath + ".tmp.npy"
        np.save(tmp, emb.astype(np.float32)); os.replace(tmp, cpath)
        processed += 1
        el = (time.perf_counter() - t0) / 60
        left = (n_chunks - c - 1)
        eta = el / processed * left if processed else 0
        print(f"  chunk {c+1}/{n_chunks}  ({el:.1f} min this run, ~{eta:.0f} min left)", flush=True)

    # All chunks present -> stitch them into one file, in order.
    if all(os.path.exists(os.path.join(chunk_dir, f"chunk_{c:05d}.npy")) for c in range(n_chunks)):
        parts = [np.load(os.path.join(chunk_dir, f"chunk_{c:05d}.npy")) for c in range(n_chunks)]
        emb = np.concatenate(parts, axis=0)
        np.save(emb_path, emb.astype(np.float32))
        print(f"done -> {emb_path}  shape {emb.shape}", flush=True)
    else:
        print("some chunks still missing -- re-run to finish", flush=True)


if __name__ == "__main__":
    main()
