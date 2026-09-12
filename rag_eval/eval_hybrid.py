"""The pivot: hybrid retrieval = dense meaning search + BM25 keyword search.

Why this, after the reranker did not hold up: the shipped Hebrew model matches
*meaning* well, but on a legal/benefits corpus the answer often hinges on an
*exact* word -- a form number, a law name, a specific benefit -- that a meaning
model can smear together. BM25 is the opposite: it rewards exact word overlap.
Fusing the two (reciprocal rank fusion) lets each cover the other's blind spot,
and -- unlike a cross-encoder -- it does not try to overrule the retriever that
was trained on these very questions; it *adds* a second, independent retriever.

This measures it with the SAME honesty as the reranker sweep:
  - pick the fusion strength on the first half of the questions (dev),
  - report it once on the untouched second half (held-out),
  - ship only if held-out Hit@1 or MRR improves and Hit@5 does not regress.

"Baseline" is today's dense-only system, computed exactly like eval_baseline.py
(top-50 paragraphs -> dedup to pages), so the before-number matches.

Runs wherever a built cache lives (needs the paragraph embeddings + the model):
    .venv/Scripts/python rag_eval/eval_hybrid.py --tag full --dev 100
"""
import argparse, json, math, os, re, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Webiks-Hebrew-RAGbot"))
from webiks_hebrew_ragbot.rank_fusion import reciprocal_rank_fusion
import common

KS = (1, 3, 5, 10)
_WORD = re.compile(r"\w+", re.UNICODE)      # Hebrew letters and digits are word chars


def tokenize(text):
    return _WORD.findall(text.lower())


class BM25:
    """Compact BM25 (Okabi), Elasticsearch-like defaults (k1=1.2, b=0.75).

    This is a stand-in for Elasticsearch's own BM25 -- close, not identical
    (ES analysis differs), which is fine: the offline harness is a proxy, and
    the engine integration uses ES's real BM25.
    """
    def __init__(self, docs_tokens, k1=1.2, b=0.75):
        self.N, self.k1, self.b = len(docs_tokens), k1, b
        self.dl = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        self.tf, self.postings, df = [], defaultdict(list), defaultdict(int)
        for i, toks in enumerate(docs_tokens):
            counts = Counter(toks)
            self.tf.append(counts)
            for t in counts:
                df[t] += 1
                self.postings[t].append(i)
        self.idf = {t: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_tokens):
        s = np.zeros(self.N)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s[i] += idf * (f * (self.k1 + 1)) / denom
        return s


def dedup_to_pages(para_idx_order, para_doc_ids):
    pages, seen = [], set()
    for i in para_idx_order:
        did = para_doc_ids[i]
        if did not in seen:
            seen.add(did); pages.append(did)
    return pages


def metrics_over(rows_pages, rows_accepted):
    hit = {k: 0 for k in KS}; mrr = 0.0
    for pages, accepted in zip(rows_pages, rows_accepted):
        hits, rr, _ = common.score_ranking(pages, accepted, KS)
        for k in KS:
            hit[k] += hits[k]
        mrr += rr
    n = len(rows_pages)
    return {"hit@1": hit[1]/n, "hit@3": hit[3]/n, "hit@5": hit[5]/n,
            "hit@10": hit[10]/n, "mrr@10": mrr/n}


def fmt(m):
    return (f"{m['hit@1']*100:6.1f}{m['hit@3']*100:7.1f}{m['hit@5']*100:7.1f}"
            f"{m['hit@10']*100:7.1f}{m['mrr@10']:8.3f}")


def header(title):
    print(f"\n{title}")
    print(f"  {'setting':<22}{'#1':>6}{'top3':>7}{'top5':>7}{'top10':>7}{'MRR':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dev", type=int, default=100, help="first N questions = dev; rest = held-out")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 40, 60, 80], help="RRF strengths to try")
    ap.add_argument("--depth", type=int, default=100, help="top results taken from each retriever before fusing")
    ap.add_argument("--weights", type=float, nargs=2, default=None, metavar=("DENSE", "BM25"),
                    help="optional fixed dense:bm25 weighting; default sweeps a few")
    ap.add_argument("--name", default=None, help="results folder to write under rag_eval/results/")
    args = ap.parse_args()

    out = common.cache_dir(args.tag)
    emb = np.load(os.path.join(out, "para_emb.npy"))
    paras = json.load(open(os.path.join(out, "paras.json"), encoding="utf-8"))
    questions = json.load(open(os.path.join(out, "questions.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    para_doc_ids = [p["doc_id"] for p in paras]
    if len(questions) <= args.dev:
        raise SystemExit(f"need more than --dev={args.dev} questions; cache has {len(questions)}")
    print(f"cache '{args.tag}': {len(paras)} paragraphs, {len(questions)} questions "
          f"(dev = first {args.dev}, held-out = last {len(questions)-args.dev})", flush=True)

    print("building BM25 keyword index ...", flush=True)
    bm25 = BM25([tokenize(p["content"]) for p in paras])

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    model = SentenceTransformer(common.MODEL_DIR)
    model.max_seq_length = cfg.get("seq", 512)
    model.eval()
    print("encoding questions ...", flush=True)
    q_emb = model.encode([q["question"] for q in questions], batch_size=16,
                         show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)

    # Precompute each question's dense ranking, bm25 ranking, and the dense-only
    # baseline pages -- once. The sweeps below only re-fuse, which is instant.
    accepted = [set(q["accepted"]) for q in questions]
    dense_rank, bm25_rank, base_pages, bm25_pages = [], [], [], []
    for qi in range(len(questions)):
        sims = q_emb[qi] @ emb.T
        d_order = np.argsort(-sims)
        dense_rank.append(d_order[:args.depth].tolist())
        base_pages.append(dedup_to_pages(d_order[:common.ES_SIZE].tolist(), para_doc_ids))
        b_order = np.argsort(-bm25.scores(tokenize(questions[qi]["question"])))
        bm25_rank.append(b_order[:args.depth].tolist())
        bm25_pages.append(dedup_to_pages(b_order[:common.ES_SIZE].tolist(), para_doc_ids))

    dev = slice(0, args.dev); test = slice(args.dev, len(questions))

    def fused_pages(qi, k, weights):
        order = reciprocal_rank_fusion([dense_rank[qi], bm25_rank[qi]], k, weights)
        return dedup_to_pages(order, para_doc_ids)

    base_dev = metrics_over(base_pages[dev], accepted[dev])
    base_test = metrics_over(base_pages[test], accepted[test])

    # Context: how each retriever does alone on dev.
    header("[dev] each retriever alone")
    print(f"  {'dense (baseline)':<22}{fmt(base_dev)}")
    print(f"  {'bm25 only':<22}{fmt(metrics_over(bm25_pages[dev], accepted[dev]))}")

    # Sweep: RRF strength, and a few dense:bm25 weightings.
    weight_options = [tuple(args.weights)] if args.weights else [(1, 1), (2, 1), (1, 2)]
    header("[dev] hybrid = dense + bm25 (RRF), picking here")
    print(f"  {'dense (baseline)':<22}{fmt(base_dev)}")
    configs = []
    for w in weight_options:
        for k in args.ks:
            pages = [fused_pages(qi, k, list(w)) for qi in range(args.dev)]
            m = metrics_over(pages, accepted[dev])
            print(f"  {f'RRF k={k} w={w[0]}:{w[1]}':<22}{fmt(m)}")
            configs.append((k, w, m))

    # Pick on dev: no Hit@5 regression, improve Hit@1 or MRR. Best by (MRR, Hit@1).
    tol = 1e-9
    qualified = [c for c in configs
                 if c[2]["hit@5"] >= base_dev["hit@5"] - tol
                 and (c[2]["hit@1"] > base_dev["hit@1"] + tol
                      or c[2]["mrr@10"] > base_dev["mrr@10"] + tol)]
    print("\n" + "=" * 64)
    if not qualified:
        print("DEV VERDICT: no hybrid setting beats dense baseline (Hit@1 or MRR up, Hit@5 not down).")
        print("Hybrid does not help on this corpus either -- report as a second tested negative.")
        _save(args, out, base_test, None, None, None)
        return
    k, w, m_dev = max(qualified, key=lambda c: (c[2]["mrr@10"], c[2]["hit@1"]))
    label = f"RRF k={k} w={w[0]}:{w[1]}"
    test_pages = [fused_pages(qi, k, list(w)) for qi in range(args.dev, len(questions))]
    m_test = metrics_over(test_pages, accepted[test])
    print(f"DEV PICK: {label}")
    header("Held-out confirmation (last questions, never used to pick)")
    print(f"  {'dense (baseline)':<22}{fmt(base_test)}")
    print(f"  {label:<22}{fmt(m_test)}")
    d1 = (m_test["hit@1"] - base_test["hit@1"]) * 100
    d5 = (m_test["hit@5"] - base_test["hit@5"]) * 100
    dm = m_test["mrr@10"] - base_test["mrr@10"]
    print(f"\n  held-out change: Hit@1 {d1:+.1f}, Hit@5 {d5:+.1f}, MRR {dm:+.3f}")
    ship = (d5 >= -tol) and (d1 > tol or dm > tol)
    print(f"  SHIP: {'yes' if ship else 'no'} (rule: Hit@5 not down AND Hit@1 or MRR up, on held-out)")
    _save(args, out, base_test, label, m_test, {"k": k, "weights": list(w), "depth": args.depth, "ship": ship})


def _save(args, out, base_test, label, m_test, choice):
    result = {"tag": args.tag, "method": "dense+bm25 RRF", "dev": args.dev,
              "held_out_baseline": base_test, "held_out_hybrid": m_test,
              "chosen": choice, "label": label}
    json.dump(result, open(os.path.join(out, "hybrid_results.json"), "w"), indent=2)
    if args.name:
        dst = os.path.join("rag_eval/results", args.name)
        os.makedirs(dst, exist_ok=True)
        json.dump(result, open(os.path.join(dst, "hybrid_results.json"), "w"), indent=2)
    print(f"\nsaved -> {out}/hybrid_results.json")


if __name__ == "__main__":
    main()
