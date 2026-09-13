"""Page scoring: use MORE of what the trained model already tells us.

A generic add-on tried earlier (a cross-encoder reranker) lost on the held-out
test, because it tries to overrule a retriever that was trained on this very
data. This improvement goes the other way: it keeps the trained model as the
only judge, but listens to it more carefully. (The reranker's numbers are in
`SUBMISSION.md`; its code was removed.)

Today a page's rank is just the rank of its single best paragraph. That ignores
two signals the same model gives us for free:

  title   : how well the question matches the PAGE TITLE (Kol Zchut titles are
            the topic, e.g. "child allowance"). We embed titles with the same
            model and add that similarity to the page score.
  second  : how well the page's SECOND-best paragraph matches. The correct page
            usually has several relevant paragraphs; a distractor has one.

Both only re-order the same top-50 candidates the system already retrieves --
no new model, no re-indexing -- so recall is unchanged and the comparison with
the baseline is exact (same candidates, different order).

Measured with the same discipline as before: settings are picked on the first
half of the questions (dev) and confirmed once on the untouched second half.

    .venv/Scripts/python rag_eval/eval_page_scoring.py --tag full --dev 100
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

# The scoring rule lives in the engine package, so the live system and this
# evaluation share one implementation. Import it straight from the source tree.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Webiks-Hebrew-RAGbot"))
from webiks_hebrew_ragbot.page_order import order_pages
import common

KS = (1, 3, 5, 10)


def page_scores(order, sims, para_doc_ids, title_sim, alpha, lam, margin=None):
    """Page order for one question, from its top-50 candidate paragraphs.

    score = best paragraph + alpha * second-best paragraph + lam * title match
    `margin`: only pages near the top compete (see page_scoring.order_pages)."""
    per_page, first_pos = defaultdict(list), {}
    for pos, i in enumerate(order):
        did = para_doc_ids[i]
        per_page[did].append(float(sims[i]))
        first_pos.setdefault(did, pos)
    return order_pages(per_page, title_sim, first_pos, alpha, lam, margin)


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
    print(f"  {'setting':<32}{'#1':>6}{'top3':>7}{'top5':>7}{'top10':>7}{'MRR':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dev", type=int, default=100, help="first N questions = dev; rest = held-out")
    # Gentle weights only: on a 2,000-page pilot, title weights of 1-2 let the
    # title overrule the paragraph and hurt badly; 0.1-0.5 helped or was neutral.
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5],
                    help="weight of the second-best paragraph")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5],
                    help="weight of the title match")
    ap.add_argument("--margins", type=float, nargs="+", default=[-1, 0.05],
                    help="margin gates to try; -1 = no gate")
    ap.add_argument("--name", default=None, help="results folder to write under rag_eval/results/")
    args = ap.parse_args()
    margins = [None if m < 0 else m for m in args.margins]

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

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    model = SentenceTransformer(common.MODEL_DIR)
    model.max_seq_length = cfg.get("seq", 512)
    model.eval()

    # One title per page, embedded once with the same model.
    title_of = {}
    for p in paras:
        title_of.setdefault(p["doc_id"], p["title"] or "")
    page_ids = list(title_of)
    print(f"encoding {len(page_ids)} page titles + {len(questions)} questions ...", flush=True)
    t_emb = np.asarray(model.encode([title_of[d] for d in page_ids], batch_size=32,
                                    show_progress_bar=True, normalize_embeddings=True,
                                    convert_to_numpy=True))
    q_emb = np.asarray(model.encode([q["question"] for q in questions], batch_size=16,
                                    show_progress_bar=True, normalize_embeddings=True,
                                    convert_to_numpy=True))
    title_sims_all = q_emb @ t_emb.T                      # [questions x pages]

    accepted = [set(q["accepted"]) for q in questions]
    orders, sims_list, title_sim_list = [], [], []
    for qi in range(len(questions)):
        sims = q_emb[qi] @ emb.T
        order = np.argsort(-sims)[:common.ES_SIZE].tolist()   # same top-50 as the system
        orders.append(order); sims_list.append(sims)
        title_sim_list.append({d: float(title_sims_all[qi, j]) for j, d in enumerate(page_ids)})

    def pages(qi, alpha, lam, margin=None):
        return page_scores(orders[qi], sims_list[qi], para_doc_ids, title_sim_list[qi],
                           alpha, lam, margin)

    def label_of(alpha, lam, margin):
        gate = "" if margin is None else f", gate {margin:g}"
        return f"second x{alpha:g} + title x{lam:g}{gate}"

    dev = range(0, args.dev); test = range(args.dev, len(questions))
    acc_dev = [accepted[i] for i in dev]; acc_test = [accepted[i] for i in test]
    base_dev = metrics_over([pages(i, 0, 0) for i in dev], acc_dev)
    base_test = metrics_over([pages(i, 0, 0) for i in test], acc_test)

    header("[dev] page scoring variants (same top-50 candidates, re-ordered)")
    print(f"  {'baseline (best para)':<32}{fmt(base_dev)}")
    configs = []
    for margin in margins:
        for alpha in args.alphas:
            for lam in args.lams:
                if alpha == 0 and lam == 0:
                    continue
                m = metrics_over([pages(i, alpha, lam, margin) for i in dev], acc_dev)
                print(f"  {label_of(alpha, lam, margin):<32}{fmt(m)}")
                configs.append((alpha, lam, margin, m))

    tol = 1e-9
    qualified = [c for c in configs
                 if c[3]["hit@5"] >= base_dev["hit@5"] - tol
                 and (c[3]["hit@1"] > base_dev["hit@1"] + tol
                      or c[3]["mrr@10"] > base_dev["mrr@10"] + tol)]
    print("\n" + "=" * 64)
    if not qualified:
        print("DEV VERDICT: no variant beats baseline (Hit@1 or MRR up, Hit@5 not down).")
        _save(args, out, base_test, None, None, None)
        return
    alpha, lam, margin, _ = max(qualified, key=lambda c: (c[3]["mrr@10"], c[3]["hit@1"]))
    label = label_of(alpha, lam, margin)
    m_test = metrics_over([pages(i, alpha, lam, margin) for i in test], acc_test)
    print(f"DEV PICK: {label}")
    header("Held-out confirmation (last questions, never used to pick)")
    print(f"  {'baseline (best para)':<32}{fmt(base_test)}")
    print(f"  {label:<32}{fmt(m_test)}")
    d1 = (m_test["hit@1"] - base_test["hit@1"]) * 100
    d5 = (m_test["hit@5"] - base_test["hit@5"]) * 100
    dm = m_test["mrr@10"] - base_test["mrr@10"]
    print(f"\n  held-out change: Hit@1 {d1:+.1f}, Hit@5 {d5:+.1f}, MRR {dm:+.3f}")
    ship = (d5 >= -tol) and (d1 > tol or dm > tol)
    print(f"  SHIP: {'yes' if ship else 'no'} (rule: Hit@5 not down AND Hit@1 or MRR up, on held-out)")
    # Per-question held-out rankings (top 10), so the live API can be checked
    # against the offline result (scripts/verify_demo.py).
    held = [{"qi": i, "question": questions[i]["question"], "accepted": sorted(accepted[i]),
             "baseline_pages": pages(i, 0, 0)[:10],
             "improved_pages": pages(i, alpha, lam, margin)[:10]} for i in test]
    _save(args, out, base_test, label, m_test,
          {"alpha": alpha, "lam": lam, "margin": margin, "ship": ship}, held)


def _save(args, out, base_test, label, m_test, choice, held=None):
    result = {"tag": args.tag, "method": "page scoring: best + second paragraph + title",
              "dev": args.dev, "held_out_baseline": base_test, "held_out_improved": m_test,
              "chosen": choice, "label": label, "held_out_questions": held}
    json.dump(result, open(os.path.join(out, "page_scoring_results.json"), "w"), indent=2)
    if args.name:
        dst = os.path.join("rag_eval/results", args.name)
        os.makedirs(dst, exist_ok=True)
        json.dump(result, open(os.path.join(dst, "page_scoring_results.json"), "w"), indent=2)
    print(f"\nsaved -> {out}/page_scoring_results.json")


if __name__ == "__main__":
    main()
