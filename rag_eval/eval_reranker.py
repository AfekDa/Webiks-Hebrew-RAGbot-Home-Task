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

The reranker's opinion is used in two ways, and BOTH are measured in one run:
  replace : the reranker's order alone (the classic setup).
  blend   : the reranker's order merged with the search order, so both
            opinions count (see rank_fusion.py in the engine package). This is
            what the live engine does by default, because on the full corpus
            "replace" hurt while "blend" helped a little.
--mode picks which one is the headline "after" number; the other is still
recorded and printed as a third column.

Two knobs keep it fast on a CPU (BGE is heavy):
  --top-rerank N : only re-read the top N candidates; the rest keep their old
                   order behind them (so the page set is identical to baseline).
  --n-questions M: only test the first M questions from the cache.

Run eval_baseline.py first with the same --tag (it saves candidates.json).

Usage:
    .venv/Scripts/python rag_eval/eval_reranker.py --tag main --n-questions 100 --top-rerank 20
"""
import argparse, hashlib, json, os, sys, time

# The blend function lives in the engine package so the live system and this
# evaluation share one implementation. Import it straight from the source tree.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Webiks-Hebrew-RAGbot"))
from webiks_hebrew_ragbot.rank_fusion import fuse_orders

# If the BGE model is already downloaded on this machine, load it straight from
# disk and run offline (so a flaky network can't interfere). On a fresh machine
# the folder won't exist, so we fall back to the Hugging Face id and let it
# download normally the first time.
_LOCAL_MODEL = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "models--BAAI--bge-reranker-v2-m3",
    "snapshots", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
)
if os.path.isdir(_LOCAL_MODEL):
    DEFAULT_MODEL = _LOCAL_MODEL
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"   # downloads on first use

import common


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
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--mode", choices=["blend", "replace"], default="blend",
                    help="headline 'after' number: blend with search order, or replace it")
    ap.add_argument("--blend-k", type=int, default=5, dest="blend_k",
                    help="blend strength; small = top of each list matters most")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU threads (default = physical cores, avoids thrashing)")
    args = ap.parse_args()
    if min(args.top_rerank, args.n_questions, args.seq, args.batch, args.threads, args.blend_k) < 1:
        ap.error("reranking limits, batch size, threads and blend-k must be positive")
    out = common.cache_dir(args.tag)

    paras = json.load(open(os.path.join(out, "paras.json"), encoding="utf-8"))
    candidates = json.load(open(os.path.join(out, "candidates.json"), encoding="utf-8"))
    candidates = candidates[:args.n_questions]
    if not candidates:
        raise ValueError("No candidates to evaluate; run eval_baseline.py first")
    para_doc_ids = [p["doc_id"] for p in paras]
    print(f"cache '{args.tag}': testing {len(candidates)} questions, "
          f"reranking top {args.top_rerank} of each, {len(paras)} paragraphs", flush=True)

    # Refuse to mix checkpoints from different data or model settings.
    manifest = {
        "model": args.model, "top_rerank": args.top_rerank, "seq": args.seq, "dtype": args.dtype,
        "score_transform": "raw_logits",
        "tie_break": "doc_id_content",
        "blend_k": args.blend_k,
        "n_questions": len(candidates),
        "data_sha256": hashlib.sha256(json.dumps(
            [paras, candidates], ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest(),
    }
    manifest_path = os.path.join(out, "rerank_config.json")
    prog_path = os.path.join(out, "rerank_progress.jsonl")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            if json.load(f) != manifest:
                raise ValueError("Reranker cache settings changed; use a fresh cache tag")
    elif os.path.exists(prog_path):
        raise ValueError("Reranker checkpoint has no configuration; use a fresh cache tag")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    import torch
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "float16" and device != "cuda":
        raise RuntimeError("float16 reranking requires CUDA; use --dtype float32 on CPU")
    from sentence_transformers import CrossEncoder
    print(f"loading reranker (threads={args.threads}) ...", flush=True)
    t_load = time.perf_counter()
    reranker = CrossEncoder(args.model, max_length=args.seq, device=device)
    reranker.model.to(device=device, dtype=getattr(torch, args.dtype))
    print(f"  loaded in {time.perf_counter()-t_load:.0f}s", flush=True)
    runtime = {
        "torch": torch.__version__, "device": str(reranker.model.device),
        "dtype": str(next(reranker.model.parameters()).dtype),
    }

    ks = (1, 3, 5, 10)
    # Checkpoint file: one line per finished question. We append+flush after each
    # question, so if the run is killed we lose at most one question and can
    # resume. Re-running skips questions already recorded here.
    done = {}
    if os.path.exists(prog_path):
        for line in open(prog_path, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["qi"]] = rec
        print(f"resuming: {len(done)} questions already done", flush=True)

    def write_results():
        recs = list(done.values())
        n = len(recs)
        if n == 0:
            return None, None
        def avg(side, idx): return sum(r[side][idx] for r in recs) / n
        def summary(side):
            return {"hit@1": avg(side, 0), "hit@3": avg(side, 1), "hit@5": avg(side, 2),
                    "hit@10": avg(side, 3), "mrr@10": avg(side, 4)}
        baseline = summary("b")
        result = {"tag": args.tag, "model": args.model, "n_questions": n,
                  "top_rerank": args.top_rerank, "n_paragraphs": len(paras),
                  "mode": args.mode, "blend_k": args.blend_k,
                  **summary("r"),
                  "baseline_same_subset": baseline,
                  "replace_mode": summary("replace"),
                  "blend_mode": summary("blend"),
                  "runtime": runtime,
                  "mean_rerank_seconds": sum(r["rerank_seconds"] for r in recs) / n}
        json.dump(result, open(os.path.join(out, "reranker_results.json"), "w"), indent=2)
        return result, baseline

    prog_f = open(prog_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    processed = 0
    for qi, c in enumerate(candidates):
        if qi in done:
            continue
        cand = c["cand_para_idx"]                       # top-50 in the baseline's order
        accepted = set(c["accepted"])

        # --- baseline: original order, dedup to pages ---
        b_pages = dedup_to_pages(cand, para_doc_ids)
        b_hits, b_rr, _ = common.score_ranking(b_pages, accepted, ks)

        # --- reranker: re-read the top N, re-sort them, keep the rest behind ---
        head = cand[:args.top_rerank]
        tail = cand[args.top_rerank:]
        pairs = [[c["question"], paras[i]["content"]] for i in head]
        t_query = time.perf_counter()
        scores = reranker.predict(pairs, batch_size=args.batch, show_progress_bar=False,
                                  activation_fct=torch.nn.Identity())
        rerank_seconds = time.perf_counter() - t_query
        order = sorted(range(len(head)), key=lambda j: (
            -float(scores[j]), str(paras[head[j]]["doc_id"]), paras[head[j]]["content"],
        ))
        # "replace": the reranker's order alone. "blend": merged with the search
        # order, exactly as the live engine does it (same fuse_orders function).
        orders = {"replace": order,
                  "blend": fuse_orders(list(range(len(head))), order, args.blend_k)}
        pages, metrics = {}, {}
        for name, o in orders.items():
            pages[name] = dedup_to_pages([head[j] for j in o] + tail, para_doc_ids)
            h, rr, _ = common.score_ranking(pages[name], accepted, ks)
            metrics[name] = [int(h[1]), int(h[3]), int(h[5]), int(h[10]), rr]

        rec = {"qi": qi,
               "b": [int(b_hits[1]), int(b_hits[3]), int(b_hits[5]), int(b_hits[10]), b_rr],
               "r": metrics[args.mode],                 # the headline "after" (chosen --mode)
               "replace": metrics["replace"], "blend": metrics["blend"],
               "baseline_pages": b_pages, "reranked_pages": pages[args.mode],
               "replace_pages": pages["replace"], "blend_pages": pages["blend"],
               "reranker_scores": [float(s) for s in scores],
               "rerank_seconds": rerank_seconds}
        done[qi] = rec
        prog_f.write(json.dumps(rec) + "\n"); prog_f.flush()
        processed += 1

        if processed % 5 == 0:
            el = (time.perf_counter() - t0) / 60
            todo = len(candidates) - len(done)
            eta = el / processed * todo
            print(f"  {len(done)}/{len(candidates)}  ({el:.1f} min this run, ~{eta:.0f} min left)",
                  flush=True)
            write_results()   # keep an up-to-date partial result on disk
    prog_f.close()

    result, baseline = write_results()
    if result is None or baseline is None:
        print("no questions processed -- nothing to report", flush=True)
        return
    assert result is not None and baseline is not None
    n = result["n_questions"]

    def pct(x): return f"{x*100:5.1f}%"
    rows = [("correct page at #1", "hit@1"), ("correct page in top3", "hit@3"),
            ("correct page in top5", "hit@5"), ("correct page in top10", "hit@10")]
    replace, blend = result["replace_mode"], result["blend_mode"]
    print(f"\n===== BEFORE vs AFTER  (same {n} questions, top-{args.top_rerank} reranked, "
          f"headline mode = {args.mode}) =====")
    print(f"  minutes to run : {(time.perf_counter()-t0)/60:.0f}")
    print(f"\n  {'metric':<22}{'before':>9}{'replace':>9}{'blend':>9}{'change':>9}   (change = {args.mode} - before)")
    for label, key in rows:
        b, a = baseline[key], result[key]
        print(f"  {label:<22}{pct(b):>9}{pct(replace[key]):>9}{pct(blend[key]):>9}{(a-b)*100:>+8.1f}")
    b, a = baseline["mrr@10"], result["mrr@10"]
    print(f"  {'MRR@10':<22}{b:>9.3f}{replace['mrr@10']:>9.3f}{blend['mrr@10']:>9.3f}{a-b:>+9.3f}")
    print(f"\nsaved -> {out}/reranker_results.json")


if __name__ == "__main__":
    main()
