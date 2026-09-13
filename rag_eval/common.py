"""Shared helpers for the retrieval evaluation.

Faithfully mirrors how the real system retrieves -- specifically the function
Engine.search_documents in Webiks-Hebrew-RAGbot/webiks_hebrew_ragbot/engine.py
(with the top-50 search coming from elastic_model.py):

  1. encode the text with the SAME SentenceTransformer model (it normalizes
     its output, so cosine similarity == dot product);
  2. score every paragraph by cosine similarity to the query;
  3. take the top ES_SIZE (=50) paragraphs, exactly like Elasticsearch's
     `size=50` search;
  4. collapse those paragraphs to unique pages (doc_id), first-seen order,
     exactly like Engine.search_documents.

Doing it offline in numpy avoids needing Elasticsearch/Docker and lets us
reuse one cached corpus embedding across the baseline and the page-scoring eval.
"""
import csv, json, os
from collections import defaultdict

# ---- paths (relative to project root) ----
CORPUS_PATH = "Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json"
QA_PATH = "Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv"
MODEL_DIR = "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"
CACHE_ROOT = "rag_eval/cache"

# ES returns the top-50 paragraphs by cosine (elastic_model.search default size)
ES_SIZE = 50


def load_corpus():
    """Return the corpus as a list of {doc_id(str), title, content, link}."""
    d = json.load(open(CORPUS_PATH, encoding="utf-8"))
    keys = list(d["doc_id"].keys())
    recs = []
    for k in keys:
        recs.append({
            "doc_id": str(d["doc_id"][k]),
            "title": d["title"][k],
            "content": d["content"][k],
            "link": d["link"][k],
        })
    return recs


def load_qa():
    """Return question -> set of accepted doc_ids (strings).

    A question can have several correct pages, so the value is a set.
    """
    q2docs = defaultdict(set)
    with open(QA_PATH, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row["question"].strip()
            q2docs[q].add(str(row["doc_id"]).strip())
    return dict(q2docs)


def rank_pages(sims, para_doc_ids, es_size=ES_SIZE):
    """Given similarity scores for every paragraph, reproduce the system's
    page ranking: take top `es_size` paragraphs, then dedup to unique pages
    in first-seen order.

    Returns two things:
      - pages: the ranked list of page ids (doc_ids), best first;
      - order: the top `es_size` paragraph indices (best first, before dedup).
        Later steps reuse `order` to re-score the SAME paragraphs.
    """
    import numpy as np
    order = np.argsort(-sims)[:es_size]        # top-50 paragraph indices
    pages, seen = [], set()
    for i in order:
        did = para_doc_ids[i]
        if did not in seen:
            seen.add(did)
            pages.append(did)
    return pages, order


def score_ranking(ranked_pages, accepted, ks=(1, 3, 5, 10)):
    """Score one question's page ranking against its accepted pages.

    hit@k = at least one accepted page appears in the top k pages.

    Returns three things:
      - hits: {k: True/False} for each k in `ks`;
      - rr: reciprocal rank (1/position of the first correct page, 0 if none
        in the top 10) -- averaging this over all questions gives MRR@10;
      - rank: the position (1-based) of the first correct page, or None.
    """
    rank = None
    for pos, did in enumerate(ranked_pages, start=1):
        if did in accepted:
            rank = pos
            break
    hits = {k: (rank is not None and rank <= k) for k in ks}
    rr = (1.0 / rank) if (rank is not None and rank <= 10) else 0.0
    return hits, rr, rank


def cache_dir(tag):
    return os.path.join(CACHE_ROOT, tag)
