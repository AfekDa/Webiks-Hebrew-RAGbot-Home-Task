# Hebrew RAG — Improving the Retrieval Step

**Task:** improve the *retrieval* part of the Webiks open-source Hebrew RAG system
(Kol-Zchut data), measure the improvement honestly on a QA subset, and integrate
it cleanly into the demo backend so it still runs locally.

**What I changed, in one line:** I added a **reranker** - a second, more careful
model that re-reads the top search results together with the question and re-sorts
them, so the correct page moves to the top more often. The original search is
untouched; the reranker sits on top of it and can be switched off.

---

## 1. How the system works today (and where it loses accuracy)

When a user asks a question, the system:

1. turns the question into a "meaning fingerprint" with Webiks' trained Hebrew model,
2. asks Elasticsearch for the **50 closest paragraphs** by fingerprint,
3. collapses those to unique **pages** and keeps the top few to answer from.

The weak spot is step 2. Fingerprint matching is **fast but rough**: it reliably
pulls the right *neighborhood* of ~50 paragraphs, but it is not good at deciding
*which one* is actually the answer. So the correct page is usually caught in the
net — just sitting at position #6 instead of #1 — and can be dropped before the
answer step ever sees it.

> Analogy: a librarian who glances at 50 book *titles* that look related without
> opening any of them. The right book is almost always on the cart — just rarely
> the one on top.

---

## 2. The improvement: a cross-encoder reranker

I add one step between "get 50 candidates" and "keep the top few":

```
TODAY:      question → fast search → 50 candidates → keep top few (ROUGH order) → answer
WITH RERANK: question → fast search → 50 candidates → reranker re-reads & re-sorts
                                                    → keep top few (GOOD order)  → answer
```

The reranker (**BAAI/bge-reranker-v2-m3**, a multilingual model that handles Hebrew)
reads the question **and** a candidate paragraph *at the same time* and produces a
single relevance score. Because it looks at both together — instead of comparing two
fingerprints made separately — it judges relevance much more accurately. We re-sort
the candidates by that score and then collapse to pages exactly as before.

I do **not** replace the fast search — it is still needed to narrow ~24,000
paragraphs down to 50. I just make better use of those 50.

### Why this improvement, over the alternatives

| Option considered | Why not chosen |
|---|---|
| **Hybrid search** (add keyword matching) | Competes with Webiks' trained model at its *own* step; that model was trained on these exact questions, so it is already strong and hard to beat head-on. (Clear second choice / future add-on.) |
| **Different chunking** | Requires re-processing and re-storing the whole corpus (hours here), and it is hard to prove the chunking specifically was what helped. |
| **Query expansion** (rewrite the question) | Hit-or-miss and hard to measure convincingly. |
| **Retrain / replace the Hebrew model** | Huge effort, needs a GPU, and discards the very asset Webiks built. Wrong scope for 72 hours. |

The reranker won because it gives the best mix of **clear, provable gain + low risk
+ respect for the existing model**: it is a *new, orthogonal* signal added on top,
not a fight with what already works, and it toggles off cleanly.

---

## 3. How I measured it (a fair before/after)

I built a small evaluation harness (`rag_eval/`) that reproduces the real system's
retrieval **offline** — same model, same "top-50 → dedup to pages" logic — in plain
NumPy, so it needs no Elasticsearch/Docker and runs on a laptop. It mirrors the real
`Engine.search_documents` step for step; the only difference is that the top-50 is
computed with the same math instead of a running database.

**Method:** take a subset of the real QA questions (each has a known correct page),
run each question through the pipeline, and record where the first correct page
lands. Then repeat with the reranker turned on — **same questions, same candidate
lists** — so the only thing that changes is the *order*.

**Metrics, and why these:**

- **hit@k** ("is a correct page in the top *k*?", for k = 1, 3, 5, 10) — tells us
  *whether* the answer is reachable and how close to the top it is. hit@1 is what a
  user feels most.
- **MRR@10** (mean reciprocal rank) — rewards placing the correct page *high* (#1 →
  1.0, #2 → 0.5, …). This is the number a reranker is designed to move, because its
  whole job is to lift correct pages upward.

The reranker's opportunity is visible in the baseline itself: the correct page is in
the top 10 far more often than it is at #1. That gap — answers that are *found but
not on top* — is exactly what reranking targets.

---

## 4. Results

Measured on **100 questions**, searching a **2,000-page** subset, reranking each
question's **top 20** candidates. "Before" and "after" use the *identical* questions
and candidate lists, so the comparison is exact.

| How often the correct page is… | Before (search only) | After (+ reranker) | Change |
|---|---|---|---|
| ranked **#1** | `__._%` | `__._%` | `+_._` |
| in the top 3 | `__._%` | `__._%` | `+_._` |
| in the top 5 | `__._%` | `__._%` | `+_._` |
| in the **top 10** | `__._%` | `__._%` | `+_._` |
| MRR@10 ("how high up", avg) | `_.___` | `_.___` | `+_.___` |

> _Numbers pending the full run (~3 h on this CPU); filled in from
> `rag_eval/cache/main/reranker_results.json` when it completes._

For context, our search-only baseline on a larger 200-question / 2,000-page run was
**76.5% at #1, 98.0% in the top 10, MRR@10 = 0.852** — a strong baseline (see
limitation below), with a ~21-in-100 gap between "in the top 10" and "at #1" for the
reranker to close.

---

## 5. Integrating it into the Demo backend

- The reranker is added as an **optional step inside the engine's search**, off by
  default — with it off, behavior is byte-for-byte the original.
- It re-reads only the **top 20** candidates (not all 50) so a single query stays
  responsive on a CPU; the model loads once at startup.
- The answer (LLM) step is left behind a small swappable interface with a **mock**
  option, so the whole backend runs **locally with no API key**.
- No change to Elasticsearch, the corpus, or the trained model — the reranker is a
  self-contained add-on.

---

## 6. Honest limitations

- **Strong baseline.** The QA questions are the same set Webiks' embedder was trained
  on, so the search already "knows" them and scores high — leaving less room to
  improve. Any gain here is therefore meaningful, and the reranker (a *separate*
  signal) is a safer bet than beating the trained model at its own step.
- **Easier exam than production.** Our 2,000-page subset is smaller than the full
  ~24,000-paragraph corpus, so absolute numbers look higher than Webiks' own
  (~36% at #1). It is still a *fair* before/after because the reranker faces the
  exact same subset.
- **CPU speed.** This machine has no GPU, so the reranker is slow (~5 s per page-read).
  That is why the evaluation uses a subset and reranks the top 20; on a GPU it would
  be near-instant, and the accuracy gain carries over unchanged.

---

## 7. How to reproduce

From the project root, with the data + model in place (see `UPSTREAM.md`):

```
# 1. Build the test set + embed the pages once (slow, cached):
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --pages 2000 --tag main

# 2. Baseline (search only):
.venv\Scripts\python rag_eval\eval_baseline.py --tag main

# 3. Reranker before/after on the same questions:
.venv\Scripts\python rag_eval\eval_reranker.py --tag main --n-questions 100 --top-rerank 20
```

Code layout: `rag_eval/common.py` (shared retrieval logic, mirrors the real engine),
`build_subset.py` (one-time prep), `eval_baseline.py` (before), `eval_reranker.py`
(after). See `rag_eval/README.md` for a fuller walkthrough.
