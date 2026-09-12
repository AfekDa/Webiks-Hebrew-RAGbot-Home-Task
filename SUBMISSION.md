# Hebrew RAG — Improving the Retrieval Step

**Task:** improve the *retrieval* part of the Webiks open-source Hebrew RAG system
(Kol-Zchut data), measure the improvement honestly on a QA subset, and integrate
it cleanly into the demo backend so it still runs locally.

**What I changed, in one line:** I added a **reranker** - a second, more careful
model that re-reads the top search results together with the question and gives
its own opinion on the order. The original search is untouched; the reranker sits
on top of it and can be switched off.

**What I found, in one line:** measured on the *full* corpus, letting the reranker
**replace** the search order made results worse; letting it **blend** with the
search order (both opinions count) recovered the loss and gave a small gain in
the top 5. The honest headline is a modest gain plus a clear explanation of why a
strong general reranker cannot beat a search model that was trained on these very
questions.

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

**Two ways to use the reranker's opinion** (a setting, `RERANK_MODE`):

- **replace** - use the reranker's order alone. The classic setup.
- **blend** - each candidate gets points from its position in *both* lists (the
  search order and the reranker order), and the points are added up. A page both
  models like lands on top; a page only one of them likes is pulled up only
  part-way. This is called reciprocal rank fusion; the implementation is one
  small dependency-free file (`rank_fusion.py`) shared by the engine and the
  evaluation. **This is the default**, because it measured better (section 4).

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

Measured on **200 questions** against the **full corpus** (all 24,487 paragraphs,
7,007 pages), reranking each question's **top 50** candidates on a GPU. "Before"
and both "after" columns use the *identical* questions and candidate lists, so the
only thing that differs is the order.

| How often the correct page is… | Before (search only) | Reranker **replaces** order | Reranker **blended** with order |
|---|---|---|---|
| ranked **#1** | 44.0% | 41.0% (-3.0) | 42.5% (-1.5) |
| in the top 3 | 70.5% | 61.5% (-9.0) | 70.5% (0.0) |
| in the top 5 | 78.5% | 74.5% (-4.0) | **81.5% (+3.0)** |
| in the **top 10** | 89.5% | 87.0% (-2.5) | 89.5% (0.0) |
| MRR@10 ("how high up", avg) | 0.588 | 0.546 (-0.042) | 0.584 (-0.004) |

**Reading the table honestly:**

- **Replacing the search order hurts, and the top-3 drop is real.** A paired
  bootstrap on the 200 questions puts the top-3 change between -16.5 and -1.5
  points, so it is not noise. Per question: the reranker fixed 24 #1 answers and
  broke 30.
- **Blending recovers the loss and adds a little.** Top 5 goes up 3 points; #1
  and MRR are within a hair of baseline. The top-5 gain's range is -2.0 to +8.0
  points, so on 200 questions it is *promising but not statistically conclusive*.
  Blending changed the #1 answer for only 29 questions (13 fixed, 16 broken).
- **The candidates are fine.** The correct page is somewhere in the 50
  candidates for 96% of questions. The reranker has room to move it up; it just
  is not better than the trained search at deciding *which* one.

**Why a strong reranker loses here.** The Hebrew search model Webiks ships was
fine-tuned on this exact QA file (it is literally the *training* dataset). So the
"before" column is a model that has already seen every test question, which is
why it scores 44% at #1 here versus the ~36% Webiks reports on unseen questions.
A general multilingual reranker that has never seen Kol-Zchut is competing with a
model that memorised the answer key. Blending works because it keeps the trained
model's opinion in play instead of discarding it.

Speed on a GPU: about 2 seconds per question to re-read all 50 candidates in
half precision. All per-question rankings, scores, confidence intervals and file
hashes are in `rag_eval/results/blend/` and `rag_eval/results/replace/` for review.

---

## 5. Integrating it into the Demo backend

- The reranker is added as an **optional step inside the engine's search**, off by
  default — with it off, behavior is byte-for-byte the original. Settings:
  `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_TOP`, `RERANK_MAX_SEQ`, `RERANK_DTYPE`,
  `RERANK_MODE` (blend / replace) and `RERANK_BLEND_K`. Bad values fail at startup
  with a clear message.
- How many candidates it re-reads is a setting (`RERANK_TOP`): all 50 on a GPU,
  20 on a CPU so a single query stays responsive. The model loads once at startup.
- Unit tests cover: reranking happens *before* pages are collapsed (so the best
  paragraph of a page is the one kept), the un-reranked tail keeps its order, tie
  handling, half-precision score ordering, the blend arithmetic, and invalid
  settings. A separate check (`scripts/verify_demo.py`) queries the running HTTP
  API and compares its page order with the offline evaluation.
- The answer (LLM) step is left behind a small swappable interface with a **mock**
  option, so the whole backend runs **locally with no API key**.
- No change to Elasticsearch, the corpus, or the trained model — the reranker is a
  self-contained add-on.

---

## 6. Honest limitations

- **The test questions are the search model's training questions.** There is no
  held-out set in the assignment data, so every number here is measured on
  questions the baseline has already seen. This inflates "before" and
  systematically disadvantages any add-on model. On genuinely new user questions
  I would expect the reranker to help more, but I cannot show that with this data.
  The fair fix is a held-out set of fresh questions, which was out of scope for 72
  hours.
- **200 questions is not many.** The top-5 gain from blending is real on this
  sample but its confidence range still crosses zero. A convincing claim needs
  roughly 4x the questions.
- **Blend strength was chosen on the same 200 questions.** I picked `k = 5` by
  re-scoring the saved rankings offline and then re-ran the real code path to
  confirm. That is a mild form of tuning on the test set; a held-out set would
  settle it.
- **Latency.** Re-reading 50 candidates costs about 2 s per question on a GPU and
  minutes on a CPU. For a CPU-only deployment, `RERANK_TOP=20` or the reranker off
  is the practical choice. Blending also happens to be safer here: shallow
  reranking (top 5-10 pages) scored as well as deep in the offline re-scoring.
- **An earlier, easier test agreed.** On a 2,000-page subset with 100 questions,
  "replace" also slightly hurt (#1 75% to 72%). That run is what motivated moving
  to the full corpus and then to blending.

---

## 7. How to reproduce

From the project root, with the data + model in place (see `UPSTREAM.md`):

```
# 1. Build the test set + embed the full corpus once (minutes on a GPU, cached):
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --pages 25000 --tag full

# 2. Baseline (search only):
.venv\Scripts\python rag_eval\eval_baseline.py --tag full

# 3. Reranker: prints before / replace / blend side by side from one run:
.venv\Scripts\python rag_eval\eval_reranker.py --tag full --n-questions 200 --top-rerank 50 --dtype float16 --mode blend

# 4. Confidence intervals, per-question wins/losses, export for review:
.venv\Scripts\python rag_eval\summarize_results.py --tag full --name blend
```

Or run all four with `scripts\run_eval.ps1`. Everything is resumable: a stopped
run continues where it left off. Drop `--dtype float16` on a CPU.

Code layout: `rag_eval/common.py` (shared retrieval logic, mirrors the real engine),
`build_subset.py` (one-time prep), `eval_baseline.py` (before), `eval_reranker.py`
(after), `summarize_results.py` (statistics + export). Engine side:
`webiks_hebrew_ragbot/reranker.py`, `rank_fusion.py`, `config.py`, and the hook in
`engine.py`. See `rag_eval/README.md` and `LOCAL_DEMO.md` for fuller walkthroughs.
