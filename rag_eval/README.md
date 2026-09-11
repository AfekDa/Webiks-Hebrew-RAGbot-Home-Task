# `rag_eval/` — the before/after measurement (plain-language walkthrough)

This folder is our **measuring tool**. It grades the retrieval system like a
**closed-book exam**: we have questions, we know the correct answers, and we
check how often the system finds them.

The one-line mental model for the three files:
**shared tools → prepare the exam (once) → run the exam.**

---

## 1. `common.py` — the shared toolbox

Does nothing on its own; holds the pieces the other two files both use, so we
don't write them twice. Four things live here:

- **`load_corpus()`** — opens the big paragraph file and returns a clean list of
  paragraphs (page id, title, text, link). The file is stored in an awkward
  column shape; this just straightens it into rows.
- **`load_qa()`** — opens the answer key and builds *question → set of correct
  pages*. It's a **set** because one question can have several correct pages.
  So a question counts as correct if the system finds **any** of them.
- **`rank_pages(sims, ...)`** — the heart. Copies **exactly what the real Webiks
  system does** when it searches:
  1. it's given a closeness score for every paragraph (vs. the question);
  2. it takes the **top 50 paragraphs** — same as the real system's
     Elasticsearch `size=50`;
  3. it **collapses paragraphs to pages**: several paragraphs can belong to the
     same page, so it keeps each page the first time it appears. Result: a
     ranked list of **pages**, best first.
  - Why this matters: the exam is graded on **pages**, not paragraphs.
- **`score_ranking(...)`** — the grader. Finds the position of the first correct
  page and reports: was it at #1? top 3? top 5? top 10? Plus the reciprocal
  rank (see metrics below).

One line to remember: *"`common.py` reproduces the real search — top 50
paragraphs, then dedup to pages — but in plain numpy, so I don't need to run
Elasticsearch."*

### What it's copying (so you can check it's faithful)

`rank_pages` deliberately mirrors the real Webiks search function,
**`Engine.search_documents`**, in
`Webiks-Hebrew-RAGbot/webiks_hebrew_ragbot/engine.py:103`. That function does
three steps: (1) encode the question, (2) ask Elasticsearch for the closest
paragraphs (it returns the top 50), (3) loop through them and keep each **page**
(`doc_id`) the first time it appears, until it has `top_k` pages.

Our copy matches it step for step — the only difference is *how* we get the 50
closest: they use Elasticsearch (a running database), we do the same math in
numpy (`sims = q_emb @ emb.T`). Same result, no database to install — which is
exactly why these tests run on a laptop.

So if asked *"how do you know your offline test matches the real system?"*:
**"I copied `Engine.search_documents` step for step — top 50 paragraphs, then
dedup to unique pages — and only swapped Elasticsearch for the same calculation
in numpy."**

---

## 2. `build_subset.py` — prepare the exam (slow step, run once)

The problem it solves: turning all ~24,000 paragraphs into "meaning-numbers"
takes ~1–2 hours on a CPU laptop. We don't want to pay that every time, so this
does the slow part **once** and saves the result.

What it does, in order:
1. Picks a sample of questions to test (default 200).
2. Builds the **set of pages to search**. Always includes the **correct pages'**
   paragraphs (so the answer is findable), plus lots of random **other** pages
   up to the target size (default 2,000), so the test is not too easy.
3. Runs the Hebrew model over every page to make the meaning-numbers — **this
   is the slow part**.
4. Saves four files to a cache folder: the numbers, the pages, the questions,
   and the settings used.

Interview point — **"why a subset, not all 24,000?"** Two honest reasons: the
task explicitly allows a subset, and the full corpus is ~1–2 hours on CPU. It's
still fair because the correct answers are always included, mixed among ~2,000
random other pages — hard enough to be meaningful.

---

## 3. `eval_baseline.py` — run the exam (fast step)

Reads the saved cache from step 2 and produces the **"before"** numbers.

What it does:
1. Loads the saved page numbers from step 1.
2. Turns the **questions** into meaning-numbers (fast — a few hundred short
   questions).
3. For each question: scores it against every paragraph, calls `rank_pages`
   (top 50 → pages) and `score_ranking` (did we find a correct page, and where?).
4. Averages over all questions and prints: correct page at #1 / top-3 / top-5 /
   top-10, and MRR@10.
5. Saves each question's **top-50 paragraphs** — so the reranker later can
   reorder the exact same 50, giving a clean before/after.

The "score every paragraph" is one line: `sims = q_emb[qi] @ emb.T` — in plain
terms, *"compare this question's fingerprint to every paragraph's fingerprint
and get a closeness score for each."*

---

## The two metrics they'll ask about

- **hit@k** ("correct page in top-k"): out of all questions, the fraction that
  had a correct page somewhere in the top *k*. hit@1 = found it right at the
  top; hit@10 = found it within the first 10. Higher = better.
- **MRR@10** (mean reciprocal rank): rewards putting the correct page **high**.
  Right page at #1 → 1.0, at #2 → 0.5, at #3 → 0.33, nothing in top 10 → 0.
  Then average over all questions.

**Why both?** hit@k tells you *whether* the answer is in reach; MRR tells you
*how well-ordered* it is. The reranker's whole job is to push correct pages
**higher**, so MRR is the number we most expect to improve.

---

## How to run it

From the project root (`C:\Users\GIGABYTE\Documents\webiks`):

```
# Step 1 — slow, one-time (leave running / overnight if needed):
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --pages 2000 --tag main

# Step 2 — fast, prints the before-numbers:
.venv\Scripts\python rag_eval\eval_baseline.py --tag main
```

Tiny trial run to check things work (~3 min):

```
.venv\Scripts\python rag_eval\build_subset.py --questions 10 --pages 60 --tag smoke
.venv\Scripts\python rag_eval\eval_baseline.py --tag smoke
```

> Note: a tiny run scores ~100% because with only ~60 paragraphs the correct
> page is trivially easy to find. That only proves the plumbing works — it is
> **not** a real result. The real numbers come from the `main` run above.
