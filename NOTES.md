# Working Notes — Hebrew RAG Home Task

> Scratch / status file. Plain-language, high-level. NOT the final submission.
> At the end we'll distill the good parts into `SUBMISSION.md`.

---

## Status tracker

- [x] Read the task
- [x] Clone the repos and understand how the system works
- [x] Pick an improvement direction — **reranker** (see below)
- [x] Get the data + model downloaded — corpus ✓, QA questions ✓, Hebrew model ✓ (unzipped, ready)
- [x] Set up a clean Python environment (`.venv`) that actually works
- [x] Build a before/after measurement — **scripts written + smoke-tested ✓** (`rag_eval/`)
- [ ] Run the real baseline (the ~1-2h one-time encode) — **DO LATER, see "How to run" below**
- [ ] Build the reranker and measure again
- [ ] Wire the reranker into the demo backend (must still run locally)
- [ ] Write `SUBMISSION.md` (1–2 pages) + 2–3 slides

---

## What this system is (in plain terms)

It's the same idea as the OpenAI vector store, but Webiks built every piece themselves so it works well in **Hebrew**:

- **Chunks:** the Kol-Zchut website (Israeli rights/benefits info) cut into paragraphs.
- **Embedder:** a model they trained to turn Hebrew text into "meaning-numbers" (a fingerprint that captures what the text is about).
- **Search + storage:** Elasticsearch — a database that's good at "find the paragraphs whose fingerprints are closest to the question."
- **Answer step:** the top paragraphs get sent to GPT to write the final answer. This part can be *faked* ("mock" mode), because the task only grades the **finding**, not the **writing**.

**The whole task is really about one thing: how well does it find the right paragraphs?**

---

## The improvement: today vs. reranker

### The problem today

When you ask a question, the system grabs the 50 closest paragraphs by their rough fingerprint match — then keeps only the top few and throws the rest away.

The catch: the fingerprint match is **fast but crude**. It's good at pulling the right *neighborhood* of ~50 paragraphs, but bad at ranking *which one* is actually the answer. So the real answer is often somewhere in the 50 — but sitting at position #7, not #1 — and gets thrown away before anyone reads it.

> Analogy: a librarian who glances at 50 book *titles* that look related, without opening any of them. The right book is usually on the cart — just not the one on top.

### The fix: add a reranker (a "re-orderer")

Add a second, more careful step. It takes those same 50 paragraphs and actually reads each one *together with the question*, then re-sorts them by real relevance. Now the genuinely-best paragraph moves to the top and survives the cut.

**Two stages: a fast grabber to shortlist 50, then a careful reader to rank them.**

```
TODAY:
  question -> fast search -> 50 candidates -> keep top ~3 by the ROUGH score -> answer
                                               (real answer might be #7 -> thrown away)

WITH RERANKER:
  question -> fast search -> 50 candidates -> CAREFUL re-check reorders all 50
                                           -> keep top ~3 by the GOOD order -> answer
                                               (real answer moves up to #1-3 -> kept)
```

We do **not** replace the fast search — we still need it to narrow thousands of paragraphs down to 50. We just make better use of those 50.

### Why we chose this

- **Doesn't fight their trained model.** The fast Hebrew search stays exactly as-is; the reranker is a separate second opinion added on top.
- **Low risk / clean.** It's one extra step that can be turned on or off; if off, everything works like before.
- **Easy to prove it helped.** We run the same questions before and after and count how often the right paragraph lands near the top.

### Other options we considered (and why not)

The task says to pick **one** improvement, so here's what else was on the table and why the reranker won:

1. **Mix in keyword matching ("hybrid search").**
   Today it only matches by *meaning*. This option adds old-fashioned *exact-word* matching too, which helps with names, numbers, and specific terms the meaning-match can blur.
   *Why not:* it tries to do a better job at the *same* step their trained Hebrew model already does — and since that model was trained on these exact questions, it's already strong and hard to beat head-on. The reranker instead adds a *new* step on top, which is a safer win. (This is our clear second choice / possible add-on.)

2. **Change how paragraphs are cut up ("chunking").**
   How the text is split into paragraphs affects what can be found. Better splitting can help.
   *Why not:* it means re-processing and re-storing the **entire** corpus, which takes hours on this machine, and it's hard to prove the splitting specifically was what helped. High effort, fuzzy story.

3. **Rewrite the question before searching ("query expansion").**
   Add synonyms / rephrasings to the question so the search casts a wider net.
   *Why not:* the results are hit-or-miss and hard to measure convincingly — sometimes it helps, sometimes it adds noise. Weak, unclear payoff.

4. **Replace or retrain the Hebrew model itself.**
   *Why not:* huge effort, needs a strong GPU and hours of training, and it throws away the very thing Webiks built and are proud of. Wrong direction for a 72-hour task.

**Bottom line:** the reranker gives the best mix of *clear, provable improvement* + *low risk* + *respects their existing model*, which is why we picked it.

---

## Important caveat to be honest about

The set of questions Webiks gives us for testing (the "QA dataset") is the **exact same set their embedder was trained on**. So the fast search already "knows" these questions and will look strong. Our improvement has to add value *on top of* an already-good baseline — which is another reason the reranker (a separate, added signal) is a safer bet than trying to beat their model at its own game.

We should measure the baseline **first**. If there's clear room to improve, the reranker story is strong. If the baseline is already near-perfect, we rethink.

---

## Strong signal: the authors' own report card (found inside the model zip)

The model download included Webiks' own evaluation of their trained search (`eval/Information-Retrieval_evaluation_results.csv`). Their final numbers:

| How often the correct page is... | Rate |
|---|---|
| ranked **#1** | **~36%** |
| in the top 3 | ~59% |
| in the top 5 | ~66% |
| in the **top 10** | **~71%** |
| MRR@10 (a "how high up" score) | 0.485 |

**Why this matters:** the right page is in the top 10 ~71% of the time, but ranked #1 only ~36% of the time. That big gap is *exactly* the problem a reranker fixes — the answer is caught in the net but not placed on top. In principle, reranking could push the "#1" rate from ~36% toward the ~71% ceiling. This is direct, authors'-own evidence that we picked a real, worthwhile problem.

(Caveats: measured on a small sample; we'll still produce our own before/after numbers. Also: the embedder is `me5-large`-based, 1024-dim, trained with MultipleNegativesRankingLoss.)

---

## How we measure (built + smoke-tested)

- Take a subset of the QA questions (each has a known correct page).
- For each question, reproduce exactly what the real system does — offline, in plain math, no Elasticsearch/Docker needed:
  1. turn the question into meaning-numbers with the same Hebrew model,
  2. score every paragraph by closeness,
  3. take the top 50 (same as the real system), collapse to unique pages,
  4. check: did a correct page land at #1 / top-3 / top-5 / top-10? (+ an "how high up" score, MRR)
- Report those numbers for the baseline, then again after adding the reranker. Same questions, same haystack → a fair before/after.
- Why offline math is safe: the real system's search is literally a cosine-similarity lookup; copying that in numpy gives identical rankings but runs instantly and lets us reuse one cached encoding for both baseline and reranker. The real Elasticsearch is only needed for the final live demo, not for measuring.

### The one slow reality (why we subset)

- This machine is **CPU-only** (no GPU) and encodes ~1 paragraph every **~3 seconds** at full length.
- Paragraphs are long (median ~410 tokens, 94% near the 512 limit), so we must NOT shorten them — that would change results.
- => Encoding the full 24k corpus would take ~17 hours. The task explicitly allows using a subset, so we do.
- Smoke test confirmed: with a haystack of only correct pages (no distractors) the score is 100% — meaningless. The real run needs distractors so the test is honest.

### HOW TO RUN THE REAL BASELINE (do later)

Open a terminal in `C:\Users\GIGABYTE\Documents\webiks` and run these two commands.

**Step 1 — build + encode the test subset (the slow, one-time part):**
```
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --haystack 2000 --tag main
```
- `--haystack` = how many paragraphs to search through. This is the time knob:
  - ~1000 paragraphs ≈ 50-60 min (fastest, but an easier test)
  - ~2000 paragraphs ≈ 1.5-2 hours (good balance — recommended)
  - ~3500 paragraphs ≈ ~3 hours (most realistic / most convincing)
- It shows a progress bar with a live time estimate. Runs once; results are cached in `rag_eval\cache\main\`.

**Step 2 — measure the baseline (fast, a few minutes):**
```
.venv\Scripts\python rag_eval\eval_baseline.py --tag main
```
- Prints the before-numbers (correct page at #1 / top-3 / top-5 / top-10, and MRR@10) and saves them.
- Also saves each question's top-50 candidates so the reranker step can reorder the exact same ones.

Tip: you can leave Step 1 running in the background / overnight. Once it's done, Step 2 and all the reranker work are quick.

### Environment (already set up, for reference)

- Clean virtual env at `.venv` (the base Anaconda Python had a broken numpy/pandas).
- Pinned to versions that work together: `torch==2.3.1` (CPU), `sentence-transformers==3.0.1`, `transformers==4.42.3`, `numpy<2`. (Newer torch/transformers hit Windows DLL / bug issues.)
- Scripts live in `rag_eval/`: `common.py` (shared logic), `build_subset.py` (step 1), `eval_baseline.py` (step 2). The reranker script comes next.

---

## What the data looks like (looked at 2026-09-11)

**QA file** (`Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv`) — this is our **answer key** for testing. 4 columns:
- `question` — a real Hebrew question a user asked
- `paragraph` — a paragraph that correctly answers it
- `link` — the Kol-Zchut web page it came from
- `doc_id` — the ID of that page

Numbers:
- 3,890 rows, but only **2,950 unique questions** (some questions appear more than once).
- ~77% of questions have exactly **1 correct page**; ~23% have **2+** correct pages (a few up to 7).
- Answers point to **1,640 different pages**.

Why it matters:
- We grade automatically — ask each question, compare the pages returned against this key. No human judging.
- Because some questions have several right answers, we'll score *"did at least one correct page show up near the top?"* — the fair and simple way.

**Why the same question appears multiple times:** each row is one *(question → one correct paragraph)* pair, so a question with several correct answers gets several rows.
- 700 of 2,950 questions repeat; 689 of those because they have **multiple correct pages**, 19 because of **multiple correct paragraphs on the same page**.
- Example: the question about whether the Freedom of Information Law applies to private insurers has **two** correct pages — the law itself (1963) and how to file a request (7487) — so it appears twice.
- Not a data bug — it's the answer key saying "any of these pages counts as correct." This is exactly why we score "at least one correct page near the top."

**Paragraph corpus** (`Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json`, ~150 MB) — the "haystack" the system searches. Downloaded ✓. Columns: `doc_id, title, content, link, license`.
- **24,487 paragraphs** across **7,007 pages** (~3.5 paragraphs/page, biggest page = 79).
- **All 1,640 answer-key pages are present in the corpus** (0 missing) → we can grade cleanly.
- Small enough (~24k, not millions) that we can process the **whole** corpus for measuring — no tiny subset needed. The "indexing takes hours" warning is about the slow database, which we skip for measuring.

**How they fit (open-book exam analogy):** corpus = the textbook the system searches; QA file = the answer key saying which page is correct for each question.

**Extra fact:** the Hebrew embedder is built on `me5-large` (multilingual-e5-large); paragraphs were split to fit its 512-token limit.

**Heads-up:** the base Anaconda Python has a broken pandas/numpy. We'll make a clean, separate environment for the real work.

---

## Environment notes

- Working folder: `C:\Users\GIGABYTE\Documents\webiks`
- Cloned: `Webiks-Hebrew-RAGbot-Demo` (backend), `Webiks-Hebrew-RAGbot` (search engine), QA dataset repo.
- Have: Python 3.11, git, conda. No Docker yet (needed later for the live demo's database).
- Data still to download: paragraph corpus + QA questions CSV (Google Drive), the Hebrew model (Google Drive).

---

## Open questions / decisions

- Which exact reranker model to use (needs to handle Hebrew). — TBD, decide after baseline.
- How big a corpus subset to use for measuring. — TBD.
- Docker for the final live demo, or an easier alternative. — TBD.
