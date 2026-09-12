# Working Notes — Hebrew RAG Home Task

> Working diary, in plain language. The final, graded write-up is
> `SUBMISSION.md`; how to run things is in `LOCAL_DEMO.md` and `rag_eval/README.md`.
> The first section below is the up-to-date story. Everything after the
> "History" line is kept as it was written at the time (the reranker plan, early
> numbers on a small test), so the reasoning trail is visible.

---

## What actually happened (the short version)

We tried **three** improvements. Each was built, integrated, and measured the
same honest way: pick any setting on the first half of the questions, report it
once on the untouched second half, ship only if the **first result** improves
without the top-5 getting worse. Two failed that test; the third passed clearly.

| # | Idea | Held-out: correct page at #1 | Verdict |
|---|---|---|---|
| 1 | **Cross-encoder reranker** (BGE) re-reads question + paragraph, re-sorts | 44% → 41% (replace), 45% → 40% (blend) | rejected |
| 2 | **Hybrid: dense + BM25 keyword search** (reciprocal rank fusion) | 45% → 36% | rejected |
| 3 | **Page scoring**: best paragraph + second paragraph + title match | 38.8% → **54.8%** (500 fresh questions, 250/250) | **shipped** |

### Why the reranker did not work (this surprised us, so here it is properly)

A reranker is normally the safest retrieval upgrade there is. Here it lost, and
the reasons are specific to this system, not a bug:

1. **The search model already knows these questions.** Webiks' embedder was
   fine-tuned on the very QA file we test with (the task sheet says so). On these
   questions it is not a "rough fingerprint" at all: it is a model that has seen
   the answer key. A general reranker that has never seen Kol-Zchut is being
   asked to overrule an expert on the expert's own exam. It fixed some #1s
   (24 out of 200) and broke more (30).
2. **The reranker reads the paragraph alone; the topic lives in the title.**
   Kol-Zchut paragraphs are fragments of a page. Many say "the payment is X" or
   "apply at office Y" without naming the benefit. The page *title* names it.
   The reranker scores the paragraph text only, so it cannot tell two similar
   fragments from different pages apart. The search model does not suffer from
   this as much because it was trained on exactly these fragment ↔ question pairs.
3. **Deep reranking pulls weak pages up from far below.** Reranking all 50
   candidates let a look-alike from position 30 jump to #1. Restricting the
   reranker to the top few, or blending its order with the search order,
   recovered most of the loss (top-5 even went up) but never beat the baseline
   at #1 on the held-out half. A gain only on ranks 4–5 is not what users feel.
4. **It is also slow and heavy.** A 2.2 GB model, ~2 s per question on a GPU and
   minutes on a CPU, for a result that was not better. Not worth shipping.

**The lesson that led to the fix:** don't fight the trained model; give it more
to say. The title and the second paragraph are signals the *same* model
produces, and combining them is what finally moved the first result.

### Why hybrid (BM25) did not work

Exact-word matching is weak on this corpus on its own (13% at #1): Hebrew
inflection, many pages sharing the same official terms, and questions phrased
colloquially. Fusing a weak ranker with a strong one mostly imports the weak
one's mistakes near the top. It did help recall a little (top-5), same pattern
as the reranker: better at ranks 4–5, worse at #1.

### Why page scoring worked

The baseline finds the right page (in the top 10 for ~87% of questions) but
ranks it by its single best paragraph. The right page usually has *several*
matching paragraphs and a title that names the topic; a look-alike page has one
lucky paragraph. Scoring the page by best + 0.25 × second-best + 0.25 × title
match, with a small gate so generic "hub" pages can't jump from far below, lifts
the right page to #1 for 16 more questions in every 100, with no new model and
no re-indexing. Full numbers and limitations: `SUBMISSION.md` section 4 and 6.

### Status

- [x] Understand the system, get data + model, clean environment, measurement harness
- [x] Baseline on the full corpus (24,487 paragraphs)
- [x] Reranker: built, integrated (optional), measured — **rejected** (numbers above)
- [x] Hybrid BM25 + dense: built, measured — **rejected**
- [x] Page scoring: built, integrated (optional, on in the local demo), measured on 500 fresh questions — **shipped**
- [x] `SUBMISSION.md` written around the shipped result
- [x] 2–3 slides for the interview (in `slides/`; export a PDF from the canvas to present)
- [ ] Run `scripts/verify_demo.py` once on the GPU PC against the live API and commit its output

---

## File map — what each file we added is for

Plain one-liners so a reviewer (or future me) knows why each file exists.
Everything not listed here is the original upstream Webiks code, unchanged.

**The write-up and this diary**
- `SUBMISSION.md` — the graded 1–2 page write-up: the improvement, why, results, limits.
- `NOTES.md` — this working diary (the reasoning trail; not the graded doc).
- `README.md` — repo front page: what shipped, the one headline number, where to look.
- `LOCAL_DEMO.md` — how to run the upgraded backend locally (Elasticsearch, seed, launch).
- `UPSTREAM.md` — where the big files (corpus, model, QA) come from; how to set up the env.

**The shipped improvement (page scoring)** — in the search engine
- `webiks_hebrew_ragbot/page_order.py` — the scoring rule itself (one small, dependency-free function). Shared by the engine and the evaluation so both behave identically.
- `webiks_hebrew_ragbot/page_scoring.py` — wraps that rule for the live engine: groups the search hits by page, embeds the titles, reorders.
- `webiks_hebrew_ragbot/engine.py` / `config.py` — **edited** to call page scoring as an optional step (off by default) with validated settings.
- `tests/test_page_scoring.py` — unit tests for the rule and the engine wiring.

**The two ideas I tried and rejected** — kept as evidence for the write-up
- `webiks_hebrew_ragbot/reranker.py` — the cross-encoder (BGE) reranker step. Optional, off. Rejected.
- `webiks_hebrew_ragbot/rank_fusion.py` — the "combine two rankings" maths (used by the reranker's blend mode and by the hybrid eval).
- `tests/test_reranker.py`, `tests/test_rank_fusion.py` — their unit tests.

**The evaluation harness** — proves the before/after honestly (`rag_eval/`)
- `common.py` — shared logic; reproduces the real engine's search offline in plain maths.
- `build_subset.py` — step 1: pick questions, gather pages, embed them once (slow, cached).
- `eval_baseline.py` — step 2: measure today's system (the "before"); save each question's 50 candidates.
- `eval_page_scoring.py` — the shipped improvement's before/after, dev/held-out split.
- `eval_reranker.py`, `sweep_blend.py`, `summarize_results.py` — the reranker's evaluation, its no-model tuning sweep, and its stats/export. Evidence for the rejection.
- `eval_hybrid.py` — the dense+BM25 hybrid evaluation. Evidence for that rejection.
- `README.md` — walkthrough of the harness.
- `results/page_scoring_500/`, `results/page_scoring_200/` — the shipped result (headline + earlier pilot).
- `results/replace/`, `results/blend/` — the reranker's committed numbers.

**Running it**
- `scripts/run_eval.ps1` — one command: baseline + the 500-question page-scoring run (alternatives behind a flag).
- `scripts/seed_demo.py` — load the corpus into local Elasticsearch for the demo.
- `scripts/run_demo.py` — launch the backend locally with page scoring on and a mock answer step (no API key).
- `scripts/verify_demo.py` — send held-out questions to the running API and check the pages match the offline result.

**The interview slides**
- `slides/*.dc.html`, `slides/canvas.json` — the three slides (problem / fix / results), editable source.
- `slides/hebrew-rag-page-scoring-slides.html` — the built, viewable slide file (~2.5 MB). Big; consider not committing it and delivering the PDF instead.

---

## History (as written at the time; superseded by the section above)

## Status tracker (old)

- [x] Read the task
- [x] Clone the repos and understand how the system works
- [x] Pick an improvement direction — **reranker** (see below)
- [x] Get the data + model downloaded — corpus ✓, QA questions ✓, Hebrew model ✓ (unzipped, ready)
- [x] Set up a clean Python environment (`.venv`) that actually works
- [x] Build a before/after measurement — **scripts written + smoke-tested ✓** (`rag_eval/`)
- [x] Run the real baseline (the ~1.5h one-time "study" step) — **DONE ✓ (numbers below)**
- [x] Build the reranker and measure again — done; it did not help (see top)
- [x] Wire the reranker into the demo backend (must still run locally) — done, kept optional/off
- [x] Write `SUBMISSION.md` (1–2 pages) — done; slides pending

### Why the three "setup" steps matter (in plain words)

The one-liner: **the ingredients, the working kitchen, and the scale that proves the new recipe is better.**

- **Get the data + model (the ingredients).** To test anything you need three things: the **corpus** (~24,000 paragraphs from the Kol-Zchut site — the "textbook" answers live in), the **QA questions** (real Hebrew questions each paired with the correct page — our **answer key**), and the **Hebrew model** (the trained tool that turns text into "meaning-numbers" and decides which paragraphs are close to a question). Without all three there's nothing to measure. They're big, so they're kept out of the repo (see `.gitignore`).
- **Clean Python environment `.venv` (the working kitchen).** A private toolbox of software for this project. We built a fresh one because the computer's default was broken, and we installed **the exact library versions Webiks used**. This means our code runs the model the same way the real system does, so the numbers are trustworthy and anyone can reproduce them.
- **Before/after measurement `rag_eval/` (the scale).** The heart of the task. It asks the answer-key questions and checks **how often the correct page shows up** at #1 / top-3 / top-5 / top-10. We run it once on today's system ("before") and once on the reranker version ("after") — same questions, same scale, fair comparison. "Scripts written + smoke-tested" = the code is done and a tiny trial run works; the full run on hundreds of questions is the slow step saved for later (see below). This is our **evidence**: not "I improved it" but "here are the before/after numbers".

---

## What this system is (in plain terms)

It's the same idea as the OpenAI vector store, but Webiks built every piece themselves so it works well in **Hebrew**:

- **Chunks:** the Kol-Zchut website (Israeli rights/benefits info) cut into paragraphs.
- **Embedder:** a model they trained to turn Hebrew text into "meaning-numbers" (a fingerprint that captures what the text is about).
- **Search + storage:** Elasticsearch — a database that's good at "find the paragraphs whose fingerprints are closest to the question."
- **Answer step:** the top paragraphs get sent to GPT to write the final answer. This part can be *faked* ("mock" mode), because the task only grades the **finding**, not the **writing**.

**The whole task is really about one thing: how well does it find the right paragraphs?**

### How the data gets loaded (embed → save)

Before anything can be searched, the paragraphs have to go into Elasticsearch. Good to know for the interview: **the code does NOT do the chunking** — the corpus file already comes cut into paragraphs. Loading is two steps:

1. **Embed each paragraph** (in `engine.py`, `update_docs` / `create_paragraphs`): run the Hebrew model on the paragraph's text to make its fingerprint, and attach that fingerprint to the paragraph. Key line: `content_vectors = self.retrieval_model.encode(...)`.
2. **Save it** (in `elastic_model.py`, `create_paragraph` / `create_or_update_documents`): write the text + its fingerprint into Elasticsearch. Key line: `self.es_client.index(index=index, body=doc)`. (If updating, it deletes the old copies of that page first, so no duplicates.)

The backend kicks this off through its `/update` route (`main.py`). One nice detail: **search later uses the exact same `encode(...)` on the question**, so the question and the paragraphs are measured with the same ruler — that's the only reason comparing them works.

Interview note: since chunking happens *before* the code, "better chunking" would be a **different** improvement lever than the reranker we picked — worth mentioning as a considered-but-not-chosen option.

---

## The improvement (original plan): today vs. reranker

> This was the plan before measuring. The reranker was built and measured and
> did **not** hold up; see "What actually happened" at the top. The problem
> description here is still right; the chosen fix changed.

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
   *Why not:* huge effort, needs a powerful graphics card and hours of training, and it throws away the very thing Webiks built and are proud of. Wrong direction for a 72-hour task.

**Bottom line (at the time):** the reranker gives the best mix of *clear, provable improvement* + *low risk* + *respects their existing model*, which is why we picked it.

**What we learned:** "respects their existing model" turned out to be the whole
game. The reranker *replaced* the model's opinion and lost; hybrid search
*diluted* it and lost; page scoring *extends* it (same model, more of its
signals) and won. Option 1 above (hybrid) was also tried after the reranker and
rejected for the reasons at the top.

---

## Important caveat to be honest about

The set of questions Webiks gives us for testing (the "QA dataset") is the **exact same set their embedder was trained on**. So the fast search already "knows" these questions and will look strong. Our improvement has to add value *on top of* an already-good baseline — which is another reason the reranker (a separate, added signal) is a safer bet than trying to beat their model at its own game.

We should measure the baseline **first**. If there's clear room to improve, the reranker story is strong. If the baseline is already near-perfect, we rethink.

*(Later: this caveat turned out to be the decisive fact. See the top section.)*

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

(Small print you can skip: this was measured on a small sample; we still make our own before/after numbers. The Hebrew model is built on a well-known ready-made multilingual model called `me5-large`.)

---

## Our baseline result (the "before" number) — measured 2026-09-12

We ran our own test: **200 questions**, searching through a **2,000-page** set (all the pages that hold answers, plus lots of random other pages mixed in). Here's how today's system did:

| How often the correct page is... | Our result |
|---|---|
| ranked **#1** | **76.5%** |
| in the top 3 | 93.0% |
| in the top 5 | 96.5% |
| in the **top 10** | **98.0%** |
| "how high up", on average (MRR@10) | **0.852** |

**What it means in plain words:** for about 3 out of 4 questions the right page is already at the very top, and almost always (98%) it's somewhere in the top 10.

**Where the reranker can help:** the right page is in the top 10 **98%** of the time but at #1 only **76.5%** of the time. That gap — about **21 questions out of 100** — is the target: the answer is already found, just not on top. The reranker's job is to lift those to #1, which should push the "#1" rate and the "how high up" score up.

**Honest note:** these numbers are higher than Webiks' own (~36% at #1). That's expected — we test on a smaller 2,000-page set, so there's less to sift through (an easier exam). It's still a fair before/after because the reranker faces the exact same exam. It just means the room to improve is smaller here, so any gain is meaningful.

*(Later: we moved to the full corpus on a GPU PC. There the baseline is ~39–44%
at #1, close to Webiks' own number, and that is where all the final
measurements were made.)*

---

## How we measure (built + smoke-tested)

- Take a subset of the QA questions (each has a known correct page).
- For each question, reproduce exactly what the real system does — offline, in plain math, no Elasticsearch/Docker needed:
  1. turn the question into meaning-numbers with the same Hebrew model,
  2. score every paragraph by closeness,
  3. take the top 50 (same as the real system), collapse to unique pages,
  4. check: did a correct page land at #1 / top-3 / top-5 / top-10? (+ an "how high up" score, MRR)
- Report those numbers for the baseline, then again after adding the reranker. Same questions, same set of pages → a fair before/after.
- Why doing it ourselves is safe: the real system's search is just a "how close are these two things" calculation. We redo that exact same calculation in our own code — same ranking, but it runs instantly and lets us reuse the one-time "studying" for both the before and the after test. The real search database (Elasticsearch) is only needed for the final live demo, not for measuring.

### The one slow reality (why we subset)

- This machine has no graphics card, so "studying" pages is slow — about **1 page every 3 seconds**.
- Pages are long (close to the model's size limit), and we must NOT shorten them — that would change the results.
- => Studying all 24,000 pages would take ~17 hours. The task explicitly allows using a smaller set, so we do (2,000 pages ≈ 1.5 hours, done once).
- Quick trial confirmed: if the set is made of *only* correct pages (no random extras) the score is 100% — meaningless. The real run needs random extra pages mixed in so the test is honest.

### HOW TO RUN THE BASELINE (already done once — here's how to re-run)

Open a terminal in `C:\Users\GIGABYTE\Documents\webiks` and run these two commands.

**Step 1 — build the test set + "study" the pages (the slow, one-time part):**
```
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --pages 2000 --tag main
```
- `--pages` = how many pages to search through in total. This is the time knob:
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
- **Fair-measurement detail:** Step 2 cuts each question to the same length limit that Step 1 used on the paragraphs (it reads that number back from the saved cache). Questions are short so this basically never changes anything, but it keeps both sides measured exactly the same way — no accidental apples-to-oranges.

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

**Paragraph corpus** (`Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json`, ~150 MB) — all the pages the system searches through. Downloaded ✓. Columns: `doc_id, title, content, link, license`.
- **24,487 paragraphs** across **7,007 pages** (~3.5 paragraphs/page, biggest page = 79).
- **All 1,640 answer-key pages are present in the corpus** (0 missing) → we can grade cleanly.
- Small enough (~24k, not millions) that we can process the **whole** corpus for measuring — no tiny subset needed. The "indexing takes hours" warning is about the slow database, which we skip for measuring.

**How they fit (open-book exam analogy):** corpus = the textbook the system searches; QA file = the answer key saying which page is correct for each question.

**Extra fact:** the Hebrew model is built on a well-known ready-made multilingual model (`me5-large`); the pages were cut into paragraphs small enough for that model to read in one go.

**Heads-up:** the base Anaconda Python has a broken pandas/numpy. We'll make a clean, separate environment for the real work.

---

## Submission repo (set up)

- **Private repo:** https://github.com/AfekDa/Webiks-Hebrew-RAGbot-Home-Task
- One repo holding both projects as subfolders: `Webiks-Hebrew-RAGbot-Demo/` (backend) and `Webiks-Hebrew-RAGbot/` (search engine), plus `rag_eval/` and the notes.
- **Commit strategy for easy review:** commit #1 is the pristine upstream code (see `UPSTREAM.md` for exact sources/commits); every later commit's diff shows exactly what we changed for the task.
- Big files (corpus, QA csv, model, `.venv`, eval cache) are **not** in git — see `.gitignore` + `UPSTREAM.md` for where to download them.
- The improvement will touch **both** subfolders: the reranker step goes in the engine's search; the backend is pointed at the upgraded engine. Reranker kept optional so existing behavior still works.
- Reviewers: when ready, add them as collaborators on the private repo (Settings → Collaborators), or we can switch to a zip.

---

## Environment notes

- Working folder: `C:\Users\GIGABYTE\Documents\webiks`
- Cloned: `Webiks-Hebrew-RAGbot-Demo` (backend), `Webiks-Hebrew-RAGbot` (search engine), QA dataset repo.
- Have: Python 3.11, git, conda. No Docker yet (needed later for the live demo's database).
- Data still to download: paragraph corpus + QA questions CSV (Google Drive), the Hebrew model (Google Drive).

---

## Open questions / decisions (resolved)

- Which exact reranker to use. — **BAAI/bge-reranker-v2-m3** (multilingual, handles Hebrew). Built and rejected.
- How big a page set to use for measuring. — started with 2,000 pages on CPU; **final: the full corpus** on a GPU PC.
- Docker for the final live demo, or an easier alternative. — **standalone Elasticsearch 8.12.2** under `.runtime/` (see `LOCAL_DEMO.md`); Docker also works.
- What to ship. — **page scoring** (see top). Reranker and hybrid stay in the code, off by default, as evaluated alternatives.
