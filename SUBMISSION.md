# Hebrew RAG — Improving the Retrieval Step

**Task:** improve the *retrieval* part of the Webiks open-source Hebrew RAG system
(Kol-Zchut data), measure it honestly on a QA subset, and integrate it cleanly
into the demo backend so it still runs locally.

**What I changed, in one line:** the engine now ranks a page by **more than its
single best paragraph** — it also counts how well the question matches the page
**title** and the page's **second-best paragraph**, using the *same* trained
Hebrew model. Only the order of the retrieved candidates changes.

**Result, in one line:** on the full corpus, on 250 held-out questions the
weights were never tuned on, the correct page is at **#1 for 54.8% of questions
instead of 38.8%** (+16 points), MRR@10 **0.547 → 0.650**, with top-5 unchanged.

---

## 1. Where the system loses accuracy

When a user asks a question, the system turns it into a "meaning fingerprint"
with Webiks' trained Hebrew model, asks Elasticsearch for the **50 closest
paragraphs**, and collapses those to unique **pages**, keeping the top few.

The baseline is good at *finding* the right page but bad at *putting it first*:
on the full corpus the correct page is in the top 10 for ~87% of questions but
at #1 for only ~39%. The answer is on the cart, just rarely on top.

A page's rank today is simply the rank of its single best paragraph. So a page
with *one* paragraph that happens to look like the question can beat the page
that is actually *about* the question.

---

## 2. The improvement: page scoring

The retrieval path changes only after the existing model has produced its 50
paragraph candidates:

```text
today: question -> dense search -> 50 paragraphs -> best paragraph per page -> top ~3 pages -> answer

after: question -> dense search -> same 50 paragraphs -> score each candidate page -> top ~3 pages -> answer
```

The new page score is

```
score = best paragraph
      + 0.25 × second-best paragraph      (the right page usually has several matches;
                                           a look-alike has one)
      + 0.25 × title match                (Kol-Zchut titles name the topic:
                                           "child allowance", "mourning days")
```

All three numbers come from the same retrieval model (title match = cosine
between the question and the page title, embedded at query time for the ≤50
candidates). A small **margin gate** lets only pages whose best paragraph is
close to the top compete on the full score, so a generic "hub" page with a broad
title ("guide to foster care") cannot jump up from far below — the one way the
title signal was seen to misfire.

Why this works where two stronger-looking ideas failed (section 4): the shipped
retriever was **trained on this very QA data** and is unusually strong. Anything
that tries to *overrule* it loses. Page scoring does not overrule it; it listens
to it more carefully.

**Cost:** no new model, no new dependency, no re-indexing, one extra small
encode per query (≤50 short titles). A flag turns it off, restoring the original
behaviour exactly.

---

## 3. How I measured it

A small harness (`rag_eval/`) reproduces the engine's retrieval offline in
NumPy — same model, same "top-50 → collapse to pages" logic — so it runs without
Elasticsearch. **Metrics:** *hit@k* (is a correct page in the top k, for k = 1,
3, 5, 10) because #1 is what the user and the answer step actually see; *MRR@10*
because it rewards moving the correct page *upward*, which is the whole job.

**Discipline, applied to every idea:** choose any setting on the first half of
the questions (**dev**), report it *once* on the untouched second half
(**held-out**), and ship only if held-out hit@1 or MRR improves while hit@5 does
not regress. This is what separates a real gain from a hand-picked one.

**Data:** the complete corpus (24,487 paragraphs, 7,007 pages). Headline run:
**500 questions** that do not overlap an earlier 200-question pilot, split
250 dev / 250 held-out. The baseline was re-run independently and matches.

---

## 4. Results

**Held-out (250 questions the weights were never tuned on):**

| correct page is… | Before (search only) | After (page scoring) | Change |
|---|---|---|---|
| at **#1** | 38.8% | **54.8%** | **+16.0** |
| in top 3 | 65.6% | 71.6% | +6.0 |
| in top 5 | 77.2% | 78.4% | +1.2 |
| in top 10 | 86.8% | 87.6% | +0.8 |
| MRR@10 | 0.547 | **0.650** | **+0.10** |

Chosen on dev: second-paragraph weight 0.25, title weight 0.25, margin 0.05. The
earlier 200-question pilot on the same corpus pointed the same way (held-out
#1: 45% → 50%, MRR 0.598 → 0.626).

**Two alternatives, built, integrated and rejected by the same test:**

| Alternative | What it does | Held-out #1 | Why it loses |
|---|---|---|---|
| **Cross-encoder reranker** (BAAI/bge-reranker-v2-m3) | re-reads question+paragraph together, re-sorts; tried both replacing and blending with the search order | 45% → 40% (blend), 44% → 41% (replace) | a general model overruling a retriever trained on these questions; fixes some #1s, breaks more |
| **Hybrid dense + BM25** (reciprocal rank fusion) | adds exact-keyword search | 45% → 36% | BM25 alone is weak here (13% at #1); even dense-weighted fusion drags poor keyword matches near the top |

Both improved recall around ranks 4–5 and damaged the first result. They were
built and evaluated during development, then removed so the repo carries only
the shipped answer; the measured numbers above are the result of that work.

**Why a strong reranker lost, in plain words.** A reranker is usually the safest
retrieval upgrade, so this deserves an explanation:

- *The retriever has seen the exam.* The shipped embedder was fine-tuned on this
  QA file. On these questions it is an expert; a general reranker that has never
  seen Kol-Zchut was asked to overrule it and lost more often than it won
  (fixed 24 first results out of 200, broke 30).
- *The reranker reads the paragraph alone, but the topic lives in the title.*
  Kol-Zchut paragraphs are fragments ("the payment is X", "apply at office Y")
  that often don't name the benefit; the page title does. Two similar fragments
  from different pages look alike to the reranker. That observation is what led
  to adding the title match in page scoring.
- *Deep reranking lets look-alikes jump from far below.* Restricting the reranker
  to the top few, or blending its order with the search order, recovered most of
  the loss (top-5 even improved) but never beat the baseline at #1 on held-out
  questions. A gain only at ranks 4–5 is not what the user sees.

The same pattern explains BM25: exact-word matching alone is weak here (Hebrew
inflection, many pages sharing official terms), and fusing a weak ranker with a
strong one mostly imports the weak one's mistakes near the top.

---

## 5. Integration into the Demo backend

- Page scoring is an **optional step inside the engine's search**
  (`webiks_hebrew_ragbot/page_scoring.py`), applied to the Elasticsearch hits
  before pages are picked. Off by default (`PAGE_SCORING_ENABLED`); the local
  launcher turns it on with the evaluated weights. Settings are validated at
  startup.
- The scoring rule is one dependency-free function (`page_order.py`) shared by
  the engine and the evaluation, so what was measured is what runs.
- The answer (LLM) step is behind a mock, so the whole backend runs **locally
  with no API key**. No change to Elasticsearch, the corpus, or the trained model.
- **Tests:** unit tests for the page-scoring rule (title lift, second paragraph,
  margin gate, ties), that the engine keeps each page's best paragraph while
  reordering, that off = the original behaviour, and invalid settings.
  `scripts/verify_demo.py` sends held-out questions to the running API and
  checks the returned pages equal the offline improved ranking.

---

## 6. Honest limitations

- **The questions are the embedder's training questions.** There is no held-out
  question set in the assignment data, so none of these numbers are an
  out-of-domain estimate. The dev/held-out split guards against tuning on the
  test set, not against the model having seen the questions. On genuinely new
  user questions the *absolute* numbers will be lower; I expect the *direction*
  to hold, since titles and multi-paragraph evidence are properties of the pages.
- **Weights chosen from a small grid.** Three values each for two weights and
  the gate on/off, picked on 250 questions. Held-out confirmation on 250 more is
  the safeguard; a larger grid would need more questions.
- **Title match depends on good titles.** Kol-Zchut titles are consistently
  topical. On a corpus with vague titles the title weight should be re-tuned or
  set to zero (one setting).
- **Latency.** One extra encode of ≤50 titles per query: milliseconds on a GPU,
  well under a second on CPU.

---

## 7. How to reproduce

From the project root, with the data + model in place (see `UPSTREAM.md`):

```
# one command (GPU): embed the corpus, baseline, then the 500-question page-scoring run
powershell -ExecutionPolicy Bypass -File scripts\run_eval.ps1

# or step by step
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --pages 25000 --tag full
.venv\Scripts\python rag_eval\eval_baseline.py --tag full
.venv\Scripts\python rag_eval\build_subset.py --questions 500 --pages 25000 --tag full500 --seed 1729 --reuse-paragraph-cache full --exclude-questions-from full
.venv\Scripts\python rag_eval\eval_baseline.py --tag full500
.venv\Scripts\python rag_eval\eval_page_scoring.py --tag full500 --dev 250 --name page_scoring_500
```

Run the backend locally: `LOCAL_DEMO.md`. Evaluation walkthrough and the
rejected alternatives: `rag_eval/README.md`.
