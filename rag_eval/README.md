# Retrieval evaluation

Run from the repository root after following `UPSTREAM.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1
```

The runbook requires a working CUDA device and stops immediately if a step fails.
The individual Python scripts can also run on CPU, but take substantially longer.

## What it measures

Every script reproduces the real engine's retrieval offline in NumPy: embed the
question with the same model, cosine against every paragraph, take the top 50,
collapse to unique pages. That mirrors the engine's Elasticsearch `script_score`
query and page selection; small floating-point or tie-order differences remain
possible between implementations.

**Metrics.** An accepted answer is a page (`doc_id`); a question can have several.
Hit@k is one when any accepted page is among the first k returned pages. MRR@10
is the reciprocal rank of the first accepted page, or zero outside the first
ten. These are retrieval proxies; they do not evaluate generated answers.

**Discipline.** Every improvement is tuned on the first half of the questions
(dev) and reported once on the untouched second half (held-out), and shipped only
if held-out Hit@1 or MRR improves while Hit@5 does not regress. The QA data was
used to train the embedder, so none of this is a true out-of-domain estimate,
but the split does stop us from fooling ourselves with a hand-picked setting.

## Steps in the runbook

1. `build_subset.py --questions 200 --pages 25000 --tag full` selects 200 questions
   (seed 42) and embeds the complete 24,487-paragraph corpus (7,007 pages), in
   resumable chunks. Despite its name, `--pages` is a paragraph count.
2. `eval_baseline.py --tag full` measures today's system and saves each question's
   top-50 candidates.
3. **Headline.** `build_subset.py --questions 500 --tag full500 --seed 1729 --reuse-paragraph-cache full
   --exclude-questions-from full` builds a 500-question set with no overlap with
   the 200, reusing the same embeddings (validated, not recomputed). Then
   `eval_baseline.py --tag full500` and
   `eval_page_scoring.py --tag full500 --dev 250 --name page_scoring_500`
   pick the page-scoring weights on 250 questions and confirm on the other 250.
   The rule (`webiks_hebrew_ragbot/page_order.py`) is the exact code the engine runs.

A cross-encoder reranker was evaluated during development and rejected. Its
evaluation code and result files were removed so this folder carries only the
shipped answer; the numbers and the reasoning are in `SUBMISSION.md` section 4.

## Committed results (`rag_eval/results/`)

| folder | what |
|---|---|
| `page_scoring_500/` | **the submitted improvement**: 500 fresh questions, 250 dev / 250 held-out |
| `page_scoring_200/` | earlier pilot of the same idea on the first 200 questions (100/100) |

## Resume and outputs

Embedding chunks live under `rag_eval/cache/<tag>/` and resume on re-run. The
full corpus, model files, embeddings, and cache stay out of Git.

## Check the live API

After seeding and launching the Demo as in `LOCAL_DEMO.md`:

```powershell
.venv/Scripts/python scripts/verify_demo.py --tag full500 --name page_scoring_500
```

It sends held-out questions to the running backend and checks the returned pages
match the offline *improved* ranking, including questions where page scoring
changed the top 3.
