# Retrieval evaluation

Run from the repository root after following `UPSTREAM.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1
```

The runbook requires a working CUDA device and stops immediately if a step fails.
The individual Python scripts can also run on CPU, but take substantially longer.

## Steps

1. `build_subset.py --questions 200 --pages 25000 --tag full` selects 200 questions
   with seed 42 and embeds the complete 24,487-paragraph corpus (7,007 pages).
   Despite its name, `--pages` specifies a paragraph count. Smaller runs include
   all accepted pages' paragraphs plus sampled distractor paragraphs.
2. `eval_baseline.py --tag full` embeds the questions, computes cosine similarity
   against every paragraph, selects 50 paragraphs, and collapses them to unique
   page IDs in first-seen order. This mirrors the engine's Elasticsearch
   `script_score` query and page selection using NumPy. Small floating-point or
   tie-order differences remain possible between implementations.
3. `eval_reranker.py --tag full --n-questions 200 --top-rerank 50 --dtype float16` reranks those
   exact candidates with `BAAI/bge-reranker-v2-m3`, then applies the same page
   deduplication. The reranker uses CUDA float16 and raw logits; the embedder
   remains float32. Both models use a 512-token limit. Paragraph order is the only
   retrieval change; no query rewriting, new candidates, or score blending.
4. `summarize_results.py --tag full` verifies that per-question metrics reproduce
   the saved summaries and exports results to `rag_eval/results/full/`. It also
   computes paired bootstrap intervals, top-1 wins/losses, candidate recall, and
   measured reranking latency.

## Metrics

An accepted answer is a page (`doc_id`), and a question can have several accepted
pages. Hit@k is one when any accepted page is among the first k returned pages.
MRR@10 is the reciprocal rank of the first accepted page, or zero outside the
first ten. These are retrieval proxies; they do not evaluate generated answers
or prove that the selected paragraph contains the answer.

The QA data was used in development/training of the embedder. This paired
comparison is not a held-out estimate of generalization. Results from the
upstream model's validation artifact use a different question selection and
should not be compared directly with this sample.

## Resume and outputs

Embedding chunks and reranker records are saved under `rag_eval/cache/<tag>/`.
Re-run with the same parameters to resume. Use a fresh tag when changing the
embedding selection or model settings. Reranker checkpoints reject changed
candidate data, model path, precision, score transform, token limit, candidate
count, or question count. Float16 inference can change close rankings; CPU and
GPU results are not assumed to be numerically identical.

Reviewable outputs include summaries, question IDs/accepted pages, per-question
rankings and scores, confidence intervals, and a manifest with cache hashes.
The full corpus, model files, embeddings, and temporary cache stay out of Git.

After indexing and launching the Demo as described in `LOCAL_DEMO.md`, run:

```powershell
.venv/Scripts/python scripts/verify_demo.py --tag full
```

This queries the real HTTP API and compares returned page IDs to offline results,
including cases where reranking changed the first three pages.
