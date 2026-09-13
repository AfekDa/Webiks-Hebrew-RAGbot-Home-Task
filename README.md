# Hebrew RAG retrieval: page scoring

This repository contains the Webiks retrieval engine and Demo backend with one
retrieval improvement: **page scoring**. Instead of ranking a page by its single
best paragraph, the engine also counts how well the question matches the page
*title* and the page's *second-best* paragraph, using the same trained Hebrew
model. Only the order of the retrieved candidates changes; no new model, no
re-indexing.

On the full 24,487-paragraph corpus, measured on 250 held-out questions the
weights were never tuned on, the correct page lands at #1 for **54.8%** of
questions versus **38.8%** before (+16 points), MRR@10 0.547 → 0.650, with
top-5 unchanged.

- [SUBMISSION.md](SUBMISSION.md) — the improvement, why, how it was measured, results, limitations
- [LOCAL_DEMO.md](LOCAL_DEMO.md) — run the upgraded backend locally (no API key needed)
- [UPSTREAM.md](UPSTREAM.md) — data/model downloads and the Python environment
- [rag_eval/](rag_eval/) — the evaluation scripts and committed results

During development I also built and evaluated two other ideas the same way, and
rejected both because they lowered the first result on held-out questions: a
cross-encoder reranker (BGE) and hybrid dense + BM25 retrieval. Their code was
removed to keep the repo focused on the shipped answer; their measured numbers
stay in `rag_eval/results/` and the reasoning is in `SUBMISSION.md`.

Run the full evaluation from the repository root (GPU):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1
```

Large assets and intermediate embeddings are ignored by Git.
