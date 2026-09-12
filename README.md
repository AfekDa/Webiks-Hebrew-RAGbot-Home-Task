# Hebrew RAG retrieval reranker

This repository contains the Webiks retrieval engine and Demo backend, with an
optional BGE cross-encoder between top-50 paragraph retrieval and page selection.

- [Submission: approach, evaluation, and limitations](SUBMISSION.md)
- [Local backend setup and API commands](LOCAL_DEMO.md)
- [Data/model downloads and Python environment](UPSTREAM.md)
- [Evaluation scripts](rag_eval/)

The backend installs the modified engine from this checkout. Its local launcher
uses real Elasticsearch retrieval and a mock answer generator, so no API key is
needed. Invalid reranker settings and missing paragraph fields raise errors.

Run the full GPU evaluation from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1
```

Large assets and intermediate embeddings are ignored by Git. Verified evaluation
results are exported for review under `rag_eval/results/`: `blend/` is the
headline run (reranker merged with the search order, the engine default) and
`replace/` is the reranker-only run it is compared against.
