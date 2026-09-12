# Handoff — run this on the faster (GPU) PC

This is a runbook to continue the Hebrew-RAG reranker task on a better machine.
The current PC has **no GPU**, so the heavy steps take hours; on a GPU the whole
thing is a few minutes. **Accuracy results are identical on CPU or GPU — only the
speed changes.**

---

## 1. Where things stand right now

**Done**
- Evaluation harness built (`rag_eval/`): reproduces the real search offline.
- Reranker step built (`rag_eval/eval_reranker.py`) + integrated into the engine
  (optional, off by default — see section 5).
- Baseline + reranker measured on an **easy 2,000-page** test.

**Key result so far (the honest one):** on the easy 2,000-page test the reranker
**slightly hurt** the numbers:

| correct page is… | before | after |
|---|---|---|
| at #1 | 75.0% | 72.0% |
| top-3 | 91.0% | 90.0% |
| top-5 | 94.0% | 93.0% |
| top-10 | 97.0% | 97.0% |
| MRR@10 | 0.836 | 0.814 |

**Why:** the search model was trained on these exact questions, so on an *easy*
corpus it is already near-perfect and a general reranker can't add value. The fix
is a **harder, more realistic test** (many more pages), where the base search is
weaker and the reranker has room to help — that is exactly what to run next.

**In progress (won't transfer — redo on the new PC):** a harder **10,000-page**
run was started here but the cache is local + gitignored, so just re-run it on the
new machine (fast on a GPU).

---

## 2. One-time setup on the new PC

1. **Get the code:** clone the repo (private):
   `https://github.com/AfekDa/Webiks-Hebrew-RAGbot-Home-Task`

2. **Get the big files** (not in git — see `UPSTREAM.md` for the download links).
   Put them in the project root:
   - `Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json` (the pages)
   - `Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv` (the answer key)
   - `Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/` (the Hebrew model folder)

3. **Make a Python environment** (Python 3.11) and install the libraries.
   **Important for GPU:** install the CUDA build of PyTorch, not the CPU one.
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install --upgrade pip
   # GPU PyTorch (pick the CUDA version your PC has; cu121 shown):
   .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
   .venv\Scripts\python -m pip install "sentence-transformers==3.0.1" "transformers==4.42.3" "numpy<2"
   ```
   Check the GPU is seen: `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"` → should print `True`.
   (The scripts auto-use the GPU when it's available — no code change needed.)

The reranker model (**BAAI/bge-reranker-v2-m3**, ~2.2 GB) downloads automatically
the first time you run step 3 below. On a good connection this is quick.

---

## 3. Run the evaluation (the main thing)

All commands run from the project root. This is the **harder, realistic test** —
the one that should show the reranker helping.

```
# STEP 1 — build the test set + study the pages (the big one; minutes on a GPU).
#          Use the FULL corpus for the most convincing, realistic result.
.venv\Scripts\python rag_eval\build_subset.py --questions 200 --pages 25000 --tag full

# STEP 2 — baseline (search only). Also saves each question's candidates.
.venv\Scripts\python rag_eval\eval_baseline.py --tag full

# STEP 3 — reranker before/after on the SAME questions. On a GPU you can rerank
#          all 50 candidates (top-rerank 50); the script prints a before/after table.
.venv\Scripts\python rag_eval\eval_reranker.py --tag full --n-questions 200 --top-rerank 50
```

- `--pages 25000` = use essentially the whole ~24k-paragraph corpus (the realistic,
  hard setting where Webiks measured ~36% at #1). Smaller numbers = easier/faster.
- Step 3 prints the **before vs after** table and saves it to
  `rag_eval\cache\full\reranker_results.json`.
- Everything is **resumable**: if a run stops, just run the same command again and
  it continues where it left off (step 1 saves in chunks; step 3 saves per question).

**What to look for:** on this harder test, "before" (search only) should be much
lower than the easy run (closer to Webiks' ~36% at #1), and "after" (+ reranker)
should be **higher than before**. That gap is the improvement to report.

If it still doesn't help, two honest fallbacks are ready to try: a **hybrid score**
(blend the reranker with the original score) or **report it as a negative result**
with the reasoning.

---

## 4. Fill in the write-up

Once step 3 prints the numbers, put them into the results table in
`SUBMISSION.md` (section 4) and adjust the surrounding sentences. The rest of that
document (problem, approach, method, limitations, reproduce) is already written.

---

## 5. Turn the reranker on in the live demo (optional)

The reranker is wired into the engine but **off by default**. To switch it on for
the demo backend, add these to the backend `.env`:

```
RERANK_ENABLED=TRUE
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP=50          # on a GPU you can rerank all candidates; use 20 on CPU
RERANK_MAX_SEQ=512
```

With `RERANK_ENABLED=FALSE` (the default) the system behaves exactly as the
original — the reranker adds zero cost when off.

---

## 6. File map (what's what)

| File | What it does |
|---|---|
| `rag_eval/common.py` | Shared logic; mirrors the real `Engine.search_documents`. |
| `rag_eval/build_subset.py` | STEP 1 — pick questions + pages, study (embed) them. Resumable. |
| `rag_eval/eval_baseline.py` | STEP 2 — measure search-only; save candidates. |
| `rag_eval/eval_reranker.py` | STEP 3 — rerank the candidates, print before/after. Resumable. |
| `rag_eval/README.md` | Plain-language walkthrough of the harness. |
| `Webiks-Hebrew-RAGbot/webiks_hebrew_ragbot/reranker.py` | The reranker used inside the engine. |
| `Webiks-Hebrew-RAGbot/webiks_hebrew_ragbot/engine.py` | Reranker hooked into `search_documents` (optional). |
| `Webiks-Hebrew-RAGbot/webiks_hebrew_ragbot/config.py` | `RERANK_*` switches. |
| `SUBMISSION.md` | The 1–2 page write-up (fill in the results table). |
| `NOTES.md` | Full working notes / background. |

---

## 7. Time estimate: CPU (this PC) vs GPU (new PC)

| Step | This CPU | GPU |
|---|---|---|
| Study full corpus (~24k pages) | ~19 hours | ~5–10 min |
| Reranker before/after | ~2 hours | under a minute |
| Live demo, per question | ~1–2 min | instant |
