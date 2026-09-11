# Upstream sources

This private repository is a home-task working copy built on top of Webiks'
open-source Hebrew RAG projects. The two vendored project folders below were
copied pristine from the following upstream repositories, then modified for the
assignment. Their original licenses (`LICENSE.txt` inside each folder) apply.

| Folder | Upstream repo | Commit vendored |
|---|---|---|
| `Webiks-Hebrew-RAGbot-Demo/` | https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-Demo | `7278cfdab5173e9d36747b5ea2ab325586a37e75` |
| `Webiks-Hebrew-RAGbot/` | https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot | `8dd6c53706cf1fb0a4ae19bb2b14b052df645f3b` |

The **first commit** in this repo is those two folders exactly as vendored, so
that every later commit's diff shows precisely what was changed for the task.

## Large assets (not in git — download separately)
These are needed to run things but are excluded from the repo (see `.gitignore`):
- Paragraph corpus JSON — https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-Paragraph-Corpus (Google Drive link inside)
- QA dataset CSV — https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-QA-Training-DataSet (Google Drive link inside)
- Retrieval model `Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0` — https://drive.google.com/file/d/1i_7bTdGWC7yUVC_NLDQGk63kPRhZT7y3/view
