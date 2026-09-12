# Upstream code and assets

This repository contains the [Webiks retrieval engine](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot)
and [Demo backend](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-Demo), with the
reranker integrated into the local engine. Install the local package; installing
the published `webiks-hebrew-ragbot==1.0.1` omits this change.

Download these assets to the repository root, and extract the model ZIP there:

| Asset | Download |
|---|---|
| `Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json` | [Paragraph corpus](https://drive.google.com/file/d/1w4d6O5pnFAVPExb9SzcaCXRL8qjT9blB/view) |
| `Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv` | [QA dataset](https://drive.google.com/file/d/1YaQ8ZbpqfBzZvxZZQv01tlSSnVnFBgMz/view) |
| `Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/` | [Model ZIP](https://drive.google.com/file/d/1i_7bTdGWC7yUVC_NLDQGk63kPRhZT7y3/view) |

Source descriptions: [corpus](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-Paragraph-Corpus),
[QA data](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-QA-Training-DataSet).
The corpus contains paragraphs, not one record per page. The QA dataset was used
to train the embedder, so evaluation on it is not a held-out generalization test.

The reranker downloads from [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
The September 2026 run uses revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

## Windows GPU environment

Run from the repository root with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
Push-Location Webiks-Hebrew-RAGbot-Demo
..\.venv\Scripts\python -m pip install -r requirements.txt
Pop-Location
.venv\Scripts\python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

This session installed Python 3.11 using `python -m uv venv --python 3.11 .venv`.
The CUDA wheel includes its CUDA runtime; a separate CUDA toolkit is unnecessary.
See `LOCAL_DEMO.md` for backend launch instructions.
