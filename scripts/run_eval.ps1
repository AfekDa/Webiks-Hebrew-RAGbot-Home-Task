$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$evalPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $evalPython)) { throw 'Create .venv first; see UPSTREAM.md.' }
& $evalPython -c "import torch; assert torch.cuda.is_available(), 'GPU required for this runbook'; print(torch.__version__, torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
& $evalPython -u rag_eval/build_subset.py --questions 200 --pages 25000 --tag full
if ($LASTEXITCODE -ne 0) { throw 'Corpus embedding failed.' }
& $evalPython -u rag_eval/eval_baseline.py --tag full
if ($LASTEXITCODE -ne 0) { throw 'Baseline evaluation failed.' }
& $evalPython -u rag_eval/eval_reranker.py --tag full --n-questions 200 --top-rerank 50 --dtype float16 --mode blend --blend-k 5
if ($LASTEXITCODE -ne 0) { throw 'Reranker evaluation failed.' }
& $evalPython -u rag_eval/summarize_results.py --tag full
if ($LASTEXITCODE -ne 0) { throw 'Result verification/export failed.' }
