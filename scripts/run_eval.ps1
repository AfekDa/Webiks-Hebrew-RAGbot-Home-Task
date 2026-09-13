$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$evalPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $evalPython)) { throw 'Create .venv first; see UPSTREAM.md.' }
& $evalPython -c "import torch; assert torch.cuda.is_available(), 'GPU required for this runbook'; print(torch.__version__, torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }

# 1. Embed the full corpus once (tag "full", 200 questions; minutes on a GPU, cached after).
& $evalPython -u rag_eval/build_subset.py --questions 200 --pages 25000 --tag full
if ($LASTEXITCODE -ne 0) { throw 'Corpus embedding failed.' }
& $evalPython -u rag_eval/eval_baseline.py --tag full
if ($LASTEXITCODE -ne 0) { throw 'Baseline evaluation failed.' }

# 2. THE HEADLINE: page scoring on 500 fresh questions (none of the 200 above),
#    reusing the same embeddings. Settings are picked on 250 questions and
#    confirmed on the other 250. Exported to rag_eval/results/page_scoring_500.
if (-not (Test-Path -LiteralPath 'rag_eval/cache/full500/para_emb.npy')) {
  & $evalPython -u rag_eval/build_subset.py --questions 500 --pages 25000 --tag full500 --seed 1729 --reuse-paragraph-cache full --exclude-questions-from full
  if ($LASTEXITCODE -ne 0) { throw 'Building the 500-question set failed.' }
}
& $evalPython -u rag_eval/eval_baseline.py --tag full500
if ($LASTEXITCODE -ne 0) { throw 'Baseline (500) evaluation failed.' }
& $evalPython -u rag_eval/eval_page_scoring.py --tag full500 --dev 250 --name page_scoring_500
if ($LASTEXITCODE -ne 0) { throw 'Page-scoring evaluation failed.' }
