"""Export reviewable results, paired uncertainty, and reproducibility metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
from scipy.stats import binomtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full", help="cache folder to read (rag_eval/cache/<tag>)")
    parser.add_argument("--name", default=None,
                        help="results folder to write (rag_eval/results/<name>); defaults to the tag")
    args = parser.parse_args()
    cache = Path("rag_eval/cache") / args.tag
    output = Path("rag_eval/results") / (args.name or args.tag)
    result = json.loads((cache / "reranker_results.json").read_text())
    candidates = json.loads((cache / "candidates.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (cache / "rerank_progress.jsonl").read_text().splitlines() if line]
    records.sort(key=lambda record: record["qi"])
    n = result["n_questions"]
    if [r["qi"] for r in records] != list(range(n)) or n != len(candidates):
        raise ValueError("Only a complete run on all cached questions can be exported")
    before = np.array([r["b"] for r in records], dtype=float)
    after = np.array([r["r"] for r in records], dtype=float)
    keys = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]
    original = json.loads((cache / "baseline_results.json").read_text())
    for i, key in enumerate(keys):
        if not np.isclose(before[:, i].mean(), original[key]):
            raise ValueError(f"Recomputed baseline does not match saved baseline: {key}")
        if not np.isclose(after[:, i].mean(), result[key]):
            raise ValueError(f"Per-question results disagree with summary: {key}")

    rng = np.random.default_rng(42)
    draws = rng.integers(0, n, size=(10000, n))
    bootstraps = (after - before)[draws].mean(axis=1)
    improved = int(((after[:, 0] == 1) & (before[:, 0] == 0)).sum())
    regressed = int(((after[:, 0] == 0) & (before[:, 0] == 1)).sum())
    discordant = improved + regressed
    analysis = {
        "n_questions": n,
        "hit1_improved_questions": improved,
        "hit1_regressed_questions": regressed,
        "hit1_unchanged_questions": n - discordant,
        "hit1_exact_mcnemar_p": binomtest(improved, discordant).pvalue if discordant else 1.0,
        "paired_bootstrap": {
            "seed": 42, "resamples": 10000, "confidence": 0.95,
            "delta_intervals": {
                key: np.quantile(bootstraps[:, i], [0.025, 0.975]).tolist()
                for i, key in enumerate(keys)
            },
        },
        "candidate_hit_rate": sum(
            bool(set(r["baseline_pages"]) & set(c["accepted"]))
            for r, c in zip(records, candidates)
        ) / n,
        "rerank_latency_seconds": {
            "mean": float(np.mean([r["rerank_seconds"] for r in records])),
            "median": float(np.median([r["rerank_seconds"] for r in records])),
            "p95": float(np.quantile([r["rerank_seconds"] for r in records], 0.95)),
        },
    }
    manifest = {
        "python": platform.python_version(),
        "build_config": json.loads((cache / "config.json").read_text()),
        "rerank_config": json.loads((cache / "rerank_config.json").read_text()),
        "n_pages": len({p["doc_id"] for p in json.loads((cache / "paras.json").read_text(encoding="utf-8"))}),
        "sha256": {},
    }
    for name in ["paras.json", "questions.json", "candidates.json", "para_emb.npy"]:
        with (cache / name).open("rb") as file:
            manifest["sha256"][name] = hashlib.file_digest(file, "sha256").hexdigest()
    for name, path in {
        "embedder_weights": Path("Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/model.safetensors"),
        "reranker_weights": Path(result["model"]) / "model.safetensors",
    }.items():
        if path.is_file():
            with path.open("rb") as file:
                manifest["sha256"][name] = hashlib.file_digest(file, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for name in ["baseline_results.json", "reranker_results.json", "rerank_progress.jsonl", "questions.json"]:
        shutil.copyfile(cache / name, output / name)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))
    print(f"Exported to {output}")


if __name__ == "__main__":
    main()
