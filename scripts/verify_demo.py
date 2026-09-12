"""Compare real HTTP search responses against the saved offline reranker run."""
import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full", help="cache folder to read (rag_eval/cache/<tag>)")
    parser.add_argument("--name", default=None,
                        help="results folder to write (rag_eval/results/<name>); defaults to the tag")
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    cache = ROOT / "rag_eval/cache" / args.tag
    questions = json.loads((cache / "questions.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (cache / "rerank_progress.jsonl").read_text().splitlines() if line]
    changed = [r for r in records if r["baseline_pages"][:3] != r["reranked_pages"][:3]]
    selected = {r["qi"]: r for r in changed[:4] + records[:1]}
    if not selected:
        raise ValueError("No completed reranking records")
    health = requests.get(args.url + "/health", timeout=10)
    health.raise_for_status()
    checks = []
    for qi, record in selected.items():
        response = requests.post(args.url + "/search", json={
            "query": questions[qi]["question"], "asked_from": "local-verification",
        }, timeout=180)
        response.raise_for_status()
        body = response.json()
        actual = [str(doc["id"]) for doc in body["docs"]]
        expected = record["reranked_pages"][:3]
        if actual != expected:
            raise AssertionError(f"Question {qi}: API pages {actual} != offline pages {expected}")
        if body["metadata"]["tokens"] != 0 or not body["llm_result"].startswith("Mock answer"):
            raise AssertionError("Expected mock answer generation with zero token usage")
        check = {
            "qi": qi, "status": response.status_code, "pages": actual,
            "matches_offline": True,
            "differs_from_baseline": actual != record["baseline_pages"][:3],
            "retrieval_seconds": body["metadata"]["retrieval_time"], "llm_tokens": 0,
        }
        checks.append(check)
        print(json.dumps(check), flush=True)
    output = ROOT / "rag_eval/results" / (args.name or args.tag)
    output.mkdir(parents=True, exist_ok=True)
    (output / "demo_verification.json").write_text(json.dumps({
        "url": args.url, "health_status": health.status_code, "checks": checks,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
