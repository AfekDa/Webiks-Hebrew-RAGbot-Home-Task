"""Check that the running Demo API really uses the improvement.

Takes held-out questions from the offline page-scoring run, sends them to the
live backend, and checks that the pages the API returns match the pages the
offline evaluation predicted for the *improved* system (top 3). It picks
questions where the improvement changed the top 3, plus one it did not, so the
check would fail both if page scoring were silently off and if it differed
from the evaluated rule.

Needs: Elasticsearch seeded (scripts/seed_demo.py), the backend running
(scripts/run_demo.py), and the offline run
(rag_eval/eval_page_scoring.py --tag <tag>) done on this machine.

    .venv/Scripts/python scripts/verify_demo.py --tag full500 --name page_scoring_500
"""
import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full500", help="cache folder of the offline run")
    parser.add_argument("--name", default=None, help="results folder to write (default: the tag)")
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--changed", type=int, default=4, help="how many changed-top-3 questions to check")
    args = parser.parse_args()

    offline = json.loads((ROOT / "rag_eval/cache" / args.tag / "page_scoring_results.json")
                         .read_text(encoding="utf-8"))
    records = offline.get("held_out_questions") or []
    if not records:
        raise ValueError("offline run has no held-out records; re-run rag_eval/eval_page_scoring.py")
    changed = [r for r in records if r["baseline_pages"][:3] != r["improved_pages"][:3]]
    unchanged = [r for r in records if r["baseline_pages"][:3] == r["improved_pages"][:3]]
    selected = changed[:args.changed] + unchanged[:1]

    health = requests.get(args.url + "/health", timeout=10)
    health.raise_for_status()
    checks = []
    for record in selected:
        response = requests.post(args.url + "/search", json={
            "query": record["question"], "asked_from": "local-verification",
        }, timeout=180)
        response.raise_for_status()
        body = response.json()
        actual = [str(doc["id"]) for doc in body["docs"]][:3]
        expected = [str(p) for p in record["improved_pages"][:3]]
        if actual != expected:
            raise AssertionError(f"Question {record['qi']}: API pages {actual} != offline pages {expected}")
        if body["metadata"]["tokens"] != 0 or not body["llm_result"].startswith("Mock answer"):
            raise AssertionError("Expected mock answer generation with zero token usage")
        check = {
            "qi": record["qi"], "status": response.status_code, "pages": actual,
            "matches_offline": True,
            "differs_from_baseline": actual != [str(p) for p in record["baseline_pages"][:3]],
            "correct_in_top3": any(p in record["accepted"] for p in actual),
            "retrieval_seconds": body["metadata"]["retrieval_time"], "llm_tokens": 0,
        }
        checks.append(check)
        print(json.dumps(check), flush=True)
    output = ROOT / "rag_eval/results" / (args.name or args.tag)
    output.mkdir(parents=True, exist_ok=True)
    (output / "demo_verification.json").write_text(json.dumps({
        "url": args.url, "health_status": health.status_code,
        "rule": offline.get("label"), "checks": checks,
    }, indent=2) + "\n")
    print(f"all {len(checks)} API checks match the offline improved ranking -> {output}/demo_verification.json")


if __name__ == "__main__":
    main()
