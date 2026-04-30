"""
test_queries.json 기반 분류기 평가
query_classifier를 직접 호출해 분류 정확도와 응답속도를 측정
"""
import sys
import os
import json
import time
import asyncio
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings
from app.services.query_classifier import query_classifier

QUERIES_PATH = os.path.join(settings.BASE_DIR, "scripts", "test_queries.json")
RESULT_PATH  = os.path.join(settings.BASE_DIR, "scripts", "evaluate_queries_result.json")
RUNS = 5


def bucket(accepted: list[str]) -> str:
    # 다중 정답은 별도 분리
    return accepted[0] if len(accepted) == 1 else "AMBIGUOUS"


async def evaluate():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        suite = json.load(f)

    queries = suite["queries"]

    print(f"쿼리 분류 평가 시작 (model={settings.OLLAMA_MODEL}, 쿼리 {len(queries)}개 × {RUNS}회)")

    total_correct = 0
    total_runs = 0
    per_query_results = []
    bucket_stats: dict[str, list[int]] = {}  # bucket → [correct, runs]

    for q in queries:
        qid           = q["id"]
        query_text    = q["query"]
        expected_cat  = q["expected_category"]
        accepted      = expected_cat if isinstance(expected_cat, list) else [expected_cat]

        correct       = 0
        elapsed_list  = []
        predictions   = []

        for _ in range(RUNS):
            start      = time.time()
            actual_cat = str(await query_classifier.classify(query_text))
            elapsed_list.append(round(time.time() - start, 3))
            predictions.append(actual_cat)
            if actual_cat in accepted:
                correct += 1

        total_correct += correct
        total_runs    += RUNS

        b = bucket(accepted)
        bucket_stats.setdefault(b, [0, 0])
        bucket_stats[b][0] += correct
        bucket_stats[b][1] += RUNS

        avg_time = sum(elapsed_list) / len(elapsed_list)
        status   = "PASS" if correct == RUNS else ("부분" if correct > 0 else "FAIL")

        if status == "PASS":
            print(f"Q{qid:02d}  {correct}/{RUNS}  {avg_time:.2f}s  PASS")
        else:
            wrong = [p for p in predictions if p not in accepted]
            wrong_summary = ", ".join(f"{c}×{n}" for c, n in Counter(wrong).items())
            label = "/".join(accepted)
            print(f"Q{qid:02d}  {correct}/{RUNS}  {avg_time:.2f}s  {status}   기대={label}, 오분류={wrong_summary}")

        per_query_results.append({
            "id": qid,
            "query": query_text,
            "expected_category": expected_cat,
            "accepted": accepted,
            "runs": RUNS,
            "correct": correct,
            "accuracy": round(correct / RUNS, 3),
            "avg_time_sec": round(avg_time, 3),
            "status": status,
            "predictions": predictions,
        })

    pct = total_correct / total_runs * 100 if total_runs else 0.0

    print(f"\n[종합] 정확도 {pct:.1f}%  ({total_correct}/{total_runs})")
    for b in ("MODEL", "AGENT", "GENERAL", "AMBIGUOUS"):
        if b in bucket_stats:
            c, n = bucket_stats[b]
            print(f"  {b:<10} {c/n*100:5.1f}%  ({c}/{n})")

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "fallback_model": settings.OLLAMA_MODEL,
        "runs_per_query": RUNS,
        "note": (
            "test_queries.json 기반 분류기 평가. "
            "LLM의 비결정적 특성으로 인해 재실행 시 분류 결과가 달라질 수 있다. "
            "버킷 AMBIGUOUS는 expected_category가 다중인 쿼리 의미."
        ),
        "summary": {
            "total_queries": len(per_query_results),
            "total_runs":    total_runs,
            "total_correct": total_correct,
            "accuracy_pct":  round(pct, 1),
            "by_bucket": {
                b: {
                    "correct": c,
                    "runs": n,
                    "accuracy_pct": round(c / n * 100, 1),
                }
                for b, (c, n) in bucket_stats.items()
            },
        },
        "queries": per_query_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {os.path.relpath(RESULT_PATH, settings.BASE_DIR)}")


if __name__ == "__main__":
    asyncio.run(evaluate())
