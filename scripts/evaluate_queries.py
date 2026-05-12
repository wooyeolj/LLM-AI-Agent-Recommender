"""
test_queries.json 기반 분류기 평가 스크립트
query_classifier를 직접 호출해 분류 정확도와 응답속도를 측정
"""
import sys
import os
import json
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings
from app.services.query_classifier import query_classifier

QUERIES_PATH = os.path.join(settings.BASE_DIR, "tests", "test_queries.json")
RUNS = 5


async def evaluate():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        suite = json.load(f)

    queries = suite["queries"]

    print("====================")
    print(f"  쿼리 분류 테스트")
    print("====================")

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        expected_cat = q["expected_category"]
        accepted = expected_cat if isinstance(expected_cat, list) else [expected_cat]

        correct = 0
        elapsed_list = []
        actual_cats = []

        for _ in range(RUNS):
            start = time.time()
            actual_cat = await query_classifier.classify(query_text)
            elapsed_list.append(time.time() - start)
            actual_cats.append(actual_cat)
            if actual_cat in accepted:
                correct += 1

        avg_time = sum(elapsed_list) / len(elapsed_list)
        label = " 또는 ".join(accepted)
        actual_label = " ".join(actual_cats)
        status = "[PASS]" if correct == RUNS else "[FAIL]"

        print(f"  Q{qid:02d} | {query_text}")
        print(f"       기대: {label:<20} 실제: {actual_label:<8} 정확도: {correct}/{RUNS}  평균속도: {avg_time:.1f}s  {status}")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(evaluate())
