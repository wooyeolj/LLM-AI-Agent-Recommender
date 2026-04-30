"""
test_queries.json 기반 파이프라인 평가 스크립트
실행 전 백엔드가 실행 중이어야 합니다: python3 app/main.py
"""
import sys
import os
import json
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

BASE_URL = f"http://localhost:{settings.BACKEND_PORT}"
QUERIES_PATH = os.path.join(settings.BASE_DIR, "data", "test_queries.json")


def call_chat(query: str, timeout: int = 300) -> dict:
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": query}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def evaluate():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        suite = json.load(f)

    queries = suite["queries"]
    results = []

    print("=" * 60)
    print(f"  평가 대상: {suite.get('description', QUERIES_PATH)}")
    print(f"  쿼리 수: {len(queries)}개")
    print(f"  백엔드: {BASE_URL}")
    print("=" * 60)

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        expected_cat = q["expected_category"]  # str 또는 list
        expect_table = q["expect_table"]
        accepted = expected_cat if isinstance(expected_cat, list) else [expected_cat]
        label = " 또는 ".join(accepted)

        print(f"\n[Q{qid}] {query_text}")
        print(f"  기대 카테고리: {label}")

        start = time.time()
        try:
            data = call_chat(query_text)
            elapsed = time.time() - start

            actual_cat = data.get("category", "")
            table_data = data.get("table_data") or []

            cat_ok = actual_cat in accepted
            table_ok = (len(table_data) > 0) == expect_table

            status = "PASS" if (cat_ok and table_ok) else "FAIL"

            print(f"  실제 카테고리: {actual_cat}  {'✅' if cat_ok else f'❌ (기대: {label})'}")
            print(f"  테이블 존재: {'있음' if table_data else '없음'} (기대: {'있음' if expect_table else '없음'})  {'✅' if table_ok else '❌'}")
            print(f"  응답 시간: {elapsed:.1f}s")

            if table_data:
                print(f"  추천 항목 수: {len(table_data)}개")
                for item in table_data[:3]:
                    name = item.get("name", "?")
                    desc = (item.get("description") or item.get("use_case") or "")[:50]
                    print(f"    - {name}: {desc}")

            print(f"  → {status}")

            results.append({
                "id": qid,
                "query": query_text,
                "expected_category": expected_cat,
                "actual_category": actual_cat,
                "cat_pass": cat_ok,
                "table_pass": table_ok,
                "elapsed": round(elapsed, 1),
                "item_count": len(table_data),
                "status": status,
            })

        except requests.exceptions.ConnectionError:
            print(f"  ❌ 백엔드 연결 실패 — python3 app/main.py 를 먼저 실행하세요")
            sys.exit(1)
        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            print(f"  ❌ 타임아웃 ({elapsed:.0f}s 초과)")
            results.append({"id": qid, "query": query_text, "status": "TIMEOUT", "elapsed": round(elapsed, 1)})
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            results.append({"id": qid, "query": query_text, "status": "ERROR", "error": str(e)})

    print("\n" + "=" * 60)
    print("  평가 결과 요약")
    print("=" * 60)

    passed = [r for r in results if r.get("status") == "PASS"]
    failed = [r for r in results if r.get("status") == "FAIL"]
    errors = [r for r in results if r.get("status") in ("TIMEOUT", "ERROR")]

    cat_correct = sum(1 for r in results if r.get("cat_pass"))
    table_correct = sum(1 for r in results if r.get("table_pass"))
    total_evaluated = len([r for r in results if "cat_pass" in r])

    print(f"\n  카테고리 정확도: {cat_correct}/{total_evaluated}  ({cat_correct/total_evaluated*100:.0f}%)" if total_evaluated else "")
    print(f"  테이블 정확도:   {table_correct}/{total_evaluated}  ({table_correct/total_evaluated*100:.0f}%)" if total_evaluated else "")
    print(f"\n  PASS:    {len(passed)}개")
    print(f"  FAIL:    {len(failed)}개")
    print(f"  오류:    {len(errors)}개")

    if results:
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        print(f"  평균 응답시간: {avg_time:.1f}s")

    print()
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        exp = r.get("expected_category", "?")
        exp_label = " 또는 ".join(exp) if isinstance(exp, list) else exp
        cat_info = f"{r.get('actual_category', '?')} (기대: {exp_label})"
        print(f"  {icon} Q{r['id']}: {r['status']:<7}  카테고리={cat_info}  {r['elapsed']}s")

    print("=" * 60)

    if failed:
        print("\n  실패 쿼리 분석:")
        for r in failed:
            issues = []
            if not r.get("cat_pass"):
                issues.append(f"카테고리 오분류 ({r.get('actual_category')} ≠ {r.get('expected_category')})")
            if not r.get("table_pass"):
                issues.append(f"테이블 유무 불일치 (항목 수: {r.get('item_count', 0)})")
            print(f"  - Q{r['id']}: {', '.join(issues)}")

    return len(passed) == len(queries)


if __name__ == "__main__":
    success = evaluate()
    sys.exit(0 if success else 1)
