# 전체 RAG 파이프라인(분류→크롤링→벡터검색→리랭킹→LLM 생성)이 3가지 케이스(MODEL/AGENT/GENERAL)에서 정상 동작하는지 확인
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.recommender import recommender_pipeline


async def test_full_pipeline():
    print("--- 파이프라인 통합 테스트 ---")

    cases = [
        ("DeepSeek 모델의 최신 정보를 알려줘", "MODEL"),
        ("자율 에이전트 프레임워크 추천해줘", "AGENT"),
        ("안녕하세요", "GENERAL"),
    ]

    for query, expected in cases:
        print(f"\n질문: {query}")
        result = await recommender_pipeline.run(query)

        assert result["category"] == expected, f"분류 오류: 실제={result['category']}, 기대={expected}"
        assert result["answer"], "answer 비어있음"

        # MODEL/AGENT는 추천 결과가 반드시 존재
        if expected != "GENERAL":
            assert len(result["table_data"]) > 0, f"{expected} 결과 누락 — 검색/리랭킹 파이프라인 실패"
            assert result["table_data"][0].get("name"), f"{expected} 항목명 누락"

        print(f"  카테고리: {result['category']}")
        print(f"  답변: {result['answer'][:80]}...")
        if result["table_data"]:
            print(f"  추천 항목: {len(result['table_data'])}개")

    print("\n파이프라인 통합 테스트 성공!")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
