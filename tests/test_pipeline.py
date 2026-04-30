"""전체 파이프라인 통합 테스트 (백엔드 서버 불필요 — 직접 호출)"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.recommender import recommender_pipeline


async def test_full_pipeline():
    print("--- 통합 추천 파이프라인 테스트 ---")

    cases = [
        ("DeepSeek 모델의 최신 정보를 알려줘", "MODEL"),
        ("자율 에이전트 프레임워크 추천해줘", "AGENT"),
        ("안녕하세요", "GENERAL"),
    ]

    for query, expected_category in cases:
        print(f"\n[질문] {query}")
        result = await recommender_pipeline.run(query)

        assert isinstance(result, dict), f"반환값이 dict가 아님: {type(result)}"
        assert "category" in result, "category 키 없음"
        assert "answer" in result, "answer 키 없음"
        assert "references" in result, "references 키 없음"
        assert "table_data" in result, "table_data 키 없음"
        assert result["category"] == expected_category, (
            f"카테고리 오류: 실제={result['category']}, 기대={expected_category}"
        )
        assert result["answer"], "answer가 비어있음"

        if expected_category in ("MODEL", "AGENT"):
            assert isinstance(result["table_data"], list), "table_data가 리스트가 아님"

        print(f"  카테고리: {result['category']} ✅")
        print(f"  답변 앞부분: {result['answer'][:80]}...")
        print(f"  참고 항목: {len(result['references'])}개")
        if result["table_data"]:
            print(f"  추천 항목: {len(result['table_data'])}개")

    print("\n✅ 파이프라인 통합 테스트 성공!")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
