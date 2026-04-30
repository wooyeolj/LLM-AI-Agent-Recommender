import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.query_classifier import query_classifier

async def test_classification():
    print("--- 쿼리 분류기 테스트 시작 ---")
    
    test_queries = [
        "최신 코딩용 LLM 모델 추천해줘",
        "인터넷을 스스로 검색해서 보고서를 쓰는 에이전트가 있어?",
        "안녕? 오늘 날씨 어때?"
    ]
    
    for q in test_queries:
        category = await query_classifier.classify(q)
        print(f"질문: {q} -> 분류 결과: [{category}]")

if __name__ == "__main__":
    asyncio.run(test_classification())