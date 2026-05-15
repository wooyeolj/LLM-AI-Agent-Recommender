# 리랭커(BGE-reranker-v2-m3)가 쿼리와 관련 높은 문장을 상위로 정렬하는지 확인
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.reranker import reranker


def test_run():
    print("--- 리랭커 테스트 ---")

    query = "코딩 잘하는 언어 모델 추천해줘."
    documents = [
        "하루에는 물은 2L 이상 마시면 좋습니다.",
        "CodeLlama는 프로그래밍 코딩에 강력한 LLM입니다.",
        "파이썬은 LLM 설계에 핵심적인 언어 입니다.",
        "GPT-4는 코드 프로그래밍 강력한 LLM입니다",
    ]
    metadatas = [{"name": f"doc_{i}"} for i in range(len(documents))]

    print(f"질문: {query}\n")
    results = reranker.rerank(query, documents, metadatas, top_n=4)

    for i, r in enumerate(results):
        print(f"[{i+1}위] 점수: {r['score']:.4f}")
        print(f"  문구: {r['text']}\n")

    top_text = results[0]["text"]
    assert "CodeLlama" in top_text or "GPT-4" in top_text, "리랭킹 순위 오류!"
    print("리랭커 테스트 성공!")


if __name__ == "__main__":
    test_run()
