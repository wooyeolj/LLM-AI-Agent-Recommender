import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.reranker import reranker


def test_run():
    print("--- 리랭커 테스트 ---")

    query = "코딩 성능이 뛰어난 대형 언어 모델을 추천해줘."
    documents = [
        "이 모델은 요리 레시피를 전문적으로 알려주는 AI 에이전트입니다.",
        "CodeLlama는 프로그래밍 코드 생성과 디버깅에 최적화된 강력한 LLM입니다.",
        "오늘 점심 메뉴로는 비빔밥을 추천합니다.",
        "GPT-4는 수학 문제 풀이와 코딩을 포함한 다양한 작업에서 높은 성능을 보입니다.",
    ]
    # reranker.rerank()는 metadatas를 함께 받음
    metadatas = [{"name": f"doc_{i}"} for i in range(len(documents))]

    print(f"질문: {query}\n")
    results = reranker.rerank(query, documents, metadatas, top_n=4)

    for i, r in enumerate(results):
        print(f"[{i+1}위] 점수: {r['score']:.4f}")
        print(f"  문구: {r['text']}\n")

    top_text = results[0]["text"]
    if "CodeLlama" in top_text or "GPT-4" in top_text:
        print("✅ 리랭커 테스트 성공! 관련 문장이 상단에 위치합니다.")
    else:
        print("❌ 리랭킹 결과가 예상과 다릅니다.")


if __name__ == "__main__":
    test_run()
