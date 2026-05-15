# 임베딩 모델(BGE-m3-ko)이 텍스트를 벡터로 정상 변환하는지 확인
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedder import embedder


def test_run():
    print("--- 임베딩 테스트 ---")

    text = "AI 에이전트와 LLM 모델의 차이점은 뭐야?"
    vector = embedder.get_embedding(text)

    print(f"입력: {text}")
    print(f"차원: {len(vector)} | 앞부분: {vector[:3]}")

    assert len(vector) > 0, "임베딩 벡터 공백!"
    print("임베딩 테스트 성공!")


if __name__ == "__main__":
    test_run()
