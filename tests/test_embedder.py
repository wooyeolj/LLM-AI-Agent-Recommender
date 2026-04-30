import sys
import os

# 프로젝트 루트 경로를 인식하게 합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedder import embedder

def test_run():
    print("--- 임베딩 테스트 시작 ---")
    
    test_text = "AI 에이전트와 LLM 모델의 차이점은 무엇인가요?"
    
    # 1. 벡터 변환
    vector = embedder.get_embedding(test_text)
    
    # 2. 결과 확인
    print(f"입력 문장: {test_text}")
    print(f"벡터 차원 수: {len(vector)}") # BGE-M3는 보통 1024차원입니다.
    print(f"앞부분 데이터 (5개): {vector[:5]}")
    
    if len(vector) > 0:
        print("\n✅ 임베딩 테스트 성공!")
    else:
        print("\n❌ 임베딩 실패!")

if __name__ == "__main__":
    test_run()