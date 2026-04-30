import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ollama_client import ollama_client
from app.core.config import settings


async def test_run():
    print(f"--- Ollama({settings.OLLAMA_MODEL}) 연동 테스트 ---")

    query = "Llama 3 모델은 누가 만들었어?"
    context = "Llama 3는 Meta(구 Facebook)에서 개발한 고성능 오픈 소스 거대 언어 모델입니다."

    print(f"질문: {query}")
    print("답변 생성 중...")

    response = await ollama_client.generate_response(query=query, context=context)

    print("\n" + "=" * 50)
    print(f"답변:\n{response}")
    print("=" * 50)
    print("\n✅ Ollama 테스트 완료" if response and "실패" not in response else "\n❌ Ollama 연결 실패")


if __name__ == "__main__":
    asyncio.run(test_run())
