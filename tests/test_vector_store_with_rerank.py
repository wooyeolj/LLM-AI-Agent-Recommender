import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import vector_store
from app.services.reranker import reranker


async def test_combined_search():
    print("--- Vector DB + Reranker 통합 테스트 ---")

    sample_ids = ["test_model_1", "test_model_2", "test_agent_1"]
    sample_docs = [
        "Gemma 3는 Google에서 개발한 오픈 모델로, 경량화된 구조와 강력한 한국어 성능을 자랑합니다.",
        "Llama 3는 Meta(구 Facebook)에서 공개한 오픈 소스 LLM이며, 코딩과 추론 성능이 뛰어납니다.",
        "AutoGPT는 목표를 설정하면 스스로 인터넷을 검색하고 작업을 수행하는 자율형 에이전트입니다.",
    ]
    sample_metas = [
        {"name": "Gemma 3", "type": "MODEL"},
        {"name": "Llama 3", "type": "MODEL"},
        {"name": "AutoGPT", "type": "AGENT"},
    ]

    print("[1] 테스트 데이터 저장 중...")
    await vector_store.upsert_documents(sample_ids, sample_docs, sample_metas, item_type="MODEL")

    query = "페이스북에서 만든 모델이 뭐야?"
    print(f"\n[2] 질문: {query}")

    raw = await vector_store.query(query, n_results=3, item_type="MODEL")
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]

    print("\n[1차 벡터 검색 결과]")
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        print(f"  {i+1}. {meta.get('name', '?')}: {doc[:60]}...")

    print("\n[2차 리랭킹 결과]")
    reranked = reranker.rerank(query, documents, metadatas, top_n=3)
    for i, r in enumerate(reranked):
        print(f"  {i+1}. {r['metadata'].get('name', '?')} (점수: {r['score']:.4f})")

    if "Llama 3" in reranked[0]["metadata"].get("name", ""):
        print("\n✅ 성공! Meta/Facebook → Llama 3를 1위로 올렸습니다.")
    else:
        print("\n⚠️ 예상과 다른 순위 (모델 성능 차이일 수 있음)")


if __name__ == "__main__":
    asyncio.run(test_combined_search())
