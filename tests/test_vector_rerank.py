# ChromaDB 벡터 검색 후 리랭커가 순위를 올바르게 재정렬하는지 확인
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import vector_store
from app.services.reranker import reranker


async def test_combined_search():
    print("--- 벡터 검색 + 리랭킹 테스트 ---")

    ids = ["test_model_1", "test_model_2", "test_agent_1"]
    docs = [
        "Gemma 3는 Google이 개발한 오픈 모델로 경량화된 구조와 강력한 한국어 성능을 자랑합니다.",
        "Llama 3는 Meta에서 공개한 오픈소스 LLM이며 코딩과 추론 성능이 뛰어납니다.",
        "AutoGPT는 목표를 설정하면 스스로 검색하고 작업을 수행하는 자율형 에이전트입니다.",
    ]
    metas = [
        {"name": "Gemma 3", "type": "MODEL"},
        {"name": "Llama 3", "type": "MODEL"},
        {"name": "AutoGPT", "type": "AGENT"},
    ]

    await vector_store.upsert_documents(ids, docs, metas, item_type="MODEL")

    query = "페이스북에서 만든 모델이 뭐야?"
    print(f"질문: {query}\n")

    raw = await vector_store.query(query, n_results=3, item_type="MODEL")
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]

    print("[1차 벡터 검색]")
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        print(f"  {i+1}. {meta.get('name')}: {doc[:50]}...")

    print("\n[2차 리랭킹]")
    reranked = reranker.rerank(query, documents, metadatas, top_n=3)
    for i, r in enumerate(reranked):
        print(f"  {i+1}. {r['metadata'].get('name')} (점수: {r['score']:.4f})")

    assert "Llama 3" in reranked[0]["metadata"].get("name", ""), "리랭킹 순위 오류!"
    print("\n테스트 성공!")


if __name__ == "__main__":
    asyncio.run(test_combined_search())
