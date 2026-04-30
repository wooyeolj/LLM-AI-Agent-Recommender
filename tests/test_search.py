import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import vector_store


async def test_search():
    print("--- Vector DB 검색 테스트 ---")

    for item_type, query in [
        ("MODEL", "최신 Llama 모델 알려줘"),
        ("AGENT", "자율 에이전트 프레임워크 추천"),
    ]:
        print(f"\n[{item_type}] 질문: {query}")
        results = await vector_store.query(query_text=query, n_results=3, item_type=item_type)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            print("  결과 없음 (DB가 비어있을 수 있습니다. init_db.py를 먼저 실행하세요)")
            continue

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            print(f"  {i+1}. {meta.get('name', '?')} (유사도: {1 - dist:.4f})")

    print("\n✅ 검색 테스트 완료")


if __name__ == "__main__":
    asyncio.run(test_search())
