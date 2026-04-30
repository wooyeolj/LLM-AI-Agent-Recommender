"""
벡터 검색 + 리랭킹 통합 테스트 — ChromaDB 및 임베딩/리랭킹 모델 로드 필요
실행: pytest tests/integration/test_vector_rerank.py -m integration -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app.services.vector_store import vector_store
from app.services.reranker import reranker


pytestmark = pytest.mark.integration

_IDS = ["integ_model_1", "integ_model_2", "integ_agent_1"]
_DOCS = [
    "Gemma 3는 Google이 개발한 오픈 모델로 경량화된 구조와 강력한 한국어 성능을 자랑합니다.",
    "Llama 3는 Meta에서 공개한 오픈소스 LLM이며 코딩과 추론 성능이 뛰어납니다.",
    "AutoGPT는 목표를 설정하면 스스로 검색하고 작업을 수행하는 자율형 에이전트입니다.",
]
_METAS = [
    {"name": "Gemma 3", "type": "MODEL", "source": "hf"},
    {"name": "Llama 3", "type": "MODEL", "source": "hf"},
    {"name": "AutoGPT", "type": "AGENT"},
]


@pytest.fixture(autouse=True)
async def setup_test_data():
    await vector_store.upsert_documents(_IDS, _DOCS, _METAS, item_type="MODEL")


class TestVectorSearchAndRerank:

    async def test_upsert_and_query(self):
        query = "페이스북에서 만든 모델이 뭐야?"
        raw = await vector_store.query(query, n_results=3, item_type="MODEL")
        assert raw["documents"][0], "벡터 검색 결과 없음"

    async def test_rerank_top_is_llama(self):
        query = "페이스북에서 만든 모델이 뭐야?"
        raw = await vector_store.query(query, n_results=3, item_type="MODEL")
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]

        import asyncio
        reranked = await asyncio.to_thread(reranker.rerank, query, documents, metadatas, 3)
        assert reranked, "리랭킹 결과 없음"
        assert "Llama 3" in reranked[0]["metadata"].get("name", ""), "리랭킹 1위가 Llama 3가 아님"
