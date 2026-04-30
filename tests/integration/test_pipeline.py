"""
전체 RAG 파이프라인 통합 테스트 — 모든 서비스(Ollama, ChromaDB, 임베딩, 리랭킹) 필요
실행: pytest tests/integration/test_pipeline.py -m integration -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app.pipeline.recommender import recommender_pipeline


pytestmark = pytest.mark.integration


class TestFullPipeline:

    async def test_model_query_returns_table(self):
        result = await recommender_pipeline.run("DeepSeek 모델의 최신 정보를 알려줘")
        assert result["category"] == "MODEL", f"분류 오류: {result['category']}"
        assert result["answer"], "answer 비어있음"
        assert len(result["table_data"]) > 0, "MODEL 결과 누락"
        assert result["table_data"][0].get("name"), "모델명 누락"

    async def test_agent_query_returns_table(self):
        result = await recommender_pipeline.run("자율 에이전트 프레임워크 추천해줘")
        assert result["category"] == "AGENT", f"분류 오류: {result['category']}"
        assert result["answer"], "answer 비어있음"
        assert len(result["table_data"]) > 0, "AGENT 결과 누락"
        assert result["table_data"][0].get("name"), "에이전트명 누락"

    async def test_general_query_returns_answer(self):
        result = await recommender_pipeline.run("안녕하세요")
        assert result["category"] == "GENERAL", f"분류 오류: {result['category']}"
        assert result["answer"], "GENERAL 답변 비어있음"
        assert result.get("references") == [], "GENERAL은 references 없음"
