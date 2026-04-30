
# QueryClassifier 단위 테스트 — ollama_client 모킹으로 Ollama 서버 없이 실행

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, patch
from app.core.types import ItemType


class TestClassifyByKeyword:

    async def test_model_keyword_gpt(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()
        result = await clf.classify("GPT 모델 추천해줘")
        assert result == ItemType.MODEL

    async def test_model_keyword_llm(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()
        result = await clf.classify("좋은 LLM이 뭐야?")
        assert result == ItemType.MODEL

    async def test_agent_keyword_crewai(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()
        result = await clf.classify("crewai로 자동화 만들고 싶어")
        assert result == ItemType.AGENT

    async def test_agent_keyword_korean(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()
        result = await clf.classify("자율 에이전트 시스템 필요해")
        assert result == ItemType.AGENT


class TestClassifyWithLLMFallback:

    async def test_llm_returns_model(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            mock_llm.classify_query = AsyncMock(return_value="MODEL")
            result = await clf.classify("그림 잘 그리는 AI 추천해줘")

        assert result == ItemType.MODEL

    async def test_llm_returns_agent(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            mock_llm.classify_query = AsyncMock(return_value="AGENT")
            result = await clf.classify("매일 자동으로 작업해주는 시스템")

        assert result == ItemType.AGENT

    async def test_llm_returns_general(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            mock_llm.classify_query = AsyncMock(return_value="GENERAL")
            result = await clf.classify("오늘 날씨 어때?")

        assert result == ItemType.GENERAL

    async def test_llm_verbose_response_parsed_correctly(self):
        # LLM의 첫 번째 클래스 키워드만 추출
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            mock_llm.classify_query = AsyncMock(
                return_value="입력 문장은 MODEL 카테고리에 해당합니다."
            )
            result = await clf.classify("임베딩 모델 알려줘")

        assert result == ItemType.MODEL

    async def test_llm_negation_not_misclassified(self):
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            # LLM이 GENERAL을 첫 단어로 반환
            mock_llm.classify_query = AsyncMock(return_value="GENERAL")
            result = await clf.classify("파이썬 기초 알려줘")

        assert result == ItemType.GENERAL

    async def test_ollama_failure_returns_general(self):
        # Ollama 장애 시 GENERAL로 fallback
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            mock_llm.classify_query = AsyncMock(side_effect=Exception("연결 실패"))
            result = await clf.classify("애매한 질문")

        assert result == ItemType.GENERAL

    async def test_llm_unrecognized_response_returns_general(self):
        # LLM이 세 클래스 외 이상한 응답을 줄 때 → GENERAL
        from app.services.query_classifier import QueryClassifier
        clf = QueryClassifier()

        with patch("app.services.query_classifier.ollama_client") as mock_llm:
            mock_llm.classify_query = AsyncMock(return_value="잘 모르겠습니다")
            result = await clf.classify("이상한 질문")

        assert result == ItemType.GENERAL


class TestItemTypeCompatibility:
    def test_model_equals_string(self):
        assert ItemType.MODEL == "MODEL"

    def test_agent_equals_string(self):
        assert ItemType.AGENT == "AGENT"

    def test_general_equals_string(self):
        assert ItemType.GENERAL == "GENERAL"

    def test_in_operator_with_string_tuple(self):
        assert ItemType.MODEL in ("MODEL", "AGENT")

    def test_str_conversion(self):
        assert str(ItemType.MODEL) == "MODEL"
