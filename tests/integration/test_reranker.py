"""
리랭커 통합 테스트 — BGE-reranker-v2-m3 모델 로드 필요
실행: pytest tests/integration/test_reranker.py -m integration -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app.services.reranker import reranker


pytestmark = pytest.mark.integration


class TestReranker:

    def test_top_result_is_code_related(self):
        query = "코딩 잘하는 언어 모델 추천해줘."
        documents = [
            "하루에는 물은 2L 이상 마시면 좋습니다.",
            "CodeLlama는 프로그래밍 코딩에 강력한 LLM입니다.",
            "파이썬은 LLM 설계에 핵심적인 언어 입니다.",
            "GPT-4는 코드 프로그래밍 강력한 LLM입니다",
        ]
        metadatas = [{"name": f"doc_{i}"} for i in range(len(documents))]

        results = reranker.rerank(query, documents, metadatas, top_n=4)

        assert results, "리랭킹 결과 없음"
        assert len(results) == 4
        top_text = results[0]["text"]
        assert "CodeLlama" in top_text or "GPT-4" in top_text, "리랭킹 1위가 코딩 관련 문서가 아님"

    def test_scores_descending(self):
        query = "LLM 추천"
        documents = ["GPT-4 강력한 LLM", "물 2L 마시기"]
        metadatas = [{"name": "a"}, {"name": "b"}]

        results = reranker.rerank(query, documents, metadatas, top_n=2)
        assert results[0]["score"] >= results[1]["score"], "점수가 내림차순이 아님"

    def test_empty_documents_returns_empty(self):
        results = reranker.rerank("테스트", [], [], top_n=3)
        assert results == []

    def test_top_n_limit(self):
        query = "테스트"
        documents = ["문서 A", "문서 B", "문서 C", "문서 D"]
        metadatas = [{"name": f"doc_{i}"} for i in range(4)]

        results = reranker.rerank(query, documents, metadatas, top_n=2)
        assert len(results) == 2
