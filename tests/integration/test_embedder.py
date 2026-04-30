"""
임베딩 모델 통합 테스트 — BGE-m3-ko 모델 로드 필요
실행: pytest tests/integration/test_embedder.py -m integration -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app.services.embedder import embedder


pytestmark = pytest.mark.integration


class TestEmbedder:

    def test_single_embedding_dimension(self):
        text = "AI 에이전트와 LLM 모델의 차이점은 뭐야?"
        vector = embedder.get_embedding(text)
        assert len(vector) > 0, "임베딩 벡터 공백"
        assert isinstance(vector[0], float)

    def test_cache_hit_returns_same_vector(self):
        text = "캐시 테스트용 고유 문자열 xyz123"
        v1 = embedder.get_embedding(text)
        v2 = embedder.get_embedding(text)
        assert v1 == v2, "동일 입력에 캐시가 다른 벡터를 반환함"

    def test_batch_embeddings_length_matches(self):
        texts = ["모델 A 설명", "에이전트 B 설명", "일반 질문"]
        vectors = embedder.get_embeddings(texts)
        assert len(vectors) == len(texts)
        for v in vectors:
            assert len(v) > 0
