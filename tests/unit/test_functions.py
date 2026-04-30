
# 순수 함수 단위 테스트

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest


# recommender.py

class TestNormalizeConfidence:

    def setup_method(self):
        from app.pipeline.recommender import normalize_confidence
        self.fn = normalize_confidence

    def test_empty(self):
        assert self.fn([]) == []

    def test_single_result(self):
        result = self.fn([0.85])
        assert result == ["단일 결과"]

    def test_all_same_scores(self):
        result = self.fn([0.5, 0.5, 0.5])
        assert result == ["동일", "동일", "동일"]

    def test_normal_range(self):
        result = self.fn([0.0, 0.5, 1.0])
        assert result == ["0.0%", "50.0%", "100.0%"]

    def test_order_preserved(self):
        # 높은 점수가 높은 % 를 가져야 함
        result = self.fn([1.0, 0.0])
        assert result[0] == "100.0%"
        assert result[1] == "0.0%"

    def test_negative_scores(self):
        result = self.fn([-2.0, 0.0, 2.0])
        assert result[0] == "0.0%"
        assert result[2] == "100.0%"


class TestFmtDownloads:

    def setup_method(self):
        from app.pipeline.recommender import fmt_downloads
        self.fn = fmt_downloads

    def test_millions(self):
        assert self.fn(1_500_000) == "1.5M"

    def test_thousands(self):
        assert self.fn(3_500) == "4K"

    def test_small(self):
        assert self.fn(42) == "42"

    def test_exactly_one_million(self):
        assert self.fn(1_000_000) == "1.0M"

    def test_exactly_one_thousand(self):
        assert self.fn(1_000) == "1K"

    def test_zero(self):
        assert self.fn(0) == "0"


class TestSafeInt:

    def setup_method(self):
        from app.pipeline.recommender import safe_int
        self.fn = safe_int

    def test_normal_int(self):
        assert self.fn(42) == 42

    def test_string_int(self):
        assert self.fn("100") == 100

    def test_none(self):
        assert self.fn(None) == 0

    def test_empty_string(self):
        assert self.fn("") == 0

    def test_invalid_string(self):
        assert self.fn("abc") == 0

    def test_float_string(self):
        assert self.fn("3.5") == 0

    def test_zero_string(self):
        assert self.fn("0") == 0


class TestBuildModelTable:

    def setup_method(self):
        from app.pipeline.recommender import build_model_table
        self.fn = build_model_table

    def _make_result(self, name="TestModel", downloads=1000, likes=50, source="hf"):
        return {
            "text": f"모델명: {name}",
            "score": 0.9,
            "metadata": {
                "name": name,
                "description": "테스트 모델",
                "cost": "무료 (오픈소스)",
                "created_at": "2024-01-15T00:00:00",
                "downloads": downloads,
                "likes": likes,
                "url": f"https://huggingface.co/{name}",
                "source": source,
            },
        }

    def test_basic_row(self):
        rows = self.fn([self._make_result()], ["75.0%"])
        assert len(rows) == 1
        assert rows[0]["name"] == "TestModel"
        assert rows[0]["relevance"] == "75.0%"

    def test_downloads_formatted(self):
        rows = self.fn([self._make_result(downloads=2_500_000)], ["N/A"])
        assert rows[0]["downloads"] == "2.5M"

    def test_created_at_truncated(self):
        # "2024-01-15T00:00:00" → "2024-01"
        rows = self.fn([self._make_result()], ["50.0%"])
        assert rows[0]["created_at"] == "2024-01"

    def test_missing_created_at(self):
        result = self._make_result()
        result["metadata"]["created_at"] = None
        rows = self.fn([result], ["N/A"])
        assert rows[0]["created_at"] == "N/A"

    def test_invalid_downloads_type(self):
        result = self._make_result()
        result["metadata"]["downloads"] = "not-a-number"
        rows = self.fn([result], ["N/A"])
        assert rows[0]["downloads"] == "0"

    def test_openrouter_downloads_and_likes_are_dash(self):
        # OpenRouter 모델은 HF 지표 없음 → "-" 표시
        result = self._make_result(source="openrouter", downloads=0, likes=0)
        rows = self.fn([result], ["50.0%"])
        assert rows[0]["downloads"] == "-"
        assert rows[0]["likes"] == "-"

    def test_hf_source_shows_numeric(self):
        rows = self.fn([self._make_result(source="hf", downloads=5000, likes=42)], ["50.0%"])
        assert rows[0]["downloads"] == "5K"
        assert rows[0]["likes"] == 42


# query_classifier.py
class TestClassifyByKeyword:

    def setup_method(self):
        from app.services.query_classifier import classify_by_keyword
        self.fn = classify_by_keyword

    def test_model_keyword(self):
        result = self.fn("llama 모델 추천해줘")
        assert str(result) == "MODEL"

    def test_agent_keyword(self):
        result = self.fn("crewai로 자동화 만들고 싶어")
        assert str(result) == "AGENT"

    def test_tie_returns_none(self):
        # general
        result = self.fn("오늘 날씨 어때?")
        assert result is None

    def test_model_wins(self):
        result = self.fn("llama llm rag 추천")
        assert str(result) == "MODEL"

    def test_agent_wins(self):
        result = self.fn("에이전트 agent 자율 autonomous")
        assert str(result) == "AGENT"


# openrouter_crawler.py

class TestFormatPrice:

    def setup_method(self):
        from crawler.openrouter_crawler import format_price
        self.fn = format_price

    def test_paid_model(self):
        result = self.fn("0.000003", "0.000015")
        assert "입력" in result
        assert "출력" in result
        assert "1M" in result

    def test_free_model(self):
        result = self.fn("0", "0")
        assert result == "무료"

    def test_invalid_input(self):
        result = self.fn("invalid", "data")
        assert result == "가격 정보 없음"

    def test_none_input(self):
        result = self.fn(None, None)
        assert result == "가격 정보 없음"


class TestNormalizeModelId:

    def setup_method(self):
        from crawler.openrouter_crawler import normalize
        self.fn = normalize

    def test_lowercase(self):
        assert self.fn("GPT-4") == "gpt-4"

    def test_strip_whitespace(self):
        assert self.fn("  llama  ") == "llama"

    def test_combined(self):
        assert self.fn("  OpenAI/GPT-4  ") == "openai/gpt-4"


# github_crawler.py

class TestInferLlms:
    def setup_method(self):
        from crawler.github_crawler import infer_llms
        self.fn = infer_llms

    def test_openai_detected(self):
        result = self.fn(["openai"], "")
        assert "GPT" in result

    def test_anthropic_detected(self):
        result = self.fn([], "uses anthropic claude")
        assert "Claude" in result

    def test_ollama_detected(self):
        result = self.fn(["ollama"], "")
        assert "Ollama(로컬)" in result

    def test_multiple_llms(self):
        result = self.fn(["openai", "ollama"], "uses google gemini")
        assert "GPT" in result
        assert "Ollama(로컬)" in result
        assert "Gemini" in result

    def test_none_detected_returns_default(self):
        result = self.fn([], "just a framework")
        assert result == "GPT, Claude, Ollama(로컬)"


class TestInferLocal:

    def setup_method(self):
        from crawler.github_crawler import infer_local
        self.fn = infer_local

    def test_ollama_tag(self):
        assert self.fn(["ollama"], "") is True

    def test_local_keyword_in_desc(self):
        assert self.fn([], "supports local deployment") is True

    def test_llama_cpp(self):
        assert self.fn([], "built on llama.cpp") is True

    def test_no_local_support(self):
        assert self.fn(["cloud", "saas"], "hosted only") is False
