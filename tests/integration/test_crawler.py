"""
크롤러 통합 테스트 — HuggingFace API 및 GitHub API 실제 연결 필요
실행: pytest tests/integration/test_crawler.py -m integration -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.github_crawler import GitHubCrawler
import crawler.github_crawler as _github_mod


pytestmark = pytest.mark.integration


class TestHuggingFaceCrawler:

    async def test_keyword_search(self):
        crawler = HuggingFaceCrawler()
        results = await crawler.fetch_models(search_query="DeepSeek", limit=3)
        assert results, "DeepSeek 키워드 검색 결과 누락"
        for r in results:
            assert r.get("name"), "모델명 누락"
            assert r.get("type") == "MODEL"

    async def test_pipeline_tag_filter(self):
        crawler = HuggingFaceCrawler()
        results = await crawler.fetch_models(search_query="text-to-image", limit=3)
        assert results, "text-to-image 파이프라인 태그 검색 결과 누락"
        for r in results:
            assert r.get("name"), "모델명 누락"

    async def test_top_models(self):
        crawler = HuggingFaceCrawler()
        results = await crawler.fetch_top_models(limit=5)
        assert results, "상위 LLM 수집 결과 누락"
        assert len(results) <= 5
        for r in results:
            assert r.get("name"), "모델명 누락"
            assert isinstance(r.get("downloads", 0), int)


class TestGitHubCrawler:

    async def test_agent_frameworks(self):
        crawler = GitHubCrawler()
        original_repos = _github_mod.AGENT_REPOS
        _github_mod.AGENT_REPOS = original_repos[:2]
        try:
            results = await crawler.fetch_agent_frameworks()
        finally:
            _github_mod.AGENT_REPOS = original_repos

        assert results, "GitHub 에이전트 프레임워크 수집 결과 누락"
        for r in results:
            assert r.get("name"), "프레임워크명 누락"
            assert r.get("type") == "AGENT"
            assert isinstance(r.get("github_stars", 0), int)
