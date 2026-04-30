"""HuggingFace + GitHub 크롤러 테스트"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.github_crawler import GitHubCrawler


async def test_huggingface():
    print("--- HuggingFace 크롤러 테스트 ---")
    crawler = HuggingFaceCrawler()

    # 1. 키워드 검색
    print("\n[1] 키워드 검색 (DeepSeek)")
    results = await crawler.fetch_models(search_query="DeepSeek", limit=3)
    assert isinstance(results, list), "반환값이 리스트가 아님"
    for r in results:
        assert "name" in r and "downloads" in r and "pipeline_tag" in r, f"필드 누락: {r.keys()}"
        print(f"  {r['name']} | downloads={r['downloads']:,} | pipeline={r['pipeline_tag']}")

    # 2. 파이프라인 태그 필터 (pipeline_tag 파라미터 경로)
    print("\n[2] 파이프라인 태그 필터 (text-to-image)")
    results = await crawler.fetch_models(search_query="text-to-image", limit=3)
    assert isinstance(results, list), "반환값이 리스트가 아님"
    for r in results:
        print(f"  {r['name']} | pipeline={r['pipeline_tag']}")
        assert r["pipeline_tag"] == "text-to-image", f"pipeline_tag 불일치: {r['pipeline_tag']}"

    # 3. 초기 DB용 상위 모델
    print("\n[3] 상위 LLM 수집 (top 5)")
    results = await crawler.fetch_top_models(limit=5)
    assert isinstance(results, list) and len(results) > 0, "결과가 비어있음"
    for r in results:
        print(f"  {r['name']} | downloads={r['downloads']:,}")

    print("\n✅ HuggingFace 크롤러 테스트 완료")


async def test_github():
    print("\n--- GitHub 크롤러 테스트 ---")
    crawler = GitHubCrawler()

    # 전체 목록 중 2개만 테스트 (API 호출 절약)
    from crawler.github_crawler import AGENT_REPOS
    original = crawler.__class__.__dict__.get("fetch_agent_frameworks")

    # 실제로는 첫 2개 repo만 사용하도록 임시 제한
    import crawler.github_crawler as _mod
    original_repos = _mod.AGENT_REPOS
    _mod.AGENT_REPOS = original_repos[:2]

    try:
        results = await crawler.fetch_agent_frameworks()
    finally:
        _mod.AGENT_REPOS = original_repos

    assert isinstance(results, list) and len(results) > 0, "결과가 비어있음"

    required = {"name", "url", "github_stars", "use_case", "difficulty", "supported_llms", "local_support"}
    for r in results:
        missing = required - r.keys()
        assert not missing, f"필드 누락: {missing}"
        print(f"  {r['name']} | ⭐{r['github_stars']:,} | {r['use_case']}")

    print("\n✅ GitHub 크롤러 테스트 완료")


if __name__ == "__main__":
    asyncio.run(test_huggingface())
    asyncio.run(test_github())
