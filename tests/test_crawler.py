# HuggingFace API(키워드/태그 검색)와 GitHub API(에이전트 프레임워크 수집)가 정상 동작하는지 확인
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.github_crawler import GitHubCrawler


async def test_huggingface():
    print("--- HuggingFace 크롤러 테스트 ---")
    crawler = HuggingFaceCrawler()

    print("\n[1] 키워드 검색 (DeepSeek)")
    results = await crawler.fetch_models(search_query="DeepSeek", limit=3)
    for r in results:
        print(f"  {r['name']} | downloads={r['downloads']:,} | pipeline={r['pipeline_tag']}")

    print("\n[2] 파이프라인 태그 필터 (text-to-image)")
    results = await crawler.fetch_models(search_query="text-to-image", limit=3)
    for r in results:
        print(f"  {r['name']} | pipeline={r['pipeline_tag']}")

    print("\n[3] 상위 LLM 수집 (top 5)")
    results = await crawler.fetch_top_models(limit=5)
    for r in results:
        print(f"  {r['name']} | downloads={r['downloads']:,}")

    print("\nHuggingFace 크롤러 테스트 성공!")


async def test_github():
    print("\n--- GitHub 크롤러 테스트 ---")
    crawler = GitHubCrawler()

    import crawler.github_crawler as _mod
    original_repos = _mod.AGENT_REPOS
    _mod.AGENT_REPOS = original_repos[:2]

    try:
        results = await crawler.fetch_agent_frameworks()
    finally:
        _mod.AGENT_REPOS = original_repos

    for r in results:
        print(f"  {r['name']} | stars={r['github_stars']:,} | {r['use_case']}")

    print("\nGitHub 크롤러 테스트 성공!")


if __name__ == "__main__":
    asyncio.run(test_huggingface())
    asyncio.run(test_github())
