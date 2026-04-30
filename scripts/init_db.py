"""
초기 DB 구축 스크립트 (최초 1회 실행)

수집 소스:
  - HuggingFace API  : 모델 이름, 설명, 다운로드 수, 좋아요, 태그, 출시일
  - OpenRouter API   : 가격 정보, 컨텍스트 길이 (인증 불필요)
  - GitHub API       : AI 에이전트 프레임워크 별점, 업데이트일, 지원 LLM

실행: python scripts/init_db.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.pricing_crawler import pricing_crawler
from crawler.github_crawler import github_crawler
from crawler.data_processor import data_processor


async def main():
    print("=" * 50)
    print("  LLM & Agent 추천 시스템 초기 DB 구축")
    print("=" * 50)

    # ── 1. 외부 데이터 로드 ──────────────────────────
    print("\n[1/3] OpenRouter 가격 데이터 로드 중...")
    pricing_data = await pricing_crawler.load_openrouter_prices()

    # ── 2. LLM 모델 수집 + 가격 JOIN ────────────────
    print("\n[2/3] HuggingFace 상위 모델 수집 중...")
    hf = HuggingFaceCrawler()
    models = await hf.fetch_top_models(limit=50)
    print(f"      수집된 모델: {len(models)}개")

    for m in models:
        price_info = pricing_crawler.get_price(pricing_data, m["name"])
        m["cost"] = price_info["price_display"]
        m["context_length"] = price_info["context_length"]

    paid_count = sum(1 for m in models if "무료" not in m.get("cost", "무료"))
    print(f"      유료 모델: {paid_count}개 / 무료(오픈소스): {len(models) - paid_count}개")

    await data_processor.process_and_save(models, item_type="MODEL")

    # ── 3. AI 에이전트 프레임워크 수집 ──────────────
    print("\n[3/3] GitHub 에이전트 프레임워크 수집 중...")
    agents = await github_crawler.fetch_agent_frameworks()
    print(f"      수집된 에이전트: {len(agents)}개")
    await data_processor.process_and_save(agents, item_type="AGENT")

    print("\n" + "=" * 50)
    print("  ✅ 초기 DB 구축 완료!")
    print(f"     모델: {len(models)}개 | 에이전트: {len(agents)}개")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
