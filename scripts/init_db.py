"""
초기 DB 구축 스크립트 (최초 1회 실행)

수집 항목:
  - HuggingFace API  : 오픈소스 모델 이름, 설명, 다운로드 수, 좋아요, 태그, 출시일
  - OpenRouter API   : 모델 카탈로그 + 가격
  - GitHub API       : AI 에이전트 프레임워크 별점, 업데이트일, 지원 LLM

실행: python scripts/init_db.py
"""
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.openrouter_crawler import pricing_crawler
from crawler.github_crawler import github_crawler
from crawler.data_processor import data_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("###  LLM & Agent 추천 시스템 초기 DB 구축 ###")

    # 1. 외부 데이터 로드
    logger.info("[1/4] OpenRouter 가격 데이터 로드 중...")
    pricing_data = await pricing_crawler.load_openrouter_prices()

    # 2. HuggingFace 오픈소스 모델 수집 + 가격 JOIN
    logger.info("[2/4] HuggingFace 상위 모델 수집 중...")
    hf = HuggingFaceCrawler()
    hf_models = await hf.fetch_top_models(limit=50)
    logger.info("      수집된 모델: %d개", len(hf_models))

    for m in hf_models:
        price_info = pricing_crawler.get_price(pricing_data, m["name"])
        m["cost"] = price_info["price_display"]

    await data_processor.process_and_save(hf_models, item_type="MODEL")

    # 3. OpenRouter 모델 카탈로그 수집
    logger.info("[3/4] OpenRouter 모델 카탈로그 수집 중...")
    or_models = await pricing_crawler.fetch_models()
    logger.info("      수집된 모델: %d개", len(or_models))
    paid_count = sum(1 for m in or_models if "무료" not in m.get("cost", ""))
    logger.info("      유료 모델: %d개 / 무료: %d개", paid_count, len(or_models) - paid_count)
    await data_processor.process_and_save(or_models, item_type="MODEL")

    # 4. AI 에이전트 프레임워크 수집
    logger.info("[4/4] GitHub 에이전트 프레임워크 수집 중...")
    agents = await github_crawler.fetch_agent_frameworks()
    logger.info("      수집된 에이전트: %d개", len(agents))
    await data_processor.process_and_save(agents, item_type="AGENT")

    total_models = len(hf_models) + len(or_models)
    logger.info("###  초기 DB 구축 완료 ###")
    logger.info("모델: %d개 (HF %d + OpenRouter %d) | 에이전트: %d개",
                total_models, len(hf_models), len(or_models), len(agents))


if __name__ == "__main__":
    asyncio.run(main())
