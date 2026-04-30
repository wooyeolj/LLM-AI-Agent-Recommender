"""
수동 크롤링 트리거 스크립트

특정 키워드로 HuggingFace를 검색하고 DB를 즉시 갱신
실행: python scripts/run_crawler.py [검색어]
예:  python scripts/run_crawler.py DeepSeek
     python scripts/run_crawler.py text-to-image
"""
import logging
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.openrouter_crawler import pricing_crawler
from crawler.data_processor import data_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run(search_query: str = None):
    logger.info("수동 크롤링 시작 (검색어: %s)", search_query or "text-generation 상위")

    pricing_data = await pricing_crawler.load_openrouter_prices()

    crawler = HuggingFaceCrawler()
    raw_data = await crawler.fetch_models(search_query=search_query, limit=20)

    if not raw_data:
        logger.warning("수집 결과 없음")
        return

    for m in raw_data:
        price_info = pricing_crawler.get_price(pricing_data, m["name"])
        m["cost"] = price_info["price_display"]

    logger.info("%d개 모델 수집 완료. DB 저장 중...", len(raw_data))
    await data_processor.process_and_save(raw_data, item_type="MODEL")
    logger.info("완료")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(query))
