# RAG 파이프라인 핵심 — 분류 → 벡터 검색 → 크롤링 → 리랭킹 → LLM 생성 순서로 실행
import asyncio
import logging
import os
import re
import json
from datetime import datetime, timedelta
from typing import AsyncGenerator

from app.services.query_classifier import query_classifier
from app.services.vector_store import vector_store
from app.services.reranker import reranker
from app.services.ollama_client import ollama_client
from crawler.huggingface_crawler import hf_crawler
from crawler.github_crawler import github_crawler
from crawler.openrouter_crawler import pricing_crawler
from crawler.data_processor import data_processor
from app.core.config import settings
from app.core.types import ItemType

logger = logging.getLogger(__name__)

TTL_DAYS = settings.CACHE_TTL_DAYS
CRAWL_CACHE_FILE = os.path.join(settings.BASE_DIR, "data", "crawl_cache.json")
PRICING_CACHE_FILE = os.path.join(settings.BASE_DIR, "data", "pricing_cache.json")

# 리랭커 점수 임계값 - 점수 미만은 차단해 LLM 환각 방지.
RELEVANCE_THRESHOLD = -1.0

pricing_lock = asyncio.Lock()
crawl_lock = asyncio.Lock()


def load_crawl_cache() -> dict[str, datetime]:
    if not os.path.exists(CRAWL_CACHE_FILE):
        return {}
    try:
        with open(CRAWL_CACHE_FILE, "r", encoding="utf-8") as f:
            return {k: datetime.fromisoformat(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def save_crawl_cache():
    os.makedirs(os.path.dirname(CRAWL_CACHE_FILE), exist_ok=True)
    tmp = CRAWL_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: v.isoformat() for k, v in crawl_cache.items()}, f)
    os.replace(tmp, CRAWL_CACHE_FILE)


crawl_cache: dict[str, datetime] = load_crawl_cache()


def is_recently_crawled(keyword: str) -> bool:
    if keyword not in crawl_cache:
        return False
    return datetime.now() - crawl_cache[keyword] < timedelta(days=TTL_DAYS)


async def mark_crawled(keyword: str):
    async with crawl_lock:
        crawl_cache[keyword] = datetime.now()
        await asyncio.to_thread(save_crawl_cache)


def load_pricing_cache() -> tuple[dict | None, datetime | None]:
    if not os.path.exists(PRICING_CACHE_FILE):
        return None, None
    try:
        with open(PRICING_CACHE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_at = datetime.fromisoformat(saved["saved_at"])
        if datetime.now() - saved_at > timedelta(days=TTL_DAYS):
            return None, None 
        return saved["data"], saved_at #data 키를 읽음
    except Exception:
        return None, None


def save_pricing_cache(data: dict):
    os.makedirs(os.path.dirname(PRICING_CACHE_FILE), exist_ok=True)
    tmp = PRICING_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"saved_at": datetime.now().isoformat(), "data": data}, f)
    os.replace(tmp, PRICING_CACHE_FILE)


pricing_cache, pricing_cached_at = load_pricing_cache()


def pricing_expired() -> bool:
    if pricing_cached_at is None:
        return True
    return datetime.now() - pricing_cached_at > timedelta(days=TTL_DAYS)


async def get_pricing() -> dict:
    global pricing_cache, pricing_cached_at
    async with pricing_lock:
        if not pricing_cache or pricing_expired():
            data = await pricing_crawler.load_openrouter_prices()
            if data:
                pricing_cache = data
                await asyncio.to_thread(save_pricing_cache, pricing_cache)
                pricing_cached_at = datetime.now()
    return pricing_cache or {}


def normalize_confidence(scores: list[float]) -> list[str]:
    if not scores:
        return []
    if len(scores) == 1:
        return ["단일 결과"]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s: # 모든 점수가 동일
        return ["동일"] * len(scores)
    return [f"{(s - min_s) / (max_s - min_s) * 100:.1f}%" for s in scores]


def fmt_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def safe_int(value) -> int: 
    # 타입 불일치 방어
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def build_model_table(reranked: list, confidences: list[str]) -> list[dict]:
    table = []
    for res, conf in zip(reranked, confidences):
        meta = res["metadata"]
        is_hf = meta.get("source", "hf") == "hf"
        table.append({
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "cost": meta.get("cost", "무료 (오픈소스)"),
            "created_at": (meta.get("created_at") or "")[:7] or "N/A",
            "downloads": fmt_downloads(safe_int(meta.get("downloads", 0))) if is_hf else "-",
            "likes": safe_int(meta.get("likes", 0)) if is_hf else "-",
            "relevance": conf,
            "url": meta.get("url", ""),
        })
    return table


def build_agent_table(reranked: list, confidences: list[str]) -> list[dict]:
    table = []
    for res, conf in zip(reranked, confidences):
        meta = res["metadata"]
        table.append({
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "use_case": meta.get("use_case", ""),
            "supported_llms": meta.get("supported_llms", ""),
            "local_support": meta.get("local_support", False),
            "github_stars": meta.get("github_stars", 0),
            "last_updated": meta.get("last_updated", ""),
            "relevance": conf,
            "url": meta.get("url", ""),
        })
    return table


class RecommenderPipeline:
    def __init__(self):
        self.classifier = query_classifier
        self.store = vector_store
        self.reranker = reranker
        self.llm = ollama_client
        self.hf_crawler = hf_crawler

    # HF API 예외 검색어
    _GENERIC_KW = {
        "ai", "llm", "model", "모델", "인공지능", "언어모델", "추천",
        "최신", "좋은", "best", "good", "top", "latest",
    }

    async def _extract_keyword(self, query: str) -> str | None:
        prompt = f"""\
HuggingFace 모델 검색 키워드를 하나만 출력하십시오.
규칙:
- 모델 이름이 있으면 그대로 사용 (예: DeepSeek, Llama, Gemma, FLUX)
- 없으면 영어 기술 키워드 (예: text-to-image, code-generation, translation)
- 'AI', 'LLM', '모델' 같은 일반 단어는 사용하지 않습니다
- 단어 하나만 출력
예: '그림 그리는 모델' -> text-to-image
예: '최신 DeepSeek 알려줘' -> DeepSeek
질문: {query}
검색어:"""
        try:
            response = await self.llm.generate_raw(prompt)
        except Exception:
            return None

        keyword = response.strip().split("\n")[0].replace('"', "").replace("'", "").strip()

        # LLM 출력 정제
        keyword = re.sub(r"[^a-zA-Z0-9\-_ ]", "", keyword)[:50].strip()

        if not keyword or len(keyword) < 2 or keyword.lower() in self._GENERIC_KW:
            return None
        return keyword

    # 데이터 수집
    async def _gather_model_data(
        self, query: str, pricing: dict
    ) -> tuple[list, list[dict], list[str]]:
        # 1. DB에서 top-20 검색
        search_results = await self.store.query(query, n_results=20, item_type=ItemType.MODEL)
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        # 2. TTL 체크
        if not documents or not is_recently_crawled(query):
            keyword = await self._extract_keyword(query) 
            if documents and keyword and is_recently_crawled(keyword):
                await mark_crawled(query)
            else:
                logger.info("실시간 수집 시작 (키워드: %s)", keyword or "trending")
                new_data = await self.hf_crawler.fetch_models(search_query=keyword, limit=10)
                if new_data:
                    for m in new_data:
                        price_info = pricing_crawler.get_price(pricing, m["name"])
                        m["cost"] = price_info["price_display"]
                    await data_processor.process_and_save(new_data, item_type=ItemType.MODEL)
                    await mark_crawled(query)
                    if keyword:
                        await mark_crawled(keyword)
                    search_results = await self.store.query(query, n_results=20, item_type=ItemType.MODEL)
                    documents = search_results["documents"][0] if search_results["documents"] else []
                    metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        if not documents:
            return [], [], []

        # 3. 리랭킹
        reranked = await asyncio.to_thread(self.reranker.rerank, query, documents, metadatas, 5)
        reranked = [r for r in reranked if r["score"] >= RELEVANCE_THRESHOLD]
        if not reranked:
            return [], [], []

        confidences = normalize_confidence([r["score"] for r in reranked])
        table_data = build_model_table(reranked, confidences)
        refs = [r["text"] for r in reranked] # llm은 딕셔너리보다 자연어 텍스트로 제공하는 것이 good
        return reranked, table_data, refs

    async def _gather_agent_data(
        self, query: str
    ) -> tuple[list, list[dict], list[str]]:
        search_results = await self.store.query(query, n_results=20, item_type=ItemType.AGENT)
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        if not documents or not is_recently_crawled("__agent__"):
            logger.info("GitHub에서 에이전트 프레임워크 수집 중...")
            new_data = await github_crawler.fetch_agent_frameworks()
            if new_data:
                await data_processor.process_and_save(new_data, item_type=ItemType.AGENT)
                await mark_crawled("__agent__")
                search_results = await self.store.query(query, n_results=20, item_type=ItemType.AGENT)
                documents = search_results["documents"][0] if search_results["documents"] else []
                metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        if not documents:
            return [], [], []

        reranked = await asyncio.to_thread(self.reranker.rerank, query, documents, metadatas, 3)
        reranked = [r for r in reranked if r["score"] >= RELEVANCE_THRESHOLD]
        if not reranked:
            return [], [], []

        confidences = normalize_confidence([r["score"] for r in reranked])
        table_data = build_agent_table(reranked, confidences)
        refs = [r["text"] for r in reranked]
        return reranked, table_data, refs

    async def run(self, query: str) -> dict:
        category = await self.classifier.classify(query)

        if category == ItemType.GENERAL:
            return {
                "category": category,
                "answer": await self.llm.generate_response(query=query, context=""),
                "references": [],
                "table_data": [],
            }

        if category == ItemType.AGENT:
            _, table_data, refs = await self._gather_agent_data(query)
        else:
            pricing = await get_pricing()  # MODEL 
            _, table_data, refs = await self._gather_model_data(query, pricing)

        if not refs:
            return {
                "category": category,
                "answer": "관련 정보를 찾을 수 없습니다.",
                "references": [],
                "table_data": [],
            }

        context = "\n\n".join(refs)
        answer = await self.llm.generate_response(query=query, context=context)
        return {
            "category": category,
            "answer": answer,
            "references": refs,
            "table_data": table_data,
        }

    async def run_stream(self, query: str) -> AsyncGenerator[dict, None]:
        # 다음 순서대로 추출(yield)  type: "status" | "table" | "chunk" | "done" | "error"
        yield {"type": "status", "step": 1, "message": "질문 의도 분류 중..."}
        category = await self.classifier.classify(query)

        if category == ItemType.GENERAL:
            yield {"type": "status", "step": 2, "message": "답변 생성 중..."}
            async for chunk in self.llm.stream_response(query=query, context=""):
                yield {"type": "chunk", "content": chunk}
            yield {"type": "done", "category": category}
            return

        yield {"type": "status", "step": 2, "message": "DB 검색 및 크롤링 중..."}

        if category == ItemType.AGENT:
            _, table_data, refs = await self._gather_agent_data(query)
        else:
            pricing = await get_pricing()  # MODEL 가격 조회
            _, table_data, refs = await self._gather_model_data(query, pricing)

        if not refs:
            yield {"type": "error", "message": "관련 정보를 찾을 수 없습니다."}
            yield {"type": "done", "category": category}
            return

        yield {"type": "status", "step": 3, "message": "결과 최적화 완료"}
        yield {"type": "table", "category": category, "data": table_data}

        yield {"type": "status", "step": 4, "message": "답변 생성 중..."}
        context = "\n\n".join(refs)
        async for chunk in self.llm.stream_response(query=query, context=context):
            yield {"type": "chunk", "content": chunk}

        yield {"type": "done", "category": category}


recommender_pipeline = RecommenderPipeline()