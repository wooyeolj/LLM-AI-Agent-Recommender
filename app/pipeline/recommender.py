from datetime import datetime, timedelta
from typing import List, Dict, Any, AsyncGenerator
from app.services.query_classifier import query_classifier
from app.services.vector_store import vector_store
from app.services.reranker import reranker
from app.services.ollama_client import ollama_client
from crawler.huggingface_crawler import HuggingFaceCrawler
from crawler.github_crawler import github_crawler
from crawler.pricing_crawler import pricing_crawler
from crawler.data_processor import data_processor

# 인메모리 쿼리 캐시 — 같은 키워드는 14일 내 재크롤링 안 함
_crawl_cache: Dict[str, datetime] = {}
_TTL_DAYS = 14

# 앱 실행 중 한 번만 로드
_pricing_cache: Dict | None = None


async def _get_pricing():
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = await pricing_crawler.load_openrouter_prices()
    return _pricing_cache


def _is_recently_crawled(keyword: str) -> bool:
    if keyword not in _crawl_cache:
        return False
    return datetime.now() - _crawl_cache[keyword] < timedelta(days=_TTL_DAYS)


def _mark_crawled(keyword: str):
    _crawl_cache[keyword] = datetime.now()


def _normalize_confidence(scores: List[float]) -> List[str]:
    if len(scores) <= 1:
        return ["N/A"] * len(scores)
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return ["N/A"] * len(scores)
    return [f"{(s - min_s) / (max_s - min_s) * 100:.1f}%" for s in scores]


def _fmt_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _build_model_table(reranked, confidences) -> List[Dict]:
    table = []
    for res, conf in zip(reranked, confidences):
        meta = res["metadata"]
        table.append({
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "cost": meta.get("cost", "무료 (오픈소스)"),
            "created_at": (meta.get("created_at") or "")[:7] or "N/A",
            "downloads": _fmt_downloads(int(meta.get("downloads", 0))),
            "likes": int(meta.get("likes", 0)),
            "relevance": conf,
            "url": meta.get("url", ""),
            "context_length": meta.get("context_length", 0),
        })
    return table


def _build_agent_table(reranked, confidences) -> List[Dict]:
    table = []
    for res, conf in zip(reranked, confidences):
        meta = res["metadata"]
        table.append({
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "use_case": meta.get("use_case", ""),
            "supported_llms": meta.get("supported_llms", ""),
            "local_support": meta.get("local_support", False),
            "difficulty": meta.get("difficulty", ""),
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
        self.hf_crawler = HuggingFaceCrawler()

    # 너무 일반적이어서 HF API 검색에 쓸 수 없는 단어들
    _GENERIC_KW = {
        "ai", "llm", "model", "모델", "인공지능", "언어모델", "추천",
        "최신", "좋은", "best", "good", "top", "latest",
    }

    async def _extract_keyword(self, query: str) -> str | None:
        prompt = (
            "너는 HuggingFace 모델 검색 전문가야. 아래 질문에서 HuggingFace에 검색할 구체적인 키워드를 뽑아줘.\n"
            "규칙:\n"
            "- 모델 이름이 보이면 그 이름을 그대로 사용 (예: DeepSeek, Llama, Gemma, FLUX)\n"
            "- 없으면 영어 기술 키워드 (예: text-to-image, code-generation, translation)\n"
            "- 'AI', 'LLM', '모델' 같이 너무 일반적인 단어는 절대 쓰지 마\n"
            "- 단어 하나만 출력\n"
            "예: '그림 그리는 모델' -> text-to-image\n"
            "예: '한국어 잘하는 모델' -> korean\n"
            "예: '최신 DeepSeek 알려줘' -> DeepSeek\n"
            f"질문: {query}\n검색어:"
        )
        response = await self.llm.generate_response(query=prompt, context="")
        keyword = response.strip().split("\n")[0].replace('"', "").replace("'", "").strip()
        if not keyword or len(keyword) < 2 or keyword.lower() in self._GENERIC_KW:
            return None
        return keyword

    # ── 데이터 수집 헬퍼 (run / run_stream 공용) ──────────────────────────

    async def _gather_model_data(self, query: str, pricing: Dict):
        """검색 → 크롤링(필요시) → 리랭킹 수행. (reranked, table_data, refs) 반환"""
        search_results = await self.store.query(query, n_results=20, item_type="MODEL")
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        keyword = await self._extract_keyword(query)
        cache_key = keyword or "__trending__"
        if not documents or not _is_recently_crawled(cache_key):
            print(f"[*] 실시간 수집 시작 (키워드: {keyword or 'trending'})")
            new_data = await self.hf_crawler.fetch_models(search_query=keyword, limit=10)
            if new_data:
                for m in new_data:
                    price_info = pricing_crawler.get_price(pricing, m["name"])
                    m["cost"] = price_info["price_display"]
                    m["context_length"] = price_info["context_length"]
                await data_processor.process_and_save(new_data, item_type="MODEL")
                _mark_crawled(cache_key)
                requery = keyword if keyword else query
                search_results = await self.store.query(requery, n_results=20, item_type="MODEL")
                documents = search_results["documents"][0] if search_results["documents"] else []
                metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        if not documents:
            return [], [], []

        reranked = self.reranker.rerank(query, documents, metadatas, top_n=3)
        confidences = _normalize_confidence([r["score"] for r in reranked])
        table_data = _build_model_table(reranked, confidences)
        refs = [r["text"] for r in reranked]
        return reranked, table_data, refs

    async def _gather_agent_data(self, query: str):
        """에이전트 검색 → 크롤링(필요시) → 리랭킹 수행. (reranked, table_data, refs) 반환"""
        search_results = await self.store.query(query, n_results=20, item_type="AGENT")
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        if not documents or not _is_recently_crawled("__agent__"):
            print("[*] GitHub에서 에이전트 프레임워크 수집 중...")
            new_data = await github_crawler.fetch_agent_frameworks()
            if new_data:
                await data_processor.process_and_save(new_data, item_type="AGENT")
                _mark_crawled("__agent__")
                search_results = await self.store.query(query, n_results=20, item_type="AGENT")
                documents = search_results["documents"][0] if search_results["documents"] else []
                metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []

        if not documents:
            return [], [], []

        reranked = self.reranker.rerank(query, documents, metadatas, top_n=3)
        confidences = _normalize_confidence([r["score"] for r in reranked])
        table_data = _build_agent_table(reranked, confidences)
        refs = [r["text"] for r in reranked]
        return reranked, table_data, refs

    # ── 기존 단일 응답 (하위 호환) ────────────────────────────────────────

    async def run(self, query: str) -> Dict[str, Any]:
        category = await self.classifier.classify(query)

        if category == "GENERAL":
            return {
                "category": "GENERAL",
                "answer": await self.llm.generate_response(query=query, context=""),
                "references": [],
                "table_data": [],
            }

        pricing = await _get_pricing()

        if category == "AGENT":
            _, table_data, refs = await self._gather_agent_data(query)
        else:
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
            "role": "assistant",
            "category": category,
            "answer": answer,
            "references": refs,
            "table_data": table_data,
        }

    # ── 스트리밍 응답 ────────────────────────────────────────────────────

    async def run_stream(self, query: str) -> AsyncGenerator[Dict, None]:
        """
        SSE 이벤트를 순서대로 yield 합니다.
        type: "status" | "table" | "chunk" | "done" | "error"
        """
        yield {"type": "status", "step": 1, "message": "질문 의도 분류 중..."}
        category = await self.classifier.classify(query)

        if category == "GENERAL":
            yield {"type": "status", "step": 2, "message": "답변 생성 중..."}
            async for chunk in self.llm.stream_response(query=query, context=""):
                yield {"type": "chunk", "content": chunk}
            yield {"type": "done", "category": "GENERAL"}
            return

        pricing = await _get_pricing()

        yield {"type": "status", "step": 2, "message": "DB 검색 및 크롤링 중..."}

        if category == "AGENT":
            _, table_data, refs = await self._gather_agent_data(query)
        else:
            _, table_data, refs = await self._gather_model_data(query, pricing)

        if not refs:
            yield {"type": "error", "message": "관련 정보를 찾을 수 없습니다."}
            yield {"type": "done", "category": category}
            return

        yield {"type": "status", "step": 3, "message": "결과 최적화 완료 ✓"}
        yield {"type": "table", "category": category, "data": table_data}

        yield {"type": "status", "step": 4, "message": "답변 생성 중..."}
        context = "\n\n".join(refs)
        async for chunk in self.llm.stream_response(query=query, context=context):
            yield {"type": "chunk", "content": chunk}

        yield {"type": "done", "category": category}


recommender_pipeline = RecommenderPipeline()
