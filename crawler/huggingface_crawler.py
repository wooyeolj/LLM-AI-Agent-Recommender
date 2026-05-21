# HuggingFace REST API 크롤러 - 초기 DB용
import logging
import httpx

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api/models"
HEADERS = {"Accept": "application/json"}

# HF pipeline_tag 값 목록 — 이 키워드는 search 대신 pipeline_tag 파라미터로 전달
HF_PIPELINE_TAGS = {
    "text-generation", "text-to-image", "image-to-text", "image-to-image",
    "text-to-speech", "text-to-audio", "automatic-speech-recognition",
    "translation", "summarization", "question-answering",
    "text-classification", "token-classification", "zero-shot-classification",
    "object-detection", "image-classification", "image-segmentation",
    "feature-extraction", "conversational",
}


def parse_item(item: dict) -> dict | None:
    model_id = item.get("modelId") or item.get("id", "")
    if not model_id:
        return None

    tags = item.get("tags", [])
    description = (
        item.get("cardData", {}).get("model_description")
        or item.get("description")
        or ""
    )
    if description and len(description) > 300:
        description = description[:300].rsplit(" ", 1)[0] + "..."

    return {
        "name": model_id,
        "url": f"https://huggingface.co/{model_id}",
        "description": description or f"HuggingFace 모델: {model_id}",
        "type": "MODEL",
        "pipeline_tag": item.get("pipeline_tag", ""),
        "downloads": item.get("downloads", 0),
        "likes": item.get("likes", 0),
        "tags": tags,
        "createdAt": item.get("createdAt", ""),
        "cost": "무료 (오픈소스)",
        "source": "hf",
    }


class HuggingFaceCrawler:
    def __init__(self):
        self.base_url = HF_API_BASE

    async def fetch_models(self, search_query: str = None, limit: int = 10) -> list[dict]:

        params = {"sort": "downloads", "limit": limit, "full": "true"}

        # pipeline_tag는 정확 필터, search는 키워드 매칭 — HF API 파라미터 구분
        if not search_query:
            params["pipeline_tag"] = "text-generation"
        elif search_query.lower() in HF_PIPELINE_TAGS:
            params["pipeline_tag"] = search_query.lower()
        else:
            params["search"] = search_query

        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            try:
                response = await client.get(self.base_url, params=params, timeout=20.0)
                if response.status_code == 400 and "search" in params:
                    logger.warning("HF 검색어 오류 (%r) — text-generation 상위 모델로 대체", search_query)
                    params = {"sort": "downloads", "limit": limit, "full": "true",
                              "pipeline_tag": "text-generation"}
                    response = await client.get(self.base_url, params=params, timeout=20.0)
                response.raise_for_status()
            except Exception as e:
                logger.error("HuggingFace API 연결 실패: %s", e)
                return []

        return [m for item in response.json() if (m := parse_item(item))]

    async def fetch_top_models(self, limit: int = 50) -> list[dict]:
        params = {
            "sort": "downloads",
            "limit": limit,
            "pipeline_tag": "text-generation",
            "full": "true",
        }
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            try:
                response = await client.get(self.base_url, params=params, timeout=30.0)
                response.raise_for_status()
            except Exception as e:
                logger.error("HuggingFace API 연결 실패: %s", e)
                return []

        return [m for item in response.json() if (m := parse_item(item))]


hf_crawler = HuggingFaceCrawler()
