# OpenRouter API 크롤러 — 모델 카탈로그 및 가격(per 1M tokens) 수집
import logging
from datetime import datetime, timezone
import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DESC_MAX_LEN = 300


def normalize(model_id: str) -> str:
    return model_id.lower().strip()


def format_price(prompt_price: str, completion_price: str) -> str:
    try:
        p = float(prompt_price) * 1_000_000
        c = float(completion_price) * 1_000_000
        if p == 0 and c == 0:
            return "무료"
        return f"입력 ${p:.2f} / 출력 ${c:.2f} (per 1M tokens)"
    except (ValueError, TypeError):
        return "가격 정보 없음"


class PricingCrawler:

    async def load_openrouter_prices(self) -> dict[str, dict]:
        logger.info("OpenRouter 가격 데이터 로딩 중...")

        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            try:
                r = await client.get(OPENROUTER_MODELS_URL)
                r.raise_for_status()
            except Exception as e:
                logger.error("OpenRouter API 연결 실패: %s", e)
                return {}

        result = {}
        for item in r.json().get("data", []):
            model_id = item.get("id", "")
            if not model_id:
                continue

            pricing = item.get("pricing", {})
            prompt_p = pricing.get("prompt", "0")
            completion_p = pricing.get("completion", "0")

            result[normalize(model_id)] = {
                "price_display": format_price(prompt_p, completion_p),
            }

        logger.info("OpenRouter 가격 로드 완료: %d개 모델", len(result))
        return result

    async def fetch_models(self) -> list[dict]:
        # OpenRouter 모델 카탈로그
        logger.info("OpenRouter 모델 카탈로그 로딩 중...")

        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            try:
                r = await client.get(OPENROUTER_MODELS_URL)
                r.raise_for_status()
            except Exception as e:
                logger.error("OpenRouter API 연결 실패: %s", e)
                return []

        models = []
        for item in r.json().get("data", []):
            model_id = item.get("id", "")
            if not model_id or model_id.endswith(":free"):
                continue

            pricing = item.get("pricing", {})
            cost = format_price(pricing.get("prompt", "0"), pricing.get("completion", "0"))

            description = item.get("description", "") or ""
            if len(description) > DESC_MAX_LEN:
                description = description[:DESC_MAX_LEN].rsplit(" ", 1)[0] + "..."
            if not description:
                description = f"OpenRouter 모델: {item.get('name') or model_id}"

            created_epoch = item.get("created")
            if isinstance(created_epoch, (int, float)) and created_epoch > 0:
                created_iso = datetime.fromtimestamp(created_epoch, tz=timezone.utc).isoformat()
            else:
                created_iso = ""

            models.append({
                "name": model_id,
                "url": f"https://openrouter.ai/{model_id}",
                "description": description,
                "type": "MODEL",
                "pipeline_tag": "text-generation",
                "downloads": 0,
                "likes": 0,
                "tags": [],
                "createdAt": created_iso,
                "cost": cost,
                "source": "openrouter",
            })

        logger.info("OpenRouter 모델 카탈로그 로드 완료: %d개", len(models))
        return models

    def get_price(self, pricing_data: dict[str, dict], model_id: str) -> dict:
        key = normalize(model_id)
        # org/model-name 형식과 model-name 형식 시도
        short = key.split("/")[-1] if "/" in key else key

        info = pricing_data.get(key) or pricing_data.get(short)
        if info:
            return info

        # OpenRouter에 없음 →  오픈소스 처리
        return {
            "price_display": "무료 (오픈소스)",
        }


pricing_crawler = PricingCrawler()
