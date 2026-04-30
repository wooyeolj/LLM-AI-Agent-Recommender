import httpx
from typing import Dict

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _normalize(model_id: str) -> str:
    return model_id.lower().strip()


def _format_price(prompt_price: str, completion_price: str) -> str:
    """토큰당 가격 → 1M 토큰 기준 표시 문자열로 변환"""
    try:
        p = float(prompt_price) * 1_000_000
        c = float(completion_price) * 1_000_000
        if p == 0 and c == 0:
            return "무료"
        return f"입력 ${p:.2f} / 출력 ${c:.2f} (per 1M tokens)"
    except (ValueError, TypeError):
        return "가격 정보 없음"


class PricingCrawler:

    async def load_openrouter_prices(self) -> Dict[str, Dict]:
        """
        OpenRouter API에서 모델 가격 정보를 가져옵니다.
        반환: {정규화된_모델명: {"price_display": str, "context_length": int,
                                "prompt_price": float, "completion_price": float}}
        """
        print("[*] OpenRouter 가격 데이터 로딩 중...")

        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            try:
                r = await client.get(OPENROUTER_MODELS_URL)
                r.raise_for_status()
            except Exception as e:
                print(f"[!] OpenRouter API 연결 실패: {e}")
                return {}

        result = {}
        for item in r.json().get("data", []):
            model_id = item.get("id", "")
            if not model_id:
                continue

            pricing = item.get("pricing", {})
            prompt_p = pricing.get("prompt", "0")
            completion_p = pricing.get("completion", "0")

            result[_normalize(model_id)] = {
                "price_display": _format_price(prompt_p, completion_p),
                "context_length": item.get("context_length", 0),
                "prompt_price": _safe_float(prompt_p),
                "completion_price": _safe_float(completion_p),
            }

        print(f"[*] OpenRouter 가격 로드 완료: {len(result)}개 모델")
        return result

    def get_price(self, pricing_data: Dict[str, Dict], model_id: str) -> Dict:
        """모델 ID로 가격 정보 조회. 없으면 오픈소스 무료로 반환"""
        key = _normalize(model_id)
        # org/model-name 형식과 model-name 형식 모두 시도
        short = key.split("/")[-1] if "/" in key else key

        info = pricing_data.get(key) or pricing_data.get(short)
        if info:
            return info

        # OpenRouter에 없음 → HuggingFace 전용 오픈소스 모델
        return {
            "price_display": "무료 (오픈소스)",
            "context_length": 0,
            "prompt_price": 0.0,
            "completion_price": 0.0,
        }


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


pricing_crawler = PricingCrawler()
