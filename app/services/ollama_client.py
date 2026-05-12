# Ollama HTTP 클라이언트 — 단일 응답(generate_response)과 SSE 스트리밍(stream_response) 제공
import httpx
import json
from typing import AsyncGenerator
from app.core.config import settings

# 모듈 레벨 싱글톤 클라이언트 (매 요청마다 생성 방지)
_client = httpx.AsyncClient(timeout=120.0)


def _build_prompt(query: str, context: str) -> str:
    if context:
        return f"[참고 정보]\n{context}\n\n[사용자 질문]\n{query}\n\n한국어로 답변하세요."
    return f"[사용자 질문]\n{query}\n\n한국어로 답변하세요."


class OllamaClient:

    async def generate_response(self, query: str, context: str) -> str:
        """단일 응답 (기존 방식 유지)"""
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": _build_prompt(query, context)}],
            "stream": False,
        }
        try:
            response = await _client.post(settings.ollama_chat_url, json=payload)
            if response.status_code != 200:
                return f"Ollama 에러: {response.text}"
            return response.json().get("message", {}).get("content", "답변 생성 실패")
        except Exception as e:
            return f"Ollama 연결 실패: {str(e)}"

    async def stream_response(
        self, query: str, context: str
    ) -> AsyncGenerator[str, None]:
        """토큰 단위 스트리밍 응답 — async for chunk in client.stream_response(...)"""
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": _build_prompt(query, context)}],
            "stream": True,
        }
        try:
            async with _client.stream(
                "POST", settings.ollama_chat_url, json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("done"):
                        break
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
        except Exception as e:
            yield f"\n[Ollama 연결 실패: {e}]"


ollama_client = OllamaClient()
