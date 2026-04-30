
# OllamaClient 단위 테스트 — httpx.AsyncClient 모킹으로 Ollama 서버 없이 실행

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_client(status_code: int = 200, body: dict | None = None) -> MagicMock:
    # httpx.AsyncClient mock 생성 헬퍼
    if body is None:
        body = {"message": {"role": "assistant", "content": "테스트 응답"}}

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = body
    mock_response.request = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    return mock_client


class TestGenerateRaw:

    async def test_success(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client(body={"message": {"content": "MODEL"}})

        result = await client.generate_raw("분류해줘")

        assert result == "MODEL"
        client._client.post.assert_called_once()
        # system 역할 없이 user 메시지만 전달됐는지 확인
        call_kwargs = client._client.post.call_args
        messages = call_kwargs.kwargs["json"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "분류해줘"

    async def test_http_error_raises(self):
        import httpx
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client(status_code=500)

        with pytest.raises(httpx.HTTPStatusError):
            await client.generate_raw("테스트")

    async def test_empty_content_returns_empty_string(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client(body={"message": {}})

        result = await client.generate_raw("테스트")
        assert result == ""


class TestGenerateResponse:

    async def test_success_with_context(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client(body={"message": {"content": "답변입니다"}})

        result = await client.generate_response("질문", "컨텍스트")

        assert result == "답변입니다"
        call_kwargs = client._client.post.call_args
        messages = call_kwargs.kwargs["json"]["messages"]
        # system + user 두 메시지여야 함
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_user_message_includes_context(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client()

        await client.generate_response("내 질문", "참고 자료")

        call_kwargs = client._client.post.call_args
        user_content = call_kwargs.kwargs["json"]["messages"][1]["content"]
        assert "참고 자료" in user_content
        assert "내 질문" in user_content

    async def test_no_context_user_message_is_query_only(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client()

        await client.generate_response("질문만", "")

        call_kwargs = client._client.post.call_args
        user_content = call_kwargs.kwargs["json"]["messages"][1]["content"]
        assert "질문만" in user_content
        assert "[참고 정보]" not in user_content

    async def test_http_error_raises(self):
        import httpx
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client(status_code=503)

        with pytest.raises(httpx.HTTPStatusError):
            await client.generate_response("질문", "")

    async def test_stream_false_payload(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        client._client = _make_client()

        await client.generate_response("질문", "")

        call_kwargs = client._client.post.call_args
        assert call_kwargs.kwargs["json"]["stream"] is False

    async def test_aclose_called(self):
        from app.services.ollama_client import OllamaClient
        client = OllamaClient()
        mock_http = _make_client()
        client._client = mock_http

        await client.aclose()
        mock_http.aclose.assert_called_once()
