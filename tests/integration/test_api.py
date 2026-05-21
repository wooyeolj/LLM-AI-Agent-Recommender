"""
FastAPI 엔드포인트 통합 테스트 — 백엔드 서버 실행 필요
실행: pytest tests/integration/test_api.py -m integration -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pytest
import requests

from app.core.config import settings

BASE_URL = f"http://localhost:{settings.BACKEND_PORT}"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", autouse=True)
def check_server_running():
    try:
        requests.get(f"{BASE_URL}/", timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"백엔드 서버 미실행 — {BASE_URL} 에 서버를 먼저 시작하세요")


class TestHealthCheck:

    def test_health_returns_200(self):
        r = requests.get(f"{BASE_URL}/", timeout=5)
        assert r.status_code == 200
        assert "message" in r.json()


class TestChatEndpoint:

    def test_general_classification(self):
        r = requests.post(f"{BASE_URL}/api/chat", json={"message": "안녕하세요"}, timeout=120)
        data = r.json()
        assert data["category"] == "GENERAL"
        assert data["status"] == "SUCCESS"
        assert not data["answer"].startswith("관련 정보를 찾지 못해")

    def test_model_classification_returns_table(self):
        r = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": "코딩에 좋은 LLM 모델 추천해줘"},
            timeout=300,
        )
        data = r.json()
        assert data["category"] == "MODEL"
        assert len(data["table_data"]) > 0, "MODEL 쿼리 결과 누락"
        assert data["table_data"][0].get("name"), "모델명 누락"

    def test_agent_classification_returns_table(self):
        r = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": "자율 에이전트 프레임워크 추천해줘"},
            timeout=300,
        )
        data = r.json()
        assert data["category"] == "AGENT"
        assert len(data["table_data"]) > 0, "AGENT 쿼리 결과가 비어있음"
        assert data["table_data"][0].get("name"), "에이전트명 누락"


class TestStreamEndpoint:

    def test_sse_stream_emits_chunk_and_done(self):
        chunks = []
        with requests.post(
            f"{BASE_URL}/api/chat/stream",
            json={"message": "안녕"},
            stream=True,
            timeout=120,
        ) as r:
            assert r.status_code == 200
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                chunks.append(event)
                if event["type"] == "done":
                    break

        types = [e["type"] for e in chunks]
        assert "chunk" in types, "chunk 이벤트 없음"
        assert "done" in types, "done 이벤트 없음"

        answer = "".join(e.get("content", "") for e in chunks if e["type"] == "chunk")
        assert answer.strip(), "스트리밍 응답 본문 공백"
