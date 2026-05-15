# FastAPI 엔드포인트(/api/chat, /api/chat/stream) HTTP 응답과 SSE 스트리밍이 정상 동작하는지 확인 (백엔드 실행 필요)
import sys
import os
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

BASE_URL = f"http://localhost:{settings.BACKEND_PORT}"


def test_health():
    print("[1] 헬스체크")
    r = requests.get(f"{BASE_URL}/", timeout=5)
    assert r.status_code == 200
    print(f"  {r.json()['message']}")


def test_chat_general():
    print("\n[2] GENERAL 분류")
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": "안녕하세요"}, timeout=120)
    data = r.json()
    assert data["category"] == "GENERAL"
    assert data["status"] == "SUCCESS", f"GENERAL이 검색 실패로 처리됨: status={data['status']}"
    assert not data["answer"].startswith("관련 정보를 찾지 못해"), "GENERAL 답변에 잘못된 fallback 접두사가 붙음"
    print(f"  카테고리: {data['category']} | 답변: {data['answer'][:60]}...")


def test_chat_model():
    print("\n[3] MODEL 분류")
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": "코딩에 좋은 LLM 모델 추천해줘"}, timeout=300)
    data = r.json()
    assert data["category"] == "MODEL"
    print(f"  카테고리: {data['category']} | 추천: {len(data['table_data'])}개")
    for m in data["table_data"]:
        print(f"    - {m.get('name')} | 비용: {m.get('cost')}")


def test_chat_agent():
    print("\n[4] AGENT 분류")
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": "자율 에이전트 프레임워크 추천해줘"}, timeout=300)
    data = r.json()
    assert data["category"] == "AGENT"
    print(f"  카테고리: {data['category']} | 추천: {len(data['table_data'])}개")
    for a in data["table_data"]:
        print(f"    - {a.get('name')} | 별점: {a.get('github_stars'):,}")


def test_stream():
    print("\n[5] SSE 스트리밍")
    chunks = []
    with requests.post(f"{BASE_URL}/api/chat/stream", json={"message": "안녕"}, stream=True, timeout=120) as r:
        assert r.status_code == 200
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            chunks.append(event)
            if event["type"] == "done":
                break

    types = [e["type"] for e in chunks]
    assert "chunk" in types and "done" in types
    print(f"  수신 이벤트: {types}")


if __name__ == "__main__":
    print(f"대상: {BASE_URL}")
    try:
        test_health()
        test_chat_general()
        test_chat_model()
        test_chat_agent()
        test_stream()
        print("\n모든 테스트 통과!")
    except AssertionError as e:
        print(f"\n테스트 실패: {e}")
    except requests.exceptions.ConnectionError:
        print(f"\n백엔드 연결 실패 — python3 app/main.py 를 먼저 실행하세요")
