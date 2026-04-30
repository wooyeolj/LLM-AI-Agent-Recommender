"""
FastAPI 엔드포인트 테스트
실행 전 백엔드가 실행 중이어야 합니다: python3 app/main.py
"""
import sys
import os
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

BASE_URL = f"http://localhost:{settings.BACKEND_PORT}"


def test_health():
    print("[1] 헬스체크 테스트")
    r = requests.get(f"{BASE_URL}/", timeout=5)
    assert r.status_code == 200, f"헬스체크 실패: {r.status_code}"
    print(f"  ✅ {r.json()['message']}")


def test_chat_general():
    print("\n[2] GENERAL 분류 테스트 (LLM 직접 답변)")
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "안녕하세요"},
        timeout=120,
    )
    assert r.status_code == 200, f"응답 오류: {r.status_code}"
    data = r.json()
    assert data["category"] == "GENERAL", f"분류 오류: {data['category']}"
    assert data["answer"], "답변이 비어있음"
    print(f"  ✅ 카테고리: {data['category']}")
    print(f"  답변 앞부분: {data['answer'][:80]}...")


def test_chat_model():
    print("\n[3] MODEL 분류 테스트 (RAG 추천)")
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "코딩에 좋은 LLM 모델 추천해줘"},
        timeout=300,
    )
    assert r.status_code == 200, f"응답 오류: {r.status_code}"
    data = r.json()
    assert data["category"] == "MODEL", f"분류 오류: {data['category']}"
    assert isinstance(data["table_data"], list), "table_data가 리스트가 아님"
    print(f"  ✅ 카테고리: {data['category']}")
    print(f"  추천 모델 수: {len(data['table_data'])}개")
    for m in data["table_data"]:
        print(f"    - {m.get('name')} | 비용: {m.get('cost')} | 다운로드: {m.get('downloads')}")


def test_chat_agent():
    print("\n[4] AGENT 분류 테스트 (에이전트 추천)")
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "자율 에이전트 프레임워크 추천해줘"},
        timeout=300,
    )
    assert r.status_code == 200, f"응답 오류: {r.status_code}"
    data = r.json()
    assert data["category"] == "AGENT", f"분류 오류: {data['category']}"
    assert isinstance(data["table_data"], list), "table_data가 리스트가 아님"
    print(f"  ✅ 카테고리: {data['category']}")
    print(f"  추천 에이전트 수: {len(data['table_data'])}개")
    for a in data["table_data"]:
        print(f"    - {a.get('name')} | 별점: {a.get('github_stars'):,}")


def test_stream_endpoint():
    print("\n[5] SSE 스트리밍 엔드포인트 테스트")
    chunks = []
    with requests.post(
        f"{BASE_URL}/api/chat/stream",
        json={"message": "안녕"},
        stream=True,
        timeout=120,
    ) as r:
        assert r.status_code == 200, f"응답 오류: {r.status_code}"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            chunks.append(event)
            if event["type"] == "done":
                break

    types = [e["type"] for e in chunks]
    assert "chunk" in types, "스트리밍 chunk 없음"
    assert "done" in types, "done 이벤트 없음"
    print(f"  ✅ 수신 이벤트: {types}")


if __name__ == "__main__":
    print("=" * 50)
    print("  FastAPI 엔드포인트 테스트")
    print(f"  대상: {BASE_URL}")
    print("=" * 50)
    try:
        test_health()
        test_chat_general()
        test_chat_model()
        test_chat_agent()
        test_stream_endpoint()
        print("\n" + "=" * 50)
        print("  ✅ 모든 테스트 통과")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 백엔드 연결 실패 — python3 app/main.py 를 먼저 실행하세요")
