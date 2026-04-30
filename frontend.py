import streamlit as st
import requests
import pandas as pd
import json
from app.core.config import settings

st.set_page_config(page_title="AI 추천 시스템", layout="wide")
st.title("AI 모델 & 에이전트 맞춤 추천")
st.markdown("질문을 입력하면 실시간으로 정보를 검색하여 최적의 솔루션을 제안합니다.")

BACKEND_URL = f"http://localhost:{settings.BACKEND_PORT}"


def _render_table(category: str, data: list):
    if not data:
        return

    if category == "MODEL":
        st.subheader("📊 추천 모델 비교")
        rows = []
        for item in data:
            rows.append({
                "모델명": item.get("name", "-"),
                "설명": (item.get("description", "") or "")[:80],
                "비용": item.get("cost", "-"),
                "출시": item.get("created_at", "-"),
                "다운로드/월": item.get("downloads", "-"),
                "좋아요": item.get("likes", 0),
                "연관성": item.get("relevance", "-"),
                "링크": item.get("url", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={"링크": st.column_config.LinkColumn("링크")},
        )

    elif category == "AGENT":
        st.subheader("🤖 추천 에이전트 프레임워크")
        rows = []
        for item in data:
            rows.append({
                "이름": item.get("name", "-"),
                "사용 사례": item.get("use_case", "-"),
                "지원 LLM": item.get("supported_llms", "-"),
                "로컬 지원": "✓" if item.get("local_support") else "✗",
                "난이도": item.get("difficulty", "-"),
                "GitHub ⭐": f"{item.get('github_stars', 0):,}",
                "연관성": item.get("relevance", "-"),
                "링크": item.get("url", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={"링크": st.column_config.LinkColumn("링크")},
        )


# 기존 대화 이력 출력
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("table_data"):
            _render_table(message.get("category", ""), message["table_data"])
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("어떤 AI 모델이나 에이전트를 찾고 계신가요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        table_ph = st.empty()
        answer_ph = st.empty()

        answer_text = ""
        table_data = []
        category = "GENERAL"

        try:
            with requests.post(
                f"{BACKEND_URL}/api/chat/stream",
                json={"message": prompt},
                stream=True,
                timeout=300,
            ) as resp:
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(raw_line[6:])
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "status":
                        step = event.get("step", "")
                        status_ph.info(f"{'⚙️' if step < 4 else '✅'} [{step}/4] {event.get('message', '')}")

                    elif etype == "table":
                        category = event.get("category", "MODEL")
                        table_data = event.get("data", [])
                        status_ph.empty()
                        with table_ph.container():
                            _render_table(category, table_data)

                    elif etype == "chunk":
                        answer_text += event.get("content", "")
                        answer_ph.markdown(answer_text + "▌")

                    elif etype == "done":
                        status_ph.empty()
                        answer_ph.markdown(answer_text)

                    elif etype == "error":
                        status_ph.empty()
                        st.error(event.get("message", "오류가 발생했습니다."))

        except Exception as e:
            st.error(f"서버 연결 실패: {e}")

        if answer_text:
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "category": category,
                "table_data": table_data,
            })
