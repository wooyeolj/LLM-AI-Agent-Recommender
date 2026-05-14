# Streamlit UI — 사용자 입력을 받아 SSE 스트리밍으로 백엔드 응답을 실시간 출력
import streamlit as st
import requests
import pandas as pd
import json
from app.core.config import settings

st.set_page_config(page_title="AI 추천 시스템", layout="wide")

with st.sidebar:
    st.header("AI 추천 시스템")
    st.markdown("GitHub: [링크 추가 예정](#)")
    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("AI 모델 & 에이전트 맞춤 추천")
st.markdown("사용자의 환경에 가장 적합한 모델을 제안합니다.")

BACKEND_URL = settings.BACKEND_URL


def _render_table(category: str, data: list):   # 백엔드에서 받은 추천 결과 리스트를 테이블로
    if not data:
        return

    if category == "MODEL":
        st.subheader("[추천] 모델 비교")
        rows = []
        for item in data:
            ctx = item.get("context_length", 0)
            rows.append({
                "모델명": item.get("name", "-"),
                "설명": (item.get("description", "") or "")[:120],
                "비용": item.get("cost", "-"),
                "컨텍스트": f"{ctx:,}" if ctx else "-",
                "출시": item.get("created_at", "-"),
                "다운로드/월": item.get("downloads", "-"),
                "좋아요": item.get("likes", 0),
                "연관성": item.get("relevance", "-"),
                "링크": item.get("url", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={"링크": st.column_config.LinkColumn("링크")},
        )

    elif category == "AGENT":
        st.subheader("[추천] 에이전트 프레임워크")
        rows = []
        for item in data:
            rows.append({
                "이름": item.get("name", "-"),
                "사용 사례": item.get("use_case", "-"),
                "지원 LLM": item.get("supported_llms", "-"),
                "로컬 지원": "Y" if item.get("local_support") else "N",
                "GitHub Stars": f"{item.get('github_stars', 0):,}",
                "연관성": item.get("relevance", "-"),
                "링크": item.get("url", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={"링크": st.column_config.LinkColumn("링크")},
        )


# 기존 대화 이력 출력 (Streamlit)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("table_data"):
            _render_table(message.get("category", ""), message["table_data"])
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("어떤 도움이 필요하신가요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty() # 상태 메세지
        table_ph = st.empty() # 테이블 + 분류
        answer_ph = st.empty() # LLM 답변

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
                        event = json.loads(raw_line[6:]) # data: {"type":~~~~~ 에서 "data: "를 잘라냄
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "status":
                        step = event.get("step", "")
                        symbol = "[완료]" if step >= 4 else "[...]"
                        status_ph.info(f"{symbol} [{step}/4] {event.get('message', '')}")

                    elif etype == "table":
                        category = event.get("category", "MODEL")
                        table_data = event.get("data", [])
                        status_ph.empty()
                        with table_ph.container():
                            st.caption(f"분류: **{category}**")
                            _render_table(category, table_data)

                    elif etype == "chunk":
                        answer_text += event.get("content", "")
                        answer_ph.markdown(answer_text + " |")

                    elif etype == "done":
                        status_ph.empty()
                        category = event.get("category", category)
                        if not table_data:
                            with table_ph.container():
                                st.caption(f"분류: **{category}**")
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


# 사용자 입력
# → session_state.messages에 user 메시지 저장
# → 빈 자리 3개 선점
#     status_ph  (위)
#     table_ph   (중간)
#     answer_ph  (아래)
# → 백엔드에 POST 요청
#    status (step 1)  : "[...] [1/4] 질문 의도 분류 중..."
#    status (step 2)  : "[...] [2/4] DB 검색 및 크롤링 중..."  
#    status (step 3)  : "[완료] [3/4] 결과 최적화 완료"
#    table            : 분류 + 테이블 출력
#    chunk × N        : LLM 토큰 스트리밍
#    done             :  텍스트 확정
# →응답 완료 후
#     session_state.messages에 assistant 메시지 저장
#       (content + category + table_data 포함)
#     → 다음 rerun 때 대화 이력으로 재출력
