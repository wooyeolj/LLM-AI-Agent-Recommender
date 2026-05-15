# /api/chat (단일 응답)와 /api/chat/stream (SSE 스트리밍) 엔드포인트 정의
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.pipeline.recommender import recommender_pipeline


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    category: str
    answer: str
    references: List[str]
    table_data: Optional[List[Dict[str, Any]]] = []
    status: str = "SUCCESS"

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await recommender_pipeline.run(request.message)
        category = result.get("category", "GENERAL")

        # 검색이 필요한 카테고리(MODEL/AGENT)인데 결과가 없을 때만 fallback 처리.
        # GENERAL은 검색이 불필요한 정상 경로이므로 SUCCESS로 응답.
        if category in ("MODEL", "AGENT") and not result.get("references") and not result.get("table_data"):
            return ChatResponse(
                category=category,
                answer="관련 정보를 찾지 못해 기본 지식으로 답변합니다: " + result.get("answer", ""),
                references=[],
                table_data=[],
                status="NO_DOCS",
            )

        return ChatResponse(
            category=category,
            answer=result.get("answer", ""),
            references=result.get("references", []),
            table_data=result.get("table_data", []),
            status="SUCCESS",
        )

    except Exception as e:
        print(f"[!] 서버 에러: {e}")
        return ChatResponse(
            category="ERROR",
            answer="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            references=[],
            table_data=[],
            status="ERROR",
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for event in recommender_pipeline.run_stream(request.message):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
