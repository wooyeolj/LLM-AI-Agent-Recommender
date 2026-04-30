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

        if isinstance(result, str):
            return ChatResponse(category="GENERAL", answer=result, references=[], table_data=[], status="NO_DOCS")

        if not result.get("references") and not result.get("table_data"):
            return ChatResponse(
                category=result.get("category", "GENERAL"),
                answer="관련 정보를 찾지 못해 기본 지식으로 답변합니다: " + result.get("answer", ""),
                references=[],
                table_data=[],
                status="NO_DOCS",
            )

        return ChatResponse(
            category=result["category"],
            answer=result["answer"],
            references=result["references"],
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
