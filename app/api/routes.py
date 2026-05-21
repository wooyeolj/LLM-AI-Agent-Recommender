# /api/chat , /api/chat/stream 엔드포인트
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.limiter import limiter
from app.pipeline.recommender import RecommenderPipeline


def get_pipeline(request: Request) -> RecommenderPipeline:
    return request.app.state.pipeline

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    category: str
    answer: str
    references: list[str]
    table_data: list[dict] | None = []
    status: str = "SUCCESS"


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    pipeline: RecommenderPipeline = Depends(get_pipeline),
):
    try:
        result = await pipeline.run(body.message)
        category = result.get("category", "GENERAL")

        # 결과가 없을 때 fallback 처리, GENERAL은 SUCCESS로
        if category in ("MODEL", "AGENT") and not result.get("references") and not result.get("table_data"):
            return ChatResponse(
                category=str(category),
                answer=f"관련 정보를 찾지 못해 {settings.OLLAMA_MODEL} 기본 응답으로 답변합니다: " + result.get("answer", ""),
                references=[],
                table_data=[],
                status="NO_DOCS",
            )

        return ChatResponse(
            category=str(category),
            answer=result.get("answer", ""),
            references=result.get("references", []),
            table_data=result.get("table_data", []),
            status="SUCCESS",
        )

    except Exception as e:
        logger.error("서버 에러: %s", e)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다.",
        )


@router.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    pipeline: RecommenderPipeline = Depends(get_pipeline),
):
    async def generate():
        try:
            async for event in pipeline.run_stream(body.message):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("스트리밍 서버 에러: %s", e)
            err = {"type": "error", "message": "서버 내부 오류가 발생했습니다."}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            done = {"type": "done", "category": "ERROR"}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
