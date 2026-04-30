# FastAPI 앱 진입점
import asyncio
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.limiter import limiter
from app.services.ollama_client import ollama_client
from app.services.embedder import embedder
from app.services.reranker import reranker
from app.services.vector_store import vector_store
from app.pipeline.recommender import recommender_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.GITHUB_TOKEN:
        logger.warning(
            "GITHUB_TOKEN 미설정 — GitHub API 호출이 60 req/hr로 제한됨"
        )

    await asyncio.to_thread(embedder._load)     # BGE-m3-ko
    await asyncio.to_thread(reranker._load)     # bge-reranker-v2-m3
    await asyncio.to_thread(vector_store._load) # ChromaDB
    logger.info("ML 모델 로딩 완료")

    app.state.pipeline = recommender_pipeline

    logger.info("모델 및 클라이언트 준비 완료")
    yield
    await ollama_client.aclose()
    logger.info("httpx 클라이언트 정상 해제")


app = FastAPI(
    title="LLM Recommender API",
    description="AI 모델 & 에이전트 맞춤 추천",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "LLM Recommender API is running!",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=True)
