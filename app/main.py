# FastAPI 앱 진입점
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings  # 아직 작성 전이지만 미리 임포트 구조를 잡습니다.

app = FastAPI(
    title="LLM Recommender API",
    description="사용자 요구사항에 맞는 최적의 LLM 모델을 추천하는 RAG 기반 API 서비스",
    version="1.0.0"
)

# CORS 설정 (웹 프런트엔드나 외부 호출을 위해 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 특정 도메인으로 제한하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (모든 API는 /api/v1으로 시작하도록 설정)
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    """
    서버 상태 확인을 위한 헬스체크 엔드포인트
    """
    return {
        "message": "LLM Recommender API is running!",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    # 개발 모드이므로 리로드를 활성화합니다.
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.BACKEND_PORT, reload=True)