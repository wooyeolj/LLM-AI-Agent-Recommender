import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트 절대 경로 (이 파일 기준 3단계 상위)
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM & Agent Recommender"
    BASE_DIR: str = _BASE

    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_URL: str = "http://127.0.0.1:11434"

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    # Streamlit → FastAPI 연결 주소. Docker에서는 서비스명 "app" 사용
    BACKEND_URL: str = "http://localhost:8000"

    EMBEDDING_MODEL: str = "dragonkue/BGE-m3-ko"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    MODEL_DEVICE: str = "cpu"

    # 데이터 경로 — 절대경로 기본값, .env에서 오버라이드 가능
    CHROMA_DB_PATH: str = os.path.join(_BASE, "data", "chroma_data")

    CACHE_TTL_DAYS: int = 30
    HF_TOP_N: int = 30
    GITHUB_TOKEN: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def ollama_chat_url(self) -> str:
        return f"{self.OLLAMA_URL}/api/chat"


settings = Settings()
