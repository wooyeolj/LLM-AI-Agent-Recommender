# 설정값(.env) 관리
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트 경로 설정
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    BASE_DIR: str = BASE

    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_URL: str = "http://127.0.0.1:11434"

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"

    EMBEDDING_MODEL: str = "dragonkue/BGE-m3-ko"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    MODEL_DEVICE: str = "cpu"

    CHROMA_DB_PATH: str = os.path.join(BASE, "data", "chroma_data")

    CACHE_TTL_DAYS: int = 14
    GITHUB_TOKEN: str = ""

    # .env를 상속 , 없다면 기본값
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 실제 API 주소 반환 > ollama_client.py
    @property
    def ollama_chat_url(self) -> str:
        return f"{self.OLLAMA_URL}/api/chat"

settings = Settings()
