# BGE-m3-ko 임베딩 모델
import logging
import os
import threading
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.utils import LazyProxy

logger = logging.getLogger(__name__)

if settings.OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Embedder:
    def __init__(self):
        logger.info("Loading Embedding Model: %s...", settings.EMBEDDING_MODEL)
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.MODEL_DEVICE,
            local_files_only=settings.OFFLINE_MODE,
        )
        logger.info("Embedding Model loaded successfully.")
        self._cache: dict[str, list[float]] = {}
        self._max_cache = 512
        self._lock = threading.Lock()

    def get_embedding(self, text: str) -> list[float]:
        with self._lock:
            if text not in self._cache:
                if len(self._cache) >= self._max_cache:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = self.model.encode(text).tolist()
            return self._cache[text]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            miss = [t for t in texts if t not in self._cache]
            if miss:
                vectors = self.model.encode(miss).tolist()
                for t, v in zip(miss, vectors):
                    if len(self._cache) >= self._max_cache:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[t] = v
            return [self._cache[t] for t in texts]


embedder = LazyProxy(Embedder)
