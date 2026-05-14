# BGE-m3-ko 임베딩 모델 래퍼 — 텍스트를 벡터로 변환, FIFO 캐시(512개) 포함
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class Embedder:
    def __init__(self):
        print(f"Loading Embedding Model: {settings.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.MODEL_DEVICE)
        print("Embedding Model loaded successfully.")
        self._cache: dict[str, list[float]] = {}
        self._max_cache = 512

    def get_embedding(self, text: str) -> List[float]:
        """단일 문장을 벡터로 변환. 동일 텍스트는 캐시에서 반환."""
        if text not in self._cache:
            if len(self._cache) >= self._max_cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = self.model.encode(text).tolist()
        return self._cache[text]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 문장을 벡터로 변환. 캐시 미적중 항목만 배치 인코딩."""
        miss = [t for t in texts if t not in self._cache]
        if miss:
            vectors = self.model.encode(miss).tolist()
            for t, v in zip(miss, vectors):
                if len(self._cache) >= self._max_cache:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[t] = v
        return [self._cache[t] for t in texts]


embedder = Embedder()
