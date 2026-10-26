# BGE-reranker-v2-m3 Cross-Encoder
import logging
import os
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.core.utils import LazyProxy

logger = logging.getLogger(__name__)

if settings.OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Reranker:
    def __init__(self):
        logger.info("Loading Reranker Model: %s...", settings.RERANKER_MODEL)
        self.model = CrossEncoder(
            settings.RERANKER_MODEL,
            device=settings.MODEL_DEVICE,
            local_files_only=settings.OFFLINE_MODE,
        )
        logger.info("Reranker Model loaded successfully.")

    def rerank(self, query: str, documents: list[str], metadatas: list[dict], top_n: int = 3) -> list[dict]:
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        #scores는 list임을 보장
        if isinstance(scores, float):
            scores = [scores]

        combined = [
            {"text": doc, "score": float(score), "metadata": meta}
            for doc, score, meta in zip(documents, scores, metadatas)
        ]
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_n]


reranker = LazyProxy(Reranker)
