# BGE-reranker-v2-m3 Cross-Encoder — 벡터 검색 결과를 쿼리와 직접 비교해 관련도 순으로 재정렬
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from app.core.config import settings


class Reranker:
    def __init__(self):
        print(f"Loading Reranker Model: {settings.RERANKER_MODEL}...")
        self.model = CrossEncoder(settings.RERANKER_MODEL, device=settings.MODEL_DEVICE)
        print("Reranker Model loaded successfully.")

    def rerank(self, query: str, documents: List[str], metadatas: List[Dict], top_n: int = 3) -> List[Dict[str, Any]]:
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        if isinstance(scores, float):
            scores = [scores]

        combined = [
            {"text": doc, "score": float(score), "metadata": meta}
            for doc, score, meta in zip(documents, scores, metadatas)
        ]
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_n]


reranker = Reranker()
