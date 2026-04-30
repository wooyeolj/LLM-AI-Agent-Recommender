# BAAI/bge-reranker-v2-m3 리랭킹

from typing import List, Dict, Any
from FlagEmbedding import FlagReranker
from app.core.config import settings

class Reranker:
    def __init__(self):
        print(f"Loading Reranker Model: {settings.RERANKER_MODEL}...")
        self.model = FlagReranker(
            settings.RERANKER_MODEL, 
            use_fp16=False, 
            device=settings.MODEL_DEVICE
        )
        print("Reranker Model loaded successfully.")

    def rerank(self, query: str, documents: List[str], metadatas: List[Dict], top_n: int = 3) -> List[Dict[str, Any]]:
        """질문과 문서 리스트, 메타데이터를 받아 점수순 정렬 후 반환"""
        if not documents:
            return []

        # 1. 점수 계산
        pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(pairs)
        
        # 2. 문서 + 점수 + 메타데이터 결합
        combined = []
        # scores가 단일 float일 경우 리스트로 변환 (입력이 1개일 때 대비)
        if isinstance(scores, float):
            scores = [scores]

        for doc, score, meta in zip(documents, scores, metadatas):
            combined.append({
                "text": doc,
                "score": float(score),
                "metadata": meta  # 메타데이터를 통째로 보관 (표 데이터 생성용)
            })
        
        # 3. 점수 기준 내림차순 정렬 및 상위 N개 반환
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_n]

reranker = Reranker()