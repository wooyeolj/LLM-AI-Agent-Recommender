# ChromaDB 인터페이스 — llm_items / agent_items 컬렉션 관리, cosine 유사도 검색
import chromadb
from app.core.config import settings
from app.services.embedder import embedder
from typing import List, Dict, Any


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

        # LLM 모델 컬렉션
        self.model_collection = self.client.get_or_create_collection(
            name="llm_items",
            metadata={"hnsw:space": "cosine"},
        )
        # AI 에이전트 컬렉션 (별도 분리)
        self.agent_collection = self.client.get_or_create_collection(
            name="agent_items",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"VectorStore initialized at: {settings.CHROMA_DB_PATH}")

    def _get_collection(self, item_type: str):
        return self.agent_collection if item_type == "AGENT" else self.model_collection

    async def upsert_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        item_type: str = "MODEL",
    ):
        """있으면 갱신, 없으면 추가 (DuplicateIDError 없음)"""
        collection = self._get_collection(item_type)
        try:
            embeddings = embedder.get_embeddings(texts)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"[OK] Upserted {len(ids)} {item_type} documents")
        except Exception as e:
            print(f"[ERROR] Error upserting documents: {e}")
            raise e

    async def query(
        self,
        query_text: str,
        n_results: int = 20,
        item_type: str = "MODEL",
    ) -> Dict[str, Any]:
        """질문과 가장 유사한 문서 검색 (메타데이터 포함)"""
        collection = self._get_collection(item_type)
        query_embedding = embedder.get_embedding(query_text)

        # 컬렉션이 비어있으면 빈 결과 반환
        if collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        actual_n = min(n_results, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
        )
        return results


vector_store = VectorStore()
