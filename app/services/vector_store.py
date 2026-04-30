# ChromaDB 인터페이스 — llm_items / agent_items 컬렉션 관리, cosine 유사도 검색
import asyncio
import logging
import chromadb
from app.core.config import settings
from app.core.utils import LazyProxy
from app.services.embedder import embedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

        self.model_collection = self.client.get_or_create_collection(
            name="llm_items",
            metadata={"hnsw:space": "cosine"},
        )
        self.agent_collection = self.client.get_or_create_collection(
            name="agent_items",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore initialized at: %s", settings.CHROMA_DB_PATH)

    def _get_collection(self, item_type: str):
        return self.agent_collection if item_type == "AGENT" else self.model_collection

    async def upsert_documents(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        item_type: str = "MODEL",
    ):
        collection = self._get_collection(item_type)
        try:
            embeddings = await asyncio.to_thread(embedder.get_embeddings, texts)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info("Upserted %d %s documents", len(ids), item_type)
        except Exception as e:
            logger.error("Error upserting documents: %s", e)
            raise

    async def query(
        self,
        query_text: str,
        n_results: int = 20,
        item_type: str = "MODEL",
    ) -> dict:
        collection = self._get_collection(item_type)
        query_embedding = await asyncio.to_thread(embedder.get_embedding, query_text)

        # 컬렉션이 비어있으면 빈 결과 반환
        if collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        actual_n = min(n_results, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
        )
        return results


vector_store = LazyProxy(VectorStore)
