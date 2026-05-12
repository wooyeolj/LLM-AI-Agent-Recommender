# 크롤링 데이터를 ChromaDB 저장용으로 전처리 — 텍스트 포맷 구성, 메타데이터 정규화, upsert 실행
from app.services.vector_store import vector_store
from typing import List, Dict
from datetime import datetime

# pipeline_tag → 한국어 설명 매핑 (reranker의 한국어 쿼리 매칭 보조)
_PIPELINE_KO = {
    "text-generation":              "텍스트생성 언어모델 LLM 대화 코딩",
    "text-to-image":                "이미지생성 그림생성 텍스트-이미지 image generation",
    "image-to-text":                "이미지설명 이미지이해 캡셔닝 image understanding",
    "image-to-image":               "이미지변환 이미지편집 image editing",
    "text-to-speech":               "음성합성 TTS voice synthesis",
    "text-to-audio":                "오디오생성 음악생성 audio generation",
    "automatic-speech-recognition": "음성인식 STT speech recognition",
    "translation":                  "번역 translation",
    "summarization":                "요약 summarization",
    "question-answering":           "질문답변 QA question answering",
    "image-classification":         "이미지분류 image classification",
    "object-detection":             "객체탐지 object detection",
}


class DataProcessor:
    def __init__(self):
        self.store = vector_store

    async def process_and_save(self, data_list: List[Dict], item_type: str = "MODEL"):
        if not data_list:
            print("저장할 데이터가 없습니다.")
            return

        if item_type == "AGENT":
            await self._save_agents(data_list)
        else:
            await self._save_models(data_list)

    async def _save_models(self, data_list: List[Dict]):
        texts, metadatas, ids = [], [], []
        now = datetime.now().isoformat()

        for item in data_list:
            name = item.get("name", "")
            desc = item.get("description", "")
            tags = item.get("tags", [])
            tag_str = ", ".join(tags[:10]) if isinstance(tags, list) else str(tags)
            pipeline_tag = item.get("pipeline_tag", "")
            task_str = _PIPELINE_KO.get(pipeline_tag, pipeline_tag)
            cost = item.get("cost", "무료 (오픈소스)")

            texts.append(
                f"모델명: {name}\n태스크: {task_str}\n설명: {desc}\n가격: {cost}\n태그: {tag_str}"
                #f"{name}은 {task_str}에 특화된 모델입니다. 가격: {cost}. {desc} 관련 태그: {tag_str}"
            )
            metadatas.append({
                "name": name,
                "url": item.get("url", ""),
                "type": "MODEL",
                "description": desc,
                "pipeline_tag": pipeline_tag,
                "cost": item.get("cost", "무료 (오픈소스)"),
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "tags": tag_str,
                "created_at": item.get("createdAt", ""),
                "context_length": int(item.get("context_length") or 0),
                "collected_at": now,
            })
            clean_id = name.replace("/", "_").replace(".", "-").lower()
            ids.append(f"hf_{clean_id}")

        try:
            await self.store.upsert_documents(
                ids=ids, texts=texts, metadatas=metadatas, item_type="MODEL"
            )
        except Exception as e:
            print(f"[ERROR] 모델 DB 저장 에러: {e}")

    async def _save_agents(self, data_list: List[Dict]):
        texts, metadatas, ids = [], [], []
        now = datetime.now().isoformat()

        for item in data_list:
            name = item.get("name", "")
            desc = item.get("description", "")
            use_case = item.get("use_case", "")

            texts.append(
                f"에이전트: {name}\n사용 사례: {use_case}\n설명: {desc}\n"
                f"지원 LLM: {item.get('supported_llms', '')}\n"
                f"로컬 지원: {'가능' if item.get('local_support', False) else '불가'}"
            )
            metadatas.append({
                "name": name,
                "url": item.get("url", ""),
                "type": "AGENT",
                "description": desc,
                "use_case": use_case,
                "supported_llms": item.get("supported_llms", ""),
                "local_support": bool(item.get("local_support", False)),
                "difficulty": item.get("difficulty", "보통"),
                "github_stars": int(item.get("github_stars") or 0),
                "last_updated": item.get("last_updated", ""),
                "language": item.get("language", "Python"),
                "license": item.get("license", "Unknown"),
                "collected_at": now,
            })
            clean_id = name.lower().replace(" ", "-").replace("_", "-")
            ids.append(f"agent_{clean_id}")

        try:
            await self.store.upsert_documents(
                ids=ids, texts=texts, metadatas=metadatas, item_type="AGENT"
            )
        except Exception as e:
            print(f"[ERROR] 에이전트 DB 저장 에러: {e}")


data_processor = DataProcessor()
