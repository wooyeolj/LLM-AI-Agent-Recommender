# 크롤링 데이터를 ChromaDB 저장용으로 전처리 
import logging
from datetime import datetime
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# pipeline_tag → 한국어 설명 매핑 (reranker의 한국어 쿼리 매칭 보조)
PIPELINE_KO = {
    "text-generation":              "텍스트생성 언어모델 LLM 대화 글쓰기 작문 창작",
    "text-to-image":                "이미지생성 그림생성 텍스트-이미지 그림그리기 일러스트 AI그림",
    "image-to-text":                "이미지설명 이미지이해 이미지분석 사진설명 문자인식 OCR",
    "image-to-image":               "이미지변환 이미지편집 스타일변환 사진편집 포토샵 이미지보정",
    "text-to-speech":               "TTS 텍스트읽기 음성변환 목소리생성",
    "text-to-audio":                "오디오생성 음악생성 음악작곡",
    "automatic-speech-recognition": "음성인식 STT 음성텍스트변환 받아쓰기 자막생성",
    "translation":                  "번역 언어번역 자동번역",
    "summarization":                "요약 문서요약 텍스트요약 뉴스요약 내용정리",
    "question-answering":           "질문답변 질의응답 QA FAQ 검색",
    "image-classification":         "이미지분류 사진분류 이미지인식 물체인식",
    "object-detection":             "객체탐지 물체감지 사물인식 YOLO 감지",
    "text-classification":          "텍스트분류 문서분류 자동분류",
    "token-classification":         "개체명인식 정보추출 token",
    "zero-shot-classification":     "제로샷분류 유연한분류 무학습분류",
    "image-segmentation":           "이미지분할 영역분리 배경분리 세그멘테이션",
    "feature-extraction":           "특징추출 임베딩 벡터추출",
    "conversational":               "대화 챗봇 AI어시스턴트 상담 대화형AI",
}


class DataProcessor:
    def __init__(self):
        self.store = vector_store

    async def process_and_save(self, data_list: list[dict], item_type: str = "MODEL"):
        if not data_list:
            logger.warning("저장할 데이터가 없습니다.")
            return

        if item_type == "AGENT":
            await self._save_agents(data_list)
        else:
            await self._save_models(data_list)

    async def _save_models(self, data_list: list[dict]):
        texts, metadatas, ids = [], [], []
        now = datetime.now().isoformat()

        for item in data_list:
            name = item.get("name", "")
            desc = item.get("description", "")
            tags = item.get("tags", [])
            tag_str = ", ".join(tags[:10]) if isinstance(tags, list) else str(tags)
            pipeline_tag = item.get("pipeline_tag", "")
            task_str = PIPELINE_KO.get(pipeline_tag, pipeline_tag)
            cost = item.get("cost", "무료 (오픈소스)")
            source = item.get("source", "hf")

            texts.append(
                f"모델명: {name}\n태스크: {task_str}\n설명: {desc}\n가격: {cost}\n태그: {tag_str}"
            )
            metadatas.append({
                "name": name,
                "url": item.get("url", ""),
                "type": "MODEL",
                "source": source,
                "description": desc,
                "pipeline_tag": pipeline_tag,
                "cost": item.get("cost", "무료 (오픈소스)"),
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "tags": tag_str,
                "created_at": item.get("createdAt", ""),
                "collected_at": now,
            })
            clean_id = name.replace("/", "_").replace(".", "-").lower()
            ids.append(f"{source}_{clean_id}")

        try:
            await self.store.upsert_documents(
                ids=ids, texts=texts, metadatas=metadatas, item_type="MODEL"
            )
        except Exception as e:
            logger.error("모델 DB 저장 에러: %s", e)

    async def _save_agents(self, data_list: list[dict]):
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
                "github_stars": int(item.get("github_stars") or 0),
                "last_updated": item.get("last_updated", ""),
                "collected_at": now,
            })
            clean_id = name.lower().replace(" ", "-").replace("_", "-")
            ids.append(f"agent_{clean_id}")

        try:
            await self.store.upsert_documents(
                ids=ids, texts=texts, metadatas=metadatas, item_type="AGENT"
            )
        except Exception as e:
            logger.error("에이전트 DB 저장 에러: %s", e)


data_processor = DataProcessor()
