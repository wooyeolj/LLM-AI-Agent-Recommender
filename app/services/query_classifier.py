# 사용자 질문을 MODEL / AGENT / GENERAL로 분류 — 키워드 1차 판단, 모호하면 LLM fallback
import re
from app.services.ollama_client import ollama_client
from app.core.types import ItemType

# LLM 호출을 최소화 하기 위한 키워드 매칭. LLM은 키워드로 판단이 안 될 때만 호출.
MODEL_KEYWORDS = [
    'gpt', 'llama', 'claude', 'gemini', 'deepseek',
    '모델', 'llm', '언어모델', 'model', '생성모델', '멀티모달',
    'rag', '임베딩', '파인튜닝', '프롬프트', '페르소나',
    '리팩토링', '텍스트 생성', '추론', '챗봇',
]

AGENT_KEYWORDS = [
    'autogpt', 'crewai', 'langchain', 'langgraph', 'autogen',
    '에이전트', 'agent', '멀티에이전트', 'multi-agent', '자율', 'autonomous', '비서',
    '워크플로우', '자동화', '파이프라인', '오케스트레이션',
    '모니터링', '스케줄', '리서치',
]

# LLM 분류 패턴
CLASS_PATTERN = re.compile(r'(MODEL|AGENT|GENERAL)')


def classify_by_keyword(query: str) -> ItemType | None:
    q = query.lower()
    model_score = sum(1 for k in MODEL_KEYWORDS if k in q)
    agent_score = sum(1 for k in AGENT_KEYWORDS if k in q)

    if model_score == agent_score:  # 0:0 포함 동점 → LLM 판단
        return None
    return ItemType.MODEL if model_score > agent_score else ItemType.AGENT


class QueryClassifier:
    async def classify(self, query: str) -> ItemType:
        result = classify_by_keyword(query)
        if result:
            return result

        # 키워드로 판단 불가능한 경우에만 LLM 호출
        # classify_query()는 지시문을 system 과 user로 분리해 인젝션 차단
        try:
            response = await ollama_client.classify_query(query)
        except Exception:
            return ItemType.GENERAL

        # 첫 번째 키워드만 추출
        match = CLASS_PATTERN.search(response.strip().upper())
        if match:
            return ItemType(match.group(1))
        return ItemType.GENERAL


query_classifier = QueryClassifier()
