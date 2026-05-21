# 사용자 질문을 MODEL / AGENT / GENERAL로 분류  1차 키워드 > 2차 LLM
import re
from app.services.ollama_client import ollama_client
from app.core.types import ItemType

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

CLASS_PATTERN = re.compile(r'(MODEL|AGENT|GENERAL)')


def classify_by_keyword(query: str) -> ItemType | None:
    q = query.lower()
    model_score = sum(1 for k in MODEL_KEYWORDS if k in q)
    agent_score = sum(1 for k in AGENT_KEYWORDS if k in q)

    if model_score == agent_score:
        return None
    return ItemType.MODEL if model_score > agent_score else ItemType.AGENT


class QueryClassifier:
    async def classify(self, query: str) -> ItemType:
        result = classify_by_keyword(query)
        if result:
            return result
        try:
            response = await ollama_client.classify_query(query)
        except Exception:
            return ItemType.GENERAL

        match = CLASS_PATTERN.search(response.strip().upper())
        if match:
            return ItemType(match.group(1))
        return ItemType.GENERAL


query_classifier = QueryClassifier()
