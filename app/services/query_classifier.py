# 사용자 질문을 MODEL / AGENT / GENERAL로 분류 — 키워드 1차 판단, 모호하면 LLM fallback
from app.services.ollama_client import ollama_client

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


def _classify_by_keyword(query: str) -> str | None:
    q = query.lower()
    model_score = sum(1 for k in MODEL_KEYWORDS if k in q)
    agent_score = sum(1 for k in AGENT_KEYWORDS if k in q)

    if model_score == agent_score:  # 0:0 포함 동점 → LLM 판단
        return None
    return "MODEL" if model_score > agent_score else "AGENT"


class QueryClassifier:
    async def classify(self, query: str) -> str:
        result = _classify_by_keyword(query)
        if result:
            return result

        # 키워드로 판단 불가능한 경우에만 LLM 호출
        prompt = f"""당신은 분류 전문가입니다. 사용자의 입력 문장을 분석하여 반드시 다음 세 단어 중 하나로만 분류하세요.

1. AGENT: AutoGPT, CrewAI, 스스로 / 정기적으로 / 자동으로 작업하여 특정 목적을 수행하는 'AI 에이전트' 시스템을 찾는 경우.
2. MODEL: LLM, 언어모델, 이미지 생성 모델, Llama, GPT, Gemma 등 특정 'AI 모델' 자체를 찾거나 추천해달라는 경우.
3. GENERAL: AI 모델/에이전트와 전혀 상관없는 일반적인 질문인 경우.

[입력 문장]: "{query}"

결과는 다른 설명 없이 오직 한 단어(MODEL, AGENT, GENERAL)로만 대답하십시오."""

        response = await ollama_client.generate_response(query=prompt, context="")
        cleaned = response.strip().upper()

        if "MODEL" in cleaned:
            return "MODEL"
        if "AGENT" in cleaned:
            return "AGENT"
        return "GENERAL"


query_classifier = QueryClassifier()
