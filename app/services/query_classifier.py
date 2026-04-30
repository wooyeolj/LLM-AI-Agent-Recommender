from app.services.ollama_client import ollama_client

MODEL_KEYWORDS = [
    '모델', 'llm', 'gpt', 'gemma', 'llama', 'claude', 'deepseek',
    'mistral', 'qwen', '언어모델', '추천', '비교', 'model', '파라미터',
    '생성모델', '텍스트 생성', 'ai 모델', '인공지능 모델', 'solar',
    'exaone', 'bllossom', 'hcx', 'hyperclova', 'phi', 'gemini',
    'command', 'falcon', 'vicuna', 'alpaca', 'wizard', 'orca',
]
AGENT_KEYWORDS = [
    '에이전트', 'agent', 'autogpt', 'crewai', 'langchain', 'langgraph',
    '자율', '워크플로우', '자동화', '비서', '태스크', '멀티에이전트',
    'multi-agent', 'autonomous', '플로우', 'flow', 'orchestrat',
]


def _classify_by_keyword(query: str) -> str | None:
    q = query.lower()
    model_score = sum(1 for k in MODEL_KEYWORDS if k in q)
    agent_score = sum(1 for k in AGENT_KEYWORDS if k in q)

    if model_score == 0 and agent_score == 0:
        return None  # 모호 → LLM 판단
    if model_score >= agent_score:
        return "MODEL"
    return "AGENT"


class QueryClassifier:
    async def classify(self, query: str) -> str:
        result = _classify_by_keyword(query)
        if result:
            return result

        # 키워드로 판단 불가능한 경우에만 LLM 호출
        prompt = f"""당신은 분류 전문가입니다. 사용자의 입력 문장을 분석하여 반드시 다음 세 단어 중 하나로만 분류하세요.

1. MODEL: LLM, 언어모델, 이미지 생성 모델, Llama, GPT, Gemma 등 특정 'AI 모델' 자체를 찾거나 추천해달라는 경우.
2. AGENT: AutoGPT, CrewAI, 스스로 작업하는 비서, 특정 목적을 수행하는 'AI 에이전트' 시스템을 찾는 경우.
3. GENERAL: 인사, 잡담, 날씨, 혹은 AI 모델/에이전트와 전혀 상관없는 일반적인 질문인 경우.

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
