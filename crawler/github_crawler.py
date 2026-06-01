# GitHub API 크롤러 — 에이전트 프레임워크 별점 / 업데이트일 / LLM 지원 여부 수집
import asyncio
import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com/repos"


def github_headers() -> dict:
    # 토큰이 있으면 인증 헤더 추가 (5000req/h), 없으면 비인증 (60req/h)
    headers = {"Accept": "application/vnd.github+json"}
    token = (settings.GITHUB_TOKEN or "").strip()
    if token:
        try:
            token.encode("ascii")
            headers["Authorization"] = f"Bearer {token}"
        except UnicodeEncodeError:
            logger.warning("GITHUB_TOKEN 오류! 비인증으로 진행합니다.")
    return headers

# 미리 선정한 AI 에이전트 프레임워크 목록 15개
AGENT_REPOS = [
    ("crewAIInc/crewAI",         "멀티에이전트 협업, 역할 기반 분업"),
    ("langchain-ai/langgraph",   "상태 머신 기반 에이전트 워크플로우"),
    ("microsoft/autogen",        "멀티에이전트 대화, 코드 실행 자동화"),
    ("run-llama/llama_index",    "RAG 및 데이터 연결 에이전트"),
    ("deepset-ai/haystack",      "검색 기반 RAG 파이프라인"),
    ("Significant-Gravitas/AutoGPT", "자율 목표 달성 에이전트"),
    ("microsoft/semantic-kernel","엔터프라이즈 AI 오케스트레이션"),
    ("agentops-ai/agentops",     "에이전트 모니터링 및 평가"),
    ("superagent-ai/superagent", "에이전트 SaaS 플랫폼"),
    ("OpenBMB/ChatDev",          "소프트웨어 개발 멀티에이전트"),
    ("assafelovic/gpt-researcher","자율 웹 리서치 에이전트"),
    ("yoheinakajima/babyagi",    "태스크 분해 자율 에이전트"),
    ("pydantic/pydantic-ai",     "타입 안전 에이전트, 구조화된 출력"),
    ("huggingface/smolagents",   "HuggingFace 공식 경량 에이전트 프레임워크"),
    ("stanfordnlp/dspy",         "LLM 파이프라인 프로그래밍, 자동 프롬프트 최적화"),
]


async def retry(
    client: httpx.AsyncClient, url: str, retries: int = 2
) -> httpx.Response:
    for attempt in range(retries + 1):
        r = await client.get(url)
        if r.status_code in (429, 403) and attempt < retries:
            try:
                wait = int(r.headers.get("Retry-After", min(2 ** attempt * 5, 60)))
            except (ValueError, TypeError):
                wait = min(2 ** attempt * 5, 60)
            logger.warning("GitHub API 레이트 리밋 — %d초 대기 후 재시도 (%d/%d)", wait, attempt + 1, retries)
            await asyncio.sleep(wait)
            continue
        return r
    return r


class GitHubCrawler:

    async def fetch_agent_frameworks(self) -> list[dict]:
        results = []
        async with httpx.AsyncClient(headers=github_headers(), timeout=15.0) as client:
            for repo_path, use_case in AGENT_REPOS:
                try:
                    r = await retry(client, f"{GITHUB_API}/{repo_path}")
                    if r.status_code != 200:
                        logger.warning("GitHub API 실패 (%s): %d", repo_path, r.status_code)
                        results.append(self._fallback(repo_path, use_case))
                        continue

                    data = r.json()
                    stars = data.get("stargazers_count", 0)
                    topics = data.get("topics", [])
                    desc = data.get("description") or ""

                    supported_llms = infer_llms(topics, desc)
                    local_support = infer_local(topics, desc)

                    results.append({
                        "name": data.get("name", repo_path.split("/")[-1]),
                        "url": data.get("html_url", f"https://github.com/{repo_path}"),
                        "description": desc,
                        "type": "AGENT",
                        "use_case": use_case,
                        "github_stars": stars,
                        "supported_llms": supported_llms,
                        "local_support": local_support,
                        "last_updated": data.get("updated_at", "")[:10],
                    })
                    logger.info("%s: stars=%s", repo_path, f"{stars:,}")
                except Exception as e:
                    logger.error("%s 수집 실패: %s", repo_path, e)
                    results.append(self._fallback(repo_path, use_case))

        return results

    def _fallback(self, repo_path: str, use_case: str) -> dict:
        name = repo_path.split("/")[-1]
        return {
            "name": name,
            "url": f"https://github.com/{repo_path}",
            "description": use_case,
            "type": "AGENT",
            "use_case": use_case,
            "github_stars": 0,
            "supported_llms": "GPT, Claude, Ollama(로컬)",
            "local_support": True,
            "last_updated": "",
        }


def infer_llms(topics: list, desc: str) -> str:
    text = " ".join(topics) + " " + desc.lower()
    llms = []
    if any(k in text for k in ["openai", "gpt"]):
        llms.append("GPT")
    if any(k in text for k in ["anthropic", "claude"]):
        llms.append("Claude")
    if any(k in text for k in ["ollama", "local", "llama"]):
        llms.append("Ollama(로컬)")
    if any(k in text for k in ["gemini", "google"]):
        llms.append("Gemini")
    return ", ".join(llms) if llms else "GPT, Claude, Ollama(로컬)"


def infer_local(topics: list, desc: str) -> bool:
    text = " ".join(topics) + " " + desc.lower()
    return any(k in text for k in ["ollama", "local", "llama.cpp", "open-source"])


github_crawler = GitHubCrawler()
