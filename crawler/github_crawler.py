import httpx
from typing import List, Dict
from app.core.config import settings

GITHUB_API = "https://api.github.com/repos"


def _github_headers() -> dict:
    """토큰이 있으면 인증 헤더 추가 (5000req/h), 없으면 비인증 (60req/h)"""
    headers = {"Accept": "application/vnd.github+json"}
    token = (settings.GITHUB_TOKEN or "").strip()
    if token:
        try:
            token.encode("ascii")
            headers["Authorization"] = f"Bearer {token}"
        except UnicodeEncodeError:
            print("[!] GITHUB_TOKEN에 잘못된 문자가 있습니다. 비인증으로 진행합니다.")
    return headers

# 추천 대상 AI 에이전트 프레임워크 목록 (org/repo 형식)
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
]


def _difficulty(stars: int) -> str:
    if stars > 30000:
        return "보통~어려움"
    if stars > 10000:
        return "보통"
    return "쉬움~보통"


class GitHubCrawler:

    async def fetch_agent_frameworks(self) -> List[Dict]:
        """
        GitHub API로 에이전트 프레임워크 메타데이터를 수집합니다.
        인증 없이 공개 API 사용 (시간당 60회 제한).
        """
        results = []
        async with httpx.AsyncClient(headers=_github_headers(), timeout=15.0) as client:
            for repo_path, use_case in AGENT_REPOS:
                try:
                    r = await client.get(f"{GITHUB_API}/{repo_path}")
                    if r.status_code != 200:
                        print(f"[!] GitHub API 실패 ({repo_path}): {r.status_code}")
                        results.append(self._fallback(repo_path, use_case))
                        continue

                    data = r.json()
                    stars = data.get("stargazers_count", 0)
                    topics = data.get("topics", [])
                    desc = data.get("description") or ""

                    supported_llms = _infer_llms(topics, desc)
                    local_support = _infer_local(topics, desc)

                    results.append({
                        "name": data.get("name", repo_path.split("/")[-1]),
                        "url": data.get("html_url", f"https://github.com/{repo_path}"),
                        "description": desc,
                        "type": "AGENT",
                        "use_case": use_case,
                        "github_stars": stars,
                        "difficulty": _difficulty(stars),
                        "supported_llms": supported_llms,
                        "local_support": local_support,
                        "last_updated": data.get("updated_at", "")[:10],
                        "language": data.get("language") or "Python",
                        "license": (data.get("license") or {}).get("spdx_id", "Unknown"),
                    })
                    print(f"[*] {repo_path}: ⭐{stars:,}")
                except Exception as e:
                    print(f"[!] {repo_path} 수집 실패: {e}")
                    results.append(self._fallback(repo_path, use_case))

        return results

    def _fallback(self, repo_path: str, use_case: str) -> Dict:
        name = repo_path.split("/")[-1]
        return {
            "name": name,
            "url": f"https://github.com/{repo_path}",
            "description": use_case,
            "type": "AGENT",
            "use_case": use_case,
            "github_stars": 0,
            "difficulty": "보통",
            "supported_llms": "GPT, Claude, Ollama(로컬)",
            "local_support": True,
            "last_updated": "",
            "language": "Python",
            "license": "Unknown",
        }


def _infer_llms(topics: list, desc: str) -> str:
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


def _infer_local(topics: list, desc: str) -> bool:
    text = " ".join(topics) + " " + desc.lower()
    return any(k in text for k in ["ollama", "local", "llama.cpp", "open-source"])


github_crawler = GitHubCrawler()
