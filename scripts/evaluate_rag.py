"""
RAG 품질 평가 스크립트 — RAG Triad + CrossEncoder

RAG Triad (LLM-as-judge) 답변 품질
  Context Relevance  — 검색된 문서와 쿼리의 관련도 측정          (0~1)
  Groundedness       — 생성된 답변이 소스 문서에 근거하는지 측정   (0~1)
  Answer Relevance   — 최종 답변과 질문 의도 와의 관련성 측정     (0~1)

CrossEncoder 검색 품질 (sigmoid 변환 후 0~1 확률):
  Top-1 score        — 1위 문서의 관련도 점수
  avg score          — 검색된 전체 문서 관련도 평균

Judge: Ollama (gemma3:4b)
주의: 동일 모델이 생성·평가를 수행하므로 self-flattery 편향 주의
      절대값보다 쿼리 간 상대 비교에 활용하시오

실행: python3 scripts/evaluate_rag.py
"""
import asyncio
import json
import logging
import math
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 불필요 로그 억제
logging.basicConfig(level=logging.WARNING)

from app.core.config import settings
from app.core.types import ItemType
from app.services.ollama_client import ollama_client
from app.services.query_classifier import query_classifier
from app.pipeline.recommender import recommender_pipeline, get_pricing

QUERIES_PATH = os.path.join(settings.BASE_DIR, "scripts", "test_queries.json")
RESULT_PATH  = os.path.join(settings.BASE_DIR, "scripts", "evaluate_rag_result.json")
CONTEXT_MAX  = 800 


# judge 시스템 프롬프트

SYS_CONTEXT_RELEVANCE = """
당신은 정보 검색 품질 평가 전문가입니다.
주어진 질문과 검색된 문서의 관련성을 평가합니다.
반드시 0.0(전혀 관련 없음)에서 1.0(완벽하게 관련 있음) 사이의
숫자 하나만 출력하십시오. 다른 설명은 절대 하지 않습니다.
"""

SYS_GROUNDEDNESS = """
당신은 답변 검증 전문가입니다.
생성된 답변이 제공된 참고 문서에 근거하는지 평가합니다.
반드시 0.0(전혀 근거 없음)에서 1.0(완전히 근거 있음) 사이의
숫자 하나만 출력하십시오. 다른 설명은 절대 하지 않습니다.
"""

SYS_ANSWER_RELEVANCE = """
당신은 답변 품질 평가 전문가입니다.
생성된 답변이 원래 질문에 얼마나 잘 답하는지 평가합니다.
반드시 0.0(전혀 답변 안 됨)에서 1.0(완벽하게 답변됨) 사이의
숫자 하나만 출력하십시오. 다른 설명은 절대 하지 않습니다.
"""


def sigmoid(x: float) -> float:
    return round(1.0 / (1.0 + math.exp(-x)), 4)


def parse_score(text: str) -> float | None:
    match = re.search(r'\b(0(?:\.\d+)?|1(?:\.0*)?)\b', text.strip())
    if match:
        return round(min(1.0, max(0.0, float(match.group(1)))), 3)
    return None


def truncate(text: str) -> str:
    return text[:CONTEXT_MAX] + "..." if len(text) > CONTEXT_MAX else text


def fmt(val: float | None) -> str:
    return f"{val:.3f}" if val is not None else "N/A"

async def judge_context_relevance(query: str, context: str) -> float | None:
    user = f"질문: {query}\n\n검색된 문서:\n{truncate(context)}"
    try:
        return parse_score(await ollama_client.judge(SYS_CONTEXT_RELEVANCE, user))
    except Exception:
        return None


async def judge_groundedness(context: str, answer: str) -> float | None:
    user = f"참고 문서:\n{truncate(context)}\n\n생성된 답변:\n{answer}"
    try:
        return parse_score(await ollama_client.judge(SYS_GROUNDEDNESS, user))
    except Exception:
        return None


async def judge_answer_relevance(query: str, answer: str) -> float | None:
    user = f"질문: {query}\n\n생성된 답변:\n{answer}"
    try:
        return parse_score(await ollama_client.judge(SYS_ANSWER_RELEVANCE, user))
    except Exception:
        return None


async def evaluate_one(query_text: str, category: ItemType, pricing: dict) -> dict:
    result = {
        "category": str(category),
        "retrieval": {"top1_score": None, "avg_score": None, "doc_count": 0},
        "context_relevance": None,
        "groundedness":      None,
        "answer_relevance":  None,
        "rag_score":         None,
        "error":             None,
    }

    try:
        # 1. 검색 + 리랭킹
        if category == ItemType.MODEL:
            reranked, _, refs = await recommender_pipeline._gather_model_data(query_text, pricing)
        elif category == ItemType.AGENT:
            reranked, _, refs = await recommender_pipeline._gather_agent_data(query_text)
        else:
            reranked, refs = [], []

        # 2. CrossEncoder 점수 집계 (logit → sigmoid)
        if reranked:
            scores = [sigmoid(r["score"]) for r in reranked]
            result["retrieval"]["top1_score"] = scores[0]
            result["retrieval"]["avg_score"]  = round(sum(scores) / len(scores), 4)
            result["retrieval"]["doc_count"]  = len(scores)

        # 3. 답변 생성
        context = "\n\n".join(refs)
        answer  = await ollama_client.generate_response(query_text, context)

        # 4. RAG Triad
        if refs:
            cr, gr, ar = await asyncio.gather(
                judge_context_relevance(query_text, context),
                judge_groundedness(context, answer),
                judge_answer_relevance(query_text, answer),
            )
            valid = [s for s in [cr, gr, ar] if s is not None]
            result["rag_score"] = round(sum(valid) / len(valid), 3) if valid else None
        else:
            cr, gr = None, None
            ar = await judge_answer_relevance(query_text, answer)

        result["context_relevance"] = cr
        result["groundedness"]      = gr
        result["answer_relevance"]  = ar

    except Exception as e:
        result["error"] = str(e)

    return result


async def evaluate():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        suite = json.load(f)
    queries = suite["queries"]

    print(f"RAG 품질 평가 시작 (judge={settings.OLLAMA_MODEL}, 쿼리 {len(queries)}개)")
    pricing = await get_pricing()

    all_results = []

    for q in queries:
        qid        = q["id"]
        query_text = q["query"]

        category = await query_classifier.classify(query_text)
        expected = q["expected_category"]
        expected_list = expected if isinstance(expected, list) else [expected]
        match_mark = "✓" if str(category) in expected_list else "✗"

        res = await evaluate_one(query_text, category, pricing)
        all_results.append({**q, **res})

        if res["error"]:
            print(f"Q{qid:02d} [{str(category):<7}] ERR  {res['error'][:60]}")
        else:
            print(f"Q{qid:02d} [{str(category):<7}] {match_mark}  RAG={fmt(res['rag_score'])}")


    def avg(key: str) -> float | None:
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def avg_ret(key: str) -> float | None:
        vals = [r["retrieval"][key] for r in all_results
                if r.get("retrieval", {}).get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def avg_non_general(key: str) -> float | None:
        # GENERAL은 평균에서 제외
        vals = [r[key] for r in all_results
                if r.get(key) is not None and r.get("category") != "GENERAL"]
        return round(sum(vals) / len(vals), 3) if vals else None

    t1  = avg_ret("top1_score")
    av  = avg_ret("avg_score")
    cr_avg = avg_non_general("context_relevance")
    gr_avg = avg_non_general("groundedness")
    ar_avg = avg_non_general("answer_relevance")
    rag_avg = avg_non_general("rag_score")

    retrieval_count = sum(1 for r in all_results if r["retrieval"]["doc_count"] > 0)
    rag_count = sum(1 for r in all_results
                    if r.get("rag_score") is not None and r.get("category") != "GENERAL")

    print("\n[종합 평균]")
    print(f"  AVG RAG Score    {fmt(rag_avg)}  (GENERAL 제외, {rag_count}개 쿼리)")
    print(f"  Context Rel.     {fmt(cr_avg)}")
    print(f"  Groundedness     {fmt(gr_avg)}")
    print(f"  Answer Rel.      {fmt(ar_avg)}")
    print(f"  Retrieval Top-1  {fmt(t1)} / avg {fmt(av)}")

    # 저장 -JSON

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "judge_model":  settings.OLLAMA_MODEL,
        "note": (
            "동일 모델이 생성·평가를 수행하므로 self-flattery 편향 주의. "
            "절대값보다 쿼리 간 상대 비교에 활용하시오. "
            "avg_rag_score는 GENERAL 카테고리를 제외한 MODEL/AGENT 쿼리에서만 계산 "
        ),
        "summary": {
            "total_queries":            len(all_results),
            "retrieval_queries":        retrieval_count,
            "rag_scored_queries":       rag_count,
            "retrieval_top1_score":     t1,
            "retrieval_avg_score":      av,
            "context_relevance_avg":    cr_avg,
            "groundedness_avg":         gr_avg,
            "answer_relevance_avg":     ar_avg,
            "avg_rag_score":            rag_avg,
        },
        "queries": all_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {os.path.relpath(RESULT_PATH, settings.BASE_DIR)}")

    await ollama_client.aclose()


if __name__ == "__main__":
    asyncio.run(evaluate())
