# LLM & AI Agent 추천 시스템

사용자 질문을 분석하여 HuggingFace, OpenRouter, GitHub에서 실시간으로 데이터를 수집하고, RAG(검색 증강 생성) 파이프라인으로 최적의 LLM 모델 또는 AI 에이전트 프레임워크를 추천하는 시스템입니다.

---

## 아키텍처

```
사용자 질문
    │
    ▼
[쿼리 분류기] ── 키워드 기반 1차 분류 → MODEL / AGENT / GENERAL
    │              모호한 경우 LLM fallback
    ▼
[벡터 검색] ── ChromaDB (cosine 유사도)
    │           llm_items 컬렉션 / agent_items 컬렉션
    ▼
[실시간 크롤링] ── DB 미보유 또는 TTL 만료 시
    │              HuggingFace API / GitHub API
    ▼
[리랭커] ── BGE-reranker-v2-m3 (Cross-Encoder)
    │        상위 3개 선별
    ▼
[LLM 답변 생성] ── Ollama (gemma3:4b)
    │              검색 결과를 컨텍스트로 활용
    ▼
SSE 스트리밍 → Streamlit UI
```

---

## 크롤링 전략

| 소스 | 수집 항목 | 용도 |
|------|----------|------|
| **HuggingFace API** | 모델명, 설명, 다운로드수, 좋아요, 태그, 출시일 | LLM/이미지생성 등 모델 메타데이터 |
| **OpenRouter API** | 가격(입출력 per 1M tokens), 컨텍스트 길이 | 상업 모델 가격 정보 |
| **GitHub API** | 별점, 업데이트일, 지원 LLM, 로컬 지원 여부 | 에이전트 프레임워크 인기도 |

- 키워드가 HuggingFace pipeline tag(`text-to-image`, `translation` 등)이면 `pipeline_tag` 파라미터로 정확히 필터링
- 동일 키워드는 14일간 재크롤링 하지 않음 (인메모리 캐시)
- 임베딩 결과는 LRU 방식으로 512개까지 메모리 캐싱

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| LLM | Ollama + gemma3:4b (로컬 실행) |
| 임베딩 | dragonkue/BGE-m3-ko (한국어 특화) |
| 리랭킹 | BAAI/bge-reranker-v2-m3 |
| 벡터 DB | ChromaDB (로컬 영구 저장) |
| 백엔드 | FastAPI + uvicorn (SSE 스트리밍) |
| 프론트엔드 | Streamlit |
| 컨테이너 | Docker + docker-compose |

---

## 실행 방법

### 사전 요구사항

- Python 3.11+
- [Ollama](https://ollama.ai) 설치 및 모델 다운로드

```bash
ollama pull gemma3:4b
```

### 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정 (.env 파일 복사 후 수정)
cp .env.example .env

# 3. 초기 DB 구축 (최초 1회)
python3 scripts/init_db.py

# 4. 백엔드 실행
python3 app/main.py

# 5. 프론트엔드 실행 (새 터미널)
streamlit run frontend.py
```

브라우저에서 `http://localhost:8501` 접속

### Docker 실행 (Ollama는 호스트에서 별도 실행)

```bash
docker compose up --build
```

---

## 환경변수 (.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `gemma3:4b` | 사용할 Ollama 모델 |
| `EMBEDDING_MODEL` | `dragonkue/BGE-m3-ko` | 임베딩 모델 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 리랭킹 모델 |
| `MODEL_DEVICE` | `cpu` | 추론 디바이스 (cpu / cuda) |
| `GITHUB_TOKEN` | *(선택)* | GitHub API 인증 토큰 (없으면 60req/h) |
| `BACKEND_PORT` | `8000` | FastAPI 서버 포트 |

---

## 프로젝트 구조

```
my-llm-project/
├── app/
│   ├── api/routes.py          # FastAPI 엔드포인트 (일반 + SSE 스트리밍)
│   ├── core/config.py         # 환경변수 관리 (pydantic-settings)
│   ├── pipeline/recommender.py # RAG 파이프라인 (분류→검색→크롤→리랭→생성)
│   └── services/
│       ├── embedder.py        # BGE-m3-ko 임베딩 (LRU 캐시 포함)
│       ├── vector_store.py    # ChromaDB CRUD
│       ├── reranker.py        # BGE-reranker-v2-m3
│       ├── query_classifier.py # 키워드 분류 + LLM fallback
│       └── ollama_client.py   # 일반/스트리밍 응답
├── crawler/
│   ├── huggingface_crawler.py # HF REST API (pipeline_tag 정확 필터링)
│   ├── pricing_crawler.py     # OpenRouter API 가격 수집
│   ├── github_crawler.py      # GitHub API 에이전트 프레임워크
│   ├── leaderboard_crawler.py # (예약 — 현재 공개 API 없음)
│   └── data_processor.py      # 크롤링 데이터 → ChromaDB 저장
├── scripts/
│   ├── init_db.py             # 초기 DB 구축 (최초 1회)
│   └── run_crawler.py         # 수동 크롤링 트리거
├── tests/                     # 단위/통합 테스트
├── frontend.py                # Streamlit UI (SSE 스트리밍 소비)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 테스트

### 단위 / 통합 테스트

```bash
python3 tests/test_embedder.py                  # 임베딩 모델 동작 확인
python3 tests/test_reranker.py                  # 리랭커 점수 정렬 확인
python3 tests/test_classifier.py                # 쿼리 분류 정확도 확인
python3 tests/test_crawler.py                   # HuggingFace + GitHub 크롤러 확인
python3 tests/test_search.py                    # 벡터 DB 검색 확인 (init_db 선행 필요)
python3 tests/test_vector_store_with_rerank.py  # 검색 + 리랭킹 통합
python3 tests/test_ollama.py                    # Ollama 연결 확인
python3 tests/test_pipeline.py                  # 전체 파이프라인 통합 테스트
```

### API 엔드포인트 테스트 (백엔드 실행 후)

```bash
python3 tests/test_api.py   # /api/chat + /api/chat/stream 엔드포인트 검증
```

### 분류 정확도 평가

```bash
python3 scripts/evaluate_queries.py   # test_queries.json 기반 파이프라인 평가
```

`data/test_queries.json`에 정의된 5개 쿼리를 실행하고 카테고리 분류 정확도와 테이블 유무를 검증합니다.

| 쿼리 | 허용 카테고리 | 비고 |
|------|-------------|------|
| 발표 PPT 만들어줘 | MODEL 또는 AGENT | LLM fallback 경로 |
| 그림 잘 그리는 AI 추천 | MODEL | 키워드 직접 히트 |
| 자금관리 비서 필요 | AGENT | 키워드 직접 히트 |
| 물리학 과제 도움 AI | MODEL | LLM fallback 경로 |
| 하루 물 섭취량 | GENERAL | AI 무관 일반 질문 |

> LLM fallback 경로 쿼리는 gemma3:4b의 비결정적 특성으로 인해 재실행 시 결과가 달라질 수 있습니다.
