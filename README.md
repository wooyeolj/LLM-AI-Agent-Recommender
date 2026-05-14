# LLM & AI Agent 추천 시스템

2022년 ChatGPT 출시 이후 수백 개의 LLM과 에이전트 프레임워크가 출시되면서, 목적에 맞는 AI 도구를 선택하는 것이 하나의 과제가 되었다.
이 프로젝트는 사용자의 자연어 질문을 기반으로 HuggingFace, OpenRouter, GitHub API를 이용해 실시간으로 데이터를 수집하고, RAG (검색 증강 생성) 기술로 최적의 LLM 모델 또는 AI 에이전트 프레임워크를 제공한다.

---

## 목차

- [프로젝트 소개](#프로젝트-소개)
- [프로젝트 데모](#프로젝트-데모)
- [모델 특징](#모델-특징)
- [아키텍처](#아키텍처)
- [설계 결정](#설계-결정)
- [크롤링 전략](#크롤링-전략)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [배포 설계](#배포-설계)
- [실행 방법](#실행-방법)
- [AI 모델 · 추론 환경 교체](#ai-모델--추론-환경-교체)
- [환경변수](#환경변수-env)
- [테스트](#테스트)
- [분류 정확도 평가](#분류-정확도-평가)
- [기술적 도전과 해결](#기술적-도전과-해결)
- [향후 개선 방향](#향후-개선-방향)

---

## 프로젝트 소개

본 프로젝트는 AI 도구의 선택 장벽을 낮추기 위해 사용자의 자연어 질문을 기반으로 최적의 AI 도구를 찾는다.
최초 1회 `init_db.py`로 초기 DB를 구성하며, 이후 요청마다 **분류–검색–크롤링–리랭킹–생성** 의 5단계 파이프라인이 동작한다.

[초기화]

`init_db.py` 실행 시 HuggingFace, OpenRouter, GitHub API에서 데이터를 수집해 ChromaDB에 저장

[요청 처리]

1. 사용자 질문을 **키워드 분류 + LLM fallback(gemma3:4b)** 으로 분류 (MODEL / AGENT / GENERAL)
2. 질문을 **임베딩(BGE-m3-ko)** 해 ChromaDB에서 유사 후보 20개 검색
2-2. 신규 키워드이거나 TTL(14일)이 만료된 경우 **HuggingFace / GitHub API** 에서 실시간 수집 후 DB UPSERT (UPDATE & INSERT)
3. **리랭킹(BGE-reranker-v2-m3)** 으로 쿼리 관련도를 비교해 상위 3~5개 선별
4. **Ollama(gemma3:4b)** 가 선별된 결과를 SSE 스트리밍을 통한 자연어 답변으로 제공

외부 유료 API 없이 완전 로컬 실행이 가능하며, Docker와 로컬 환경을 모두 지원한다.

---

## 프로젝트 데모

> 스크린샷 / GIF 추가 예정

---

## 모델 특징

### Gemma3:4b — 로컬 추론 LLM

| 항목 | 내용 |
|------|------|
| 크기 | 4B 파라미터로 CPU, Google Colab 무료 환경에서도 안정적 |
| 한국어 | 모델 크기 대비 한국어 이해 및 생성 품질 우수 |
| 추론 | 구조화된 답변 생성과 분류 태스크에서 안정적 |
| 선택 이유 | 외부 API 비용 없이 로컬 실행을 유지하면서도 실용적인 품질 확보 |

### dragonkue/BGE-m3-ko — 임베딩

| 항목 | 내용 |
|------|------|
| 베이스 | BAAI/BGE-M3 (다국어 임베딩 SOTA 모델) |
| 특화 | 한국어 데이터로 추가 파인튜닝 |
| 선택 이유 | 기반 모델 대비 한국어 쿼리 의미 매칭 정확도 향상 |

### BAAI/bge-reranker-v2-m3 — 리랭킹

| 항목 | 내용 |
|------|------|
| 베이스 | 임베딩 모델과 동일한 아키텍처 기반(BGE) — 언어를 같은 방식으로 이해해 검색과 리랭킹 일관성 유지 |
| 구조 | Cross-Encoder — 쿼리와 문서 쌍을 직접 비교해 정밀한 관련도 점수 산출 |
| 선택 이유 | 벡터 검색의 한계를 보완해 상위 결과의 정확도를 높임 |

---

## 아키텍처

```
사용자 질문
    │
    ▼
[쿼리 분류기] ── 키워드 기반 1차 분류 → MODEL / AGENT / GENERAL
    │              모호한 경우 LLM fallback (gemma3:4b)
    ▼
[벡터 검색] ── ChromaDB (cosine 유사도)
    │           llm_items DB / agent_items DB
    ▼
[실시간 크롤링] ── 신규 키워드 또는 TTL(14일) 만료 시 실행
    │              HuggingFace API / GitHub API
    ▼
[리랭커] ── BGE-reranker-v2-m3 (Cross-Encoder)
    │        MODEL 상위 5개 / AGENT 상위 3개 선별
    ▼
[LLM 답변 생성] ── Ollama (gemma3:4b)
    │              선별 결과로 답변 생성
    ▼
SSE 스트리밍 → Streamlit UI (실시간 출력)
```

---

## 설계 결정

주요 기술 선택의 배경과 강점을 정리

| 결정 | 대안 | 선택 이유 및 강점 | 소스코드 |
|------|------|-----------------|---------|
| **Retrieve & Rerank** ChromaDB(20개) → CrossEncoder(MODEL 5개 / AGENT 3개) | CrossEncoder만 전체 DB에 적용 | CrossEncoder는 쿼리-문서 쌍을 하나씩 비교하므로 문서 수에 비례해 시간이 증가. ChromaDB로 후보를 먼저 좁혀 정밀도 및 속도 확보 | `app/pipeline/recommender.py` |
| **카테고리 분류** MODEL / AGENT / GENERAL | 분류 없음 또는 2분류 | MODEL/AGENT는 각각 다른 DB로 저장 및 검색이 이뤄지고, GENERAL은 LLM으로 바로 처리해 불필요한 검색을 방지 | `app/services/query_classifier.py` |
| **분류 구조** 키워드 1차 → LLM fallback | 모든 쿼리를 LLM으로 분류 | "LLM", "에이전트" 등 명확한 키워드가 있는 쿼리는 수ms 내 분류 가능. LLM 호출(3~10초)은 키워드 분류가 불가능한 경우에만 발생해 평균 응답 시간 단축 | `app/services/query_classifier.py` |
| **SSE 스트리밍** | 일반 HTTP 또는 WebSocket | 본 서비스는 서버→클라이언트 단방향 스트리밍만 필요. SSE는 HTTP 표준으로 FastAPI `StreamingResponse`로 간단히 구현되어 사용자의 체감 대기 시간 단축 | `app/api/routes.py` |
| **싱글톤 패턴** (임베딩 · 리랭커) | 요청마다 모델 로드 | BGE-m3-ko, CrossEncoder 각각 초기 로드에 10~30초 소요. 앱 시작 시 한 번만 로드하는 싱글톤 구조로 추가 지연 방지 | `app/services/embedder.py`, `app/services/reranker.py` |
| **한국어 태그 보강** (`_PIPELINE_KO`) | 영어 원문 그대로 저장 | "그림 그려줘"와 "text-to-image" 간 임베딩 거리가 큼. 한국어 동의어("이미지생성 그림생성")를 미리 매핑해 한국어 쿼리와 태그 간 매칭 정확도 향상 | `crawler/data_processor.py` |
| **Ollama 로컬 LLM** | LLM API | API 키 없이 누구나 실행 가능해 접근성 확보 및 모델 교체 유연성 제공 | `app/services/ollama_client.py` |

---

## 크롤링 전략

| 소스 | 수집 항목 |
|------|----------|
| **HuggingFace API** | 모델명, 설명, pipeline_tag, 다운로드수, 좋아요, 태그, 출시일 |
| **OpenRouter API** | 가격(입출력 per 1M tokens), 컨텍스트 길이 |
| **GitHub API** | 별점, 업데이트일, 지원 LLM, 로컬 지원 여부, 설명, 사용 사례 |

- 쿼리 키워드와 pipeline tag 매핑으로 필터링 정확도 향상
- 동일 키워드는 14일간 재크롤링 하지 않음
- OpenRouter API에 가격이 기재되어 있지 않다면 무료 처리
- 임베딩 결과는 FIFO 방식으로 512개까지 메모리 캐싱

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.11+ |
| LLM | Ollama + gemma3:4b (로컬 실행) |
| 임베딩 | dragonkue/BGE-m3-ko (한국어 특화 튜닝) |
| 리랭킹 | BAAI/bge-reranker-v2-m3 |
| 벡터 DB | ChromaDB |
| 백엔드 | FastAPI + uvicorn (SSE 스트리밍) |
| 프론트엔드 | Streamlit |
| 컨테이너 | Docker |

---

## 프로젝트 구조

```
my-llm-project/
├── app/
│   ├── main.py                # FastAPI 앱 진입
│   ├── api/routes.py          # FastAPI 엔드포인트
│   ├── core/config.py         # 환경변수 관리
│   ├── pipeline/recommender.py # RAG 파이프라인 (분류→검색→크롤링→리랭킹→생성)
│   └── services/
│       ├── embedder.py        # BGE-m3-ko 임베딩
│       ├── vector_store.py    # ChromaDB 저장 및 검색
│       ├── reranker.py        # BGE-reranker-v2-m3 리랭킹
│       ├── query_classifier.py # 쿼리 분류
│       └── ollama_client.py   # LLM 응답 생성
├── crawler/
│   ├── huggingface_crawler.py # HF API 수집
│   ├── openrouter_crawler.py  # OpenRouter API 수집
│   ├── github_crawler.py      # GitHub API 수집
│   └── data_processor.py      # 크롤링 데이터 가공 · 저장
├── scripts/
│   ├── init_db.py             # 초기 DB 구축 (최초 1회)
│   ├── run_crawler.py         # 수동 크롤링 트리거
│   └── evaluate_queries.py    # 쿼리 분류 정확도 평가
├── tests/                     # 단위/통합 테스트
├── frontend.py                # Streamlit UI (SSE 스트리밍) 
├── .env.example               # 환경변수 템플릿
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 배포 설계 — 로컬 · Docker 동시 지원

Ollama(LLM)는 **호스트**에서 실행되고, 백엔드(FastAPI)와 UI(Streamlit)는 **로컬 또는 Docker** 중 선택해 실행할 수 있다.

**LLM 교체 유연성**
LLM이 호스트에서 분리 실행되므로 `OLLAMA_MODEL`로 모델을 교체하거나, `OLLAMA_URL`로 추론 환경을 Google Colab의 GPU로 전환할 수 있다.

**`.env` 수정 없이 즉시 실행**
`config.py`가 로컬 기본 주소(`127.0.0.1:11434`)를 제공하고, Docker 실행 시 `docker-compose.yml`가 도커 주소(`host.docker.internal:11434`)로 덮어써 두 구성 모두 .env 수정 없이 실행 가능하다.

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

# 2. 환경변수 설정
cp .env.example .env

# 3. 초기 DB 구축 (최초 1회)
python3 scripts/init_db.py

# 4. 백엔드 실행
python3 app/main.py

# 5. 프론트엔드 실행 (새 터미널)
streamlit run frontend.py
```

브라우저에서 `http://localhost:8501` 접속

### Docker 실행

LLM(Ollama)만 호스트에서 실행하고, FastAPI 백엔드와 Streamlit UI는 Docker로 실행한다.

```bash
# 1. Ollama 실행 (호스트)
ollama pull gemma3:4b

# 2. 환경변수 설정
cp .env.example .env

# 3. 초기 DB 구축 (최초 1회)
docker compose run --rm app python3 scripts/init_db.py

# 4. 전체 실행
docker compose up --build
```

브라우저에서 `http://localhost:8501` 접속

---

## AI 모델 · 추론 환경 교체

### AI 모델 교체

LLM 모델   : .env의 `OLLAMA_MODEL` 교체
임베딩 모델 : .env의 `EMBEDDING_MODEL` 교체
리랭킹 모델 : .env의 `RERANKER_MODEL` 교체

### 추론 환경 교체 (Colab)

```python
# Colab 셀에서 실행
!curl -fsSL https://ollama.ai/install.sh | sh
!ollama pull gemma3:4b &
!pip install pyngrok
from pyngrok import ngrok
url = ngrok.connect(11434, "http")
print(url)  # 이 URL을 .env의 OLLAMA_URL로 교체
```

### 추론 디바이스 교체

| 환경 | 설정 | 비고 |
|------|------|------|
| CPU (기본) | `MODEL_DEVICE=cpu` | 추가 설치 없음 |
| GPU (CUDA) | `MODEL_DEVICE=cuda` | CUDA 버전 PyTorch 별도 설치 필요 |
| Google Colab | `MODEL_DEVICE=cuda` | T4 GPU 무료 제공, CUDA 기본 설치됨 |

---

## 환경변수 (.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BACKEND_HOST` | `0.0.0.0` | FastAPI 서버 호스트 |
| `BACKEND_PORT` | `8000` | FastAPI 서버 포트 |
| `BACKEND_URL` | `http://localhost:8000` | 백엔드 서버 주소 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | LLM 서버 주소 |
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama LLM 모델명 |
| `MODEL_DEVICE` | `cpu` | 추론 디바이스 (cpu / cuda) |
| `EMBEDDING_MODEL` | `dragonkue/BGE-m3-ko` | 임베딩 모델 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 리랭킹 모델 |
| `CACHE_TTL_DAYS` | `14` | 크롤링 및 가격 정보 캐시 TTL (일) |
| `GITHUB_TOKEN` | (선택) | GitHub API 인증 토큰 시간당 기본 60회 인증 시 5000회 |


---

## 테스트

### 단위 / 통합 테스트

```bash
python3 tests/test_embedder.py                  # 임베딩 모델 동작 테스트
python3 tests/test_reranker.py                  # 리랭커 점수 정렬 테스트
python3 tests/test_crawler.py                   # HuggingFace + GitHub 크롤러 테스트
python3 tests/test_vector_rerank.py             # 벡터 DB 검색 + 리랭킹 통합 테스트
python3 tests/test_pipeline.py                  # 전체 파이프라인 통합 테스트 (init_db 실행 필요)
```

### 서버 응답 테스트 (백엔드 실행 필요)

```bash
python3 tests/test_api.py   # 채팅 + 스트리밍 응답 확인
```

---

## 분류 정확도 평가

```bash
python3 scripts/evaluate_queries.py 
```

`tests/test_queries.json`에 정의된 10개의 쿼리를 각 5회 실행하여 분류 정확도와 평균 분류 속도를 평가.

| 쿼리 | 기대 카테고리 | 분류 경로 |
|------|-------------|-----------|
| 무료로 내가 수집한 자료들로 발표에 사용할 ppt를 만들어줘 | MODEL 또는 AGENT | LLM fallback |
| 그림을 잘 그리는 저렴한 AI 추천해줘 | MODEL | LLM fallback |
| 내 자금관리를 도와줄 비서가 필요해 | AGENT | 키워드 히트 |
| 내 물리학 수업 과제를 도와줄 AI가 필요해 | MODEL | LLM fallback |
| 하루에 물을 얼마나 마시면 좋아? | GENERAL | LLM fallback |
| 한국어 잘하는 무료 LLM 추천해줘 | MODEL | 키워드 히트 |
| 코딩 잘하는 ai는 뭐가 있어? | MODEL | LLM fallback |
| 매일 아침 뉴스를 요약해서 자동으로 보내줘 | AGENT | LLM fallback |
| 내 일정 관리를 도와줄 시스템이 필요해 | AGENT | LLM fallback |
| 파이썬이랑 자바 중 어느 것이 취업에 유리할까? | GENERAL | LLM fallback |

LLM의 비결정적 특성으로 인해 재실행 시 분류 결과가 달라질 수 있다.

---

## 기술적 도전과 해결

### 소형 LLM의 선택지 순서 편향

**문제:** 분류 정확도 평가 중 LLM fallback 발생 시 AGENT가 정답인 쿼리를 반복적으로 MODEL로 분류하는 현상 발생

**원인:** 소형 LLM의 프롬프트에서 먼저 등장한 선택지를 선호하는 경향(Primacy Bias)으로 인해 모호한 쿼리는 항상 첫 번째 선택지인 MODEL로 분류함

**시도한 해결책:**
1. MODEL, AGENT 키워드 개선 → 특정 쿼리만 해결
2. 쿼리 문구 변경 → 근본 원인 해결 아님
3. **프롬프트 선택지 순서 변경 (`AGENT`를 먼저 제시)** → 문제 해결

**교훈:** 소형 LLM일수록 프롬프트 구조가 결과를 좌우한다.

### FlagReranker → CrossEncoder 교체

**문제:** 초기 리랭킹 구현에 FlagEmbedding 라이브러리의 `FlagReranker`를 사용했으나, `test_reranker` 실행 중 `XLMRobertaTokenizer` 호환 문제로 오류 발생.

**원인:** FlagEmbedding이 요구하는 토크나이저 버전과 설치된 transformers 버전이 충돌.

**해결:** sentence-transformers의 `CrossEncoder`로 교체. 동일한 `BAAI/bge-reranker-v2-m3` 모델을 사용하면서 의존성 충돌 없이 동작 성공.

**교훈:** 라이브러리 선택 시 기능뿐 아니라 의존성 충돌 가능성도 고려해야 한다. 더 범용적인 라이브러리가 안정성 면에서 유리하다.

---

## 향후 개선 방향

| 항목 | 현황 | 비고 |
|------|------|------|
| 벤치마크 점수 표시 | 미구현 | 주요 벤치마크 사이트의 공개 API가 없거나 동적 렌더링 구조로 크롤링이 어려움 |
| 에이전트 프레임워크 자동 탐색 | 15개 수동 선정 | `agent` 키워드 검색 시 AI와 무관한 repo가 다수 포함되어 검증된 15개의 프레임워크 수동 선정. 향후 `topic:llm-agent` 필터 + star 기준으로 자동 탐색 예정 |