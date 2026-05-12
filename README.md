# LLM & AI Agent 추천 시스템

2022년 ChatGPT 출시 이후 수백 개의 LLM과 에이전트 프레임워크가 출시되면서, 목적에 맞는 ai 도구를 선택하는 것이 하나의 과제가 되었다.

이 프로젝트는 사용자의 자연어 질문을 기반으로 HuggingFace, OpenRouter, GitHub api를 이용해 실시간으로 데이터를 수집하고, RAG (검색 증강 생성) 기술로 최적의 LLM 모델 또는 AI 에이전트 프레임워크를 제공한다.

---

## 목차

- [프로젝트 소개](#프로젝트-소개)
- [프로젝트 데모](#프로젝트-데모)
- [모델 선택 이유](#모델-선택-이유)
- [아키텍처](#아키텍처)
- [설계 결정](#설계-결정)
- [크롤링 전략](#크롤링-전략)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [배포 설계](#배포-설계)
- [실행 방법](#실행-방법)
- [LLM · 엔진 교체 방법](#llm--엔진-교체-방법)
- [환경변수](#환경변수-env)
- [테스트](#테스트)
- [분류 정확도 평가](#분류-정확도-평가)
- [기술적 도전과 해결](#기술적-도전과-해결)
- [향후 개선 방향](#향후-개선-방향)

---

## 프로젝트 소개

이 프로젝트는 AI 도구의 선택 장벽을 낮추기 위해 사용자의 자연어 질문을 기반으로 최적의 AI 도구를 찾는다.
사용자의 질문이 입력되면 **분류–검색–크롤링–리랭킹–생성**의 5단계 RAG 파이프라인이 동작한다.

1. 사용자 질문을 **키워드 분류 + LLM fallback(gemma3:4b)** 으로 자동 분류 (MODEL / AGENT / GENERAL)
2. 분류 결과에 맞는 데이터를 **HuggingFace, OpenRouter, GitHub API**에서 실시간 수집
3. 질문을 **임베딩**(BGE-m3-ko)해 ChromaDB에서 유사 후보 20개 검색
4. **리랭킹**(BGE-reranker-v2-m3)으로 쿼리 관련도를 직접 비교해 상위 3~5개 선별
5. **Ollama(gemma3:4b)**가 선별된 결과를 컨텍스트로 자연어 답변을 SSE 스트리밍으로 제공

외부 유료 API 없이 완전 로컬 실행이 가능하며, Docker와 로컬 환경을 모두 지원한다.

---

## 프로젝트 데모

> 스크린샷 / GIF 추가 예정

---

## 모델 선택 이유

### Gemma3:4b — 로컬 추론 LLM

| 항목 | 내용 |
|------|------|
| 크기 | 4B 파라미터 — CPU 또는 Google Colab 무료 환경에서 실행 가능 |
| 한국어 | 소형 모델 대비 한국어 이해 및 생성 품질 우수 |
| 추론 | 구조화된 답변 생성과 분류 태스크에서 안정적 |
| 선택 이유 | 외부 API 비용 없이 완전 로컬 실행을 유지하면서도 실용적인 품질 확보 |

### dragonkue/BGE-m3-ko — 한국어 임베딩

| 항목 | 내용 |
|------|------|
| 베이스 | BAAI/BGE-M3 (다국어 임베딩 SOTA 모델) |
| 특화 | 한국어 데이터로 추가 파인튜닝 |
| 선택 이유 | 한국어 쿼리로 영문 모델 설명을 검색할 때 의미 매칭 정확도 향상 |

### BAAI/bge-reranker-v2-m3 — 리랭킹

| 항목 | 내용 |
|------|------|
| 구조 | Cross-Encoder — 쿼리와 문서 쌍을 직접 비교해 정밀한 관련도 점수 산출 |
| 다국어 | 한국어 포함 다국어 지원 |
| 선택 이유 | 벡터 검색(Bi-Encoder)의 한계를 보완해 상위 결과의 정렬 정확도를 높임 |

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

## 설계 결정

주요 기술 선택의 배경과 대안 대비 강점을 정리합니다.

| 결정 | 대안 | 선택 이유 및 강점 |
|------|------|-----------------|
| **2단계 검색** ChromaDB(20개) → CrossEncoder(5개) | CrossEncoder만 전체 DB에 적용 | CrossEncoder는 쿼리-문서 쌍을 하나씩 비교하므로 문서 수에 비례해 시간이 증가. ChromaDB로 후보를 먼저 좁혀 정밀도와 속도를 동시에 확보 |
| **분류 구조** 키워드 1차 → LLM fallback | 모든 쿼리를 LLM으로 분류 | "LLM", "에이전트" 등 명확한 키워드가 있는 쿼리는 수ms 내 분류 완료. LLM 호출(3~10초)은 모호한 경우에만 발생해 평균 응답 시간 단축 |
| **SSE 스트리밍** | WebSocket | 이 서비스는 서버→클라이언트 단방향 스트리밍만 필요. SSE는 HTTP 표준으로 FastAPI `StreamingResponse`로 간단히 구현되며, WebSocket 대비 연결 관리 복잡도가 낮음 |
| **파일 기반 캐시** (JSON) | Redis / 인메모리 dict | 인메모리 dict는 앱 재시작 시 14일 TTL이 초기화되는 버그 유발. Redis는 별도 서버가 필요해 Docker 구성이 복잡해짐. JSON 파일은 추가 인프라 없이 재시작 후에도 TTL 유지 |
| **싱글톤 모델 로드** (임베딩 · 리랭커) | 요청마다 모델 로드 | BGE-m3-ko, CrossEncoder 각각 초기 로드에 10~30초 소요. 앱 시작 시 한 번만 로드하는 싱글톤 구조로 이후 모든 요청에서 추가 지연 없음 |
| **컬렉션 분리** llm_items / agent_items | 단일 컬렉션 + metadata 필터 | 컬렉션 분리로 쿼리 시 검색 범위 자체가 좁아져 타입이 다른 문서가 유사도 계산에 포함되지 않음. 단일 컬렉션 + where 필터는 전체 스캔 후 필터링하는 구조 |
| **한국어 태그 보강** (`_PIPELINE_KO`) | 영어 원문 그대로 저장 | "그림 그려줘"와 "text-to-image" 간 임베딩 거리가 큼. 한국어 동의어("이미지생성 그림생성")를 문서에 주입해 한국어 쿼리와 영어 메타데이터 간 크로스링구얼 매칭 정확도 향상 |
| **Ollama 로컬 LLM** | OpenAI / Claude API | API 키 없이 누구나 즉시 실행 가능해 포트폴리오 재현성 확보. `OLLAMA_URL` 환경변수 하나로 클라우드 LLM 교체도 가능해 유연성 유지 |

---

## 크롤링 전략

| 소스 | 수집 항목 | 용도 |
|------|----------|------|
| **HuggingFace API** | 모델명, 설명, 다운로드수, 좋아요, 태그, 출시일 | LLM/이미지생성 등 모델 메타데이터 |
| **OpenRouter API** | 가격(입출력 per 1M tokens), 컨텍스트 길이 | 상업 모델 가격 정보 |
| **GitHub API** | 별점, 업데이트일, 지원 LLM, 로컬 지원 여부 | 에이전트 프레임워크 인기도 |

- 키워드가 HuggingFace pipeline tag(`text-to-image`, `translation` 등)이면 `pipeline_tag` 파라미터로 정확히 필터링
- 동일 키워드는 14일간 재크롤링 하지 않음 (파일 기반 캐시 — 앱 재시작 후에도 TTL 유지)
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
│   └── data_processor.py      # 크롤링 데이터 → ChromaDB 저장
├── scripts/
│   ├── init_db.py             # 초기 DB 구축 (최초 1회)
│   ├── run_crawler.py         # 수동 크롤링 트리거
│   └── evaluate_queries.py    # 쿼리 분류 정확도 평가
├── tests/                     # 단위/통합 테스트
├── frontend.py                # Streamlit UI (SSE 스트리밍 소비)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 배포 설계 — 로컬 · Docker 동시 지원

Ollama(LLM)는 항상 **호스트 머신**에서 실행하고, 백엔드(FastAPI)와 UI(Streamlit)는 **로컬 또는 Docker** 중 선택해 실행할 수 있다.

| 구성 | Ollama | FastAPI + Streamlit |
|------|--------|---------------------|
| 로컬 실행 | 호스트 `127.0.0.1:11434` | 호스트 직접 실행 |
| Docker 실행 | 호스트 `host.docker.internal:11434` | 컨테이너 내부 실행 |

두 구성 모두 브라우저에서 `http://localhost:8501`로 동일하게 접속한다.

**핵심 설계 포인트**
- `docker-compose.yml`의 `extra_hosts: host.docker.internal:host-gateway` — 컨테이너에서 호스트 Ollama에 접근 가능하게 함
- `environment: OLLAMA_URL=http://host.docker.internal:11434` — `.env`의 로컬 주소를 Docker용으로 자동 덮어씀
- 결과적으로 `.env` 수정 없이 `docker compose up` / `python3 app/main.py` 양쪽 모두 즉시 실행 가능

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

## LLM · 엔진 교체 방법

### LLM 교체 (Ollama 엔드포인트)

`OLLAMA_URL` 환경변수만 바꾸면 다른 LLM 엔드포인트에 연결할 수 있다.

| 시나리오 | 설정 |
|---------|------|
| 로컬 Ollama (기본) | `OLLAMA_URL=http://127.0.0.1:11434` |
| 다른 Ollama 모델 | `OLLAMA_MODEL=llama3.2` 로 변경 |
| Google Colab + ngrok | `OLLAMA_URL=https://xxxx.ngrok-free.app` |
| 원격 서버 Ollama | `OLLAMA_URL=http://[서버IP]:11434` |

**Colab에서 Ollama 실행하는 방법:**
```python
# Colab 셀에서 실행
!curl -fsSL https://ollama.ai/install.sh | sh
!ollama pull gemma3:4b &
!pip install pyngrok
from pyngrok import ngrok
url = ngrok.connect(11434, "http")
print(url)  # 이 URL을 OLLAMA_URL에 설정
```

### 추론 엔진 교체 (임베딩 · 리랭킹)

임베딩과 리랭커 모델의 실행 디바이스를 `.env`에서 변경할 수 있다.

| 환경 | 설정 | 비고 |
|------|------|------|
| CPU (기본) | `MODEL_DEVICE=cpu` | 추가 설치 없음 |
| GPU (CUDA) | `MODEL_DEVICE=cuda` | CUDA 버전 PyTorch 별도 설치 필요 |
| Google Colab | `MODEL_DEVICE=cuda` | T4 GPU 무료 제공, CUDA 기본 설치됨 |

---

## 환경변수 (.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | LLM 서버 주소 (로컬/Colab/원격) |
| `OLLAMA_MODEL` | `gemma3:4b` | 사용할 Ollama 모델명 |
| `EMBEDDING_MODEL` | `dragonkue/BGE-m3-ko` | 임베딩 모델 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 리랭킹 모델 |
| `MODEL_DEVICE` | `cpu` | 추론 디바이스 (cpu / cuda) |
| `GITHUB_TOKEN` | *(선택)* | GitHub API 인증 토큰 (없으면 60req/h) |
| `BACKEND_PORT` | `8000` | FastAPI 서버 포트 |

---

## 테스트

### 단위 / 통합 테스트

```bash
python3 tests/test_embedder.py                  # 임베딩 모델 동작 확인
python3 tests/test_reranker.py                  # 리랭커 점수 정렬 확인
python3 tests/test_crawler.py                   # HuggingFace + GitHub 크롤러 확인
python3 tests/test_vector_rerank.py             # 벡터 DB 검색 + 리랭킹 통합
python3 tests/test_pipeline.py                  # 전체 파이프라인 통합 테스트 (init_db 선행 필요)
```

### API 엔드포인트 테스트 (백엔드 실행 후)

```bash
python3 tests/test_api.py   # /api/chat + /api/chat/stream 엔드포인트 검증
```

---

## 분류 정확도 평가

```bash
python3 scripts/evaluate_queries.py   # test_queries.json 기반 분류기 평가
```

`tests/test_queries.json`에 정의된 10개 쿼리를 각 5회 실행하고 카테고리 분류 정확도와 평균 분류 속도를 검증한다.

| 쿼리 | 허용 카테고리 | 분류 경로 |
|------|-------------|-----------|
| 무료로 내가 수집한 자료들로 발표에 사용할 ppt를 만들어줘 | MODEL 또는 AGENT | LLM fallback |
| 그림을 잘 그리는 저렴한 AI 추천해줘 | MODEL | 키워드 직접 히트 |
| 내 자금관리를 도와줄 비서가 필요해 | AGENT | 키워드 직접 히트 |
| 내 물리학 수업 과제를 도와줄 AI가 필요해 | MODEL | LLM fallback |
| 하루에 물을 얼마나 마시면 좋아? | GENERAL | AI 무관 일반 질문 |
| 한국어 잘하는 무료 LLM 추천해줘 | MODEL | 키워드 직접 히트 |
| 코딩 잘하는 ai는 뭐가 있어? | MODEL | LLM fallback |
| 매일 아침 뉴스를 요약해서 자동으로 보내줘 | AGENT | LLM fallback |
| 내 스케줄을 도와줄 시스템이 필요해 | AGENT | LLM fallback |
| 파이썬이랑 자바 중 어느 것이 취업에 유리할까? | GENERAL | AI 무관 일반 질문 |

> LLM fallback 경로 쿼리는 gemma3:4b의 비결정적 특성으로 인해 재실행 시 결과가 달라질 수 있다.

---

## 기술적 도전과 해결

### Primacy Bias — 소형 LLM의 선택지 순서 편향

**문제:** 키워드로 분류되지 않는 쿼리를 gemma3:4b에게 `"MODEL 또는 AGENT 중 무엇인가요?"` 형태로 물었을 때, AGENT가 정답인 쿼리도 반복적으로 MODEL로 분류되는 현상이 발생했다.

**원인:** 소형 LLM은 프롬프트에서 **먼저 등장한 선택지를 선호하는 경향(Primacy Bias)**이 있다. MODEL이 첫 번째로 제시되어 있어 모호한 쿼리는 항상 MODEL로 수렴했다.

**시도한 해결책:**
1. AGENT 관련 키워드 추가 → 특정 쿼리만 해결, 일반화 불가
2. 쿼리 문구 변경 → 근본 원인 해결 아님
3. **프롬프트 선택지 순서 변경 (`AGENT`를 먼저 제시)** → 키워드 추가 없이 10/10 달성 ✓

**교훈:** 소형 LLM을 분류기로 활용할 때는 선택지 순서가 결과에 직접 영향을 미친다. 프롬프트 엔지니어링이 모델 파인튜닝만큼 중요할 수 있다.

### FlagReranker → CrossEncoder 교체

**문제:** 초기 리랭킹 구현에 FlagEmbedding 라이브러리의 `FlagReranker`를 사용했으나, `XLMRobertaTokenizer` 호환 문제로 실행 자체가 불가능했다.

**원인:** FlagEmbedding이 내부적으로 요구하는 토크나이저 버전과 설치된 transformers 버전이 충돌했다.

**해결:** sentence-transformers의 `CrossEncoder`로 교체. 동일한 `BAAI/bge-reranker-v2-m3` 모델을 사용하면서 의존성 충돌 없이 동작했다. requirements.txt에서 FlagEmbedding 의존성도 함께 제거했다.

**교훈:** 라이브러리 선택 시 기능뿐 아니라 의존성 충돌 가능성도 고려해야 한다. 같은 모델을 여러 라이브러리에서 지원하는 경우, 더 범용적인 라이브러리가 안정성 면에서 유리하다.

---

## 향후 개선 방향

| 항목 | 현황 | 비고 |
|------|------|------|
| 벤치마크 점수 표시 | 미구현 | Chatbot Arena(LMSYS) 등 공개 API 미제공 — gated dataset, JS SPA 구조로 크롤링 불가. API 공개 시 추가 예정 |
| 에이전트 프레임워크 자동 발굴 | 15개 수동 선정 | GitHub 검색 API로 `"agent"` 키워드를 치면 AI와 무관한 수천 개 repo가 섞여 자동 분류가 어렵습니다. 현재는 검증된 15개를 수동 선정해 품질을 보장하고, 별점·업데이트 정보는 실시간으로 수집합니다. 향후 `topic:llm-agent` 필터 + star 기준으로 후보를 자동 발굴하고 검증 단계를 추가하는 방식으로 확장 예정 |
