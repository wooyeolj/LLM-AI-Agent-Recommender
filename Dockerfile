FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치 (크롤링 도구 필요)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# FastAPI 기본 실행 (docker-compose에서 오버라이드 가능)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]