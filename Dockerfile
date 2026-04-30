#docker compose up --build 실행 시 컨테이너 환경 구성

FROM python:3.11-slim

#기준 경로
WORKDIR /app

# requirements.txt를 소스코드보다 먼저 복사 > 코드 수정 시 pip install 캐시 재사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#로그 실시간 확인
ENV PYTHONUNBUFFERED=1

# docker-compose 없이 docker run 으로 직접 실행 시 기본 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]