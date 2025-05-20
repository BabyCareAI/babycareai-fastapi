FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 전체 프로젝트 복사 (requirements.txt 포함)
COPY . /app/

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# PYTHONPATH 설정 (src를 모듈로 인식하게 함)
ENV PYTHONPATH=/app/src

# 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# FastAPI 앱 실행
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]