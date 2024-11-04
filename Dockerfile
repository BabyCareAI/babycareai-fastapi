# 베이스 이미지로 공식 Python 런타임 사용
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 현재 디렉터리의 모든 내용을 컨테이너의 /app 디렉터리에 복사
COPY . /app

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 앱을 위한 포트 노출
EXPOSE 8000

# 환경 변수 설정 (필요에 따라 추가)
ENV PYTHONUNBUFFERED=1

# FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
