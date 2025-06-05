from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 추가
from src.app.domain.diagnosis.api.routers.image_validator import router as image_validation
from src.app.domain.diagnosis.api.routers.image_descriptor import router as image_description
from src.app.domain.diagnosis.api.routers.other_symptom import router as other_symptom
from src.app.domain.diagnosis.api.routers.diagnostician import router as diagnostician
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

app = FastAPI(root_path="/fastapi", title="babycareai API", version="0.1")

cors_origins = os.getenv("CORS_ORIGINS", "").split(",")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # .env에서 가져온 도메인 목록
    allow_credentials=True,  # 쿠키와 인증 정보를 포함한 요청을 허용
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # 허용할 HTTP 메서드
    allow_headers=["*"],  # 모든 헤더를 허용
)

# 라우터 포함
app.include_router(image_validation)
app.include_router(image_description)
app.include_router(other_symptom)
app.include_router(diagnostician)

# Prometheus 메트릭 수집기 설정
Instrumentator().instrument(app).expose(app)