from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 추가
from src.app.diagnosis.api.routers.diagnostician import router as nlp_router
from dotenv import load_dotenv
import os


load_dotenv()

app = FastAPI(root_path="/fastapi")

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
app.include_router(nlp_router)
