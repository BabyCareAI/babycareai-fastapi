# 진단(RAG) API 라우터
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.database import get_db
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse
from src.app.domain.diagnosis.services.diagnostician import diagnose_with_rag
from src.app.domain.diagnosis.services.diagnostician import save_diagnosis_result
from src.app.utils.redis_client import get_from_redis
from src.app.domain.diagnosis.utils.data_processor import convert_to_json_string

router = APIRouter(prefix="/api/v1/diagnosis", tags=["진단"])

@router.post("/rag", response_model=DiagnosisResponse, summary="최종 진단 (RAG)")
async def diagnose_rag(
    request: DiagnosisIdInput,
    db: AsyncSession = Depends(get_db)
) -> DiagnosisResponse:
    """
    입력 증상/부위/설명 기반으로 RAG 시스템을 활용해 진단 결과를 반환합니다.
    """
    response = await diagnose_with_rag(request, db=db)
    if not response or not response.diagnosis:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="진단 실패")
    return response