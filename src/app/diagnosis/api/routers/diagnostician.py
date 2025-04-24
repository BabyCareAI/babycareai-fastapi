# 진단(RAG) API 라우터
from fastapi import APIRouter, HTTPException, status
from src.app.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse
from src.app.diagnosis.services.diagnostician import diagnose_with_rag

router = APIRouter(prefix="/api/v1/diagnosis", tags=["diagnosis"])

@router.post("/rag", response_model=DiagnosisResponse, summary="RAG 기반 피부 질환 진단 API")
async def diagnose_rag(request: DiagnosisIdInput) -> DiagnosisResponse:
    """
    입력 증상/부위/설명 기반으로 RAG 시스템을 활용해 진단 결과를 반환합니다.
    """
    response = await diagnose_with_rag(request)
    if not response or not response.diagnosis:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="진단 실패")
    return response