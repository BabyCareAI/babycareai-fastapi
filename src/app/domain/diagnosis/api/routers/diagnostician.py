# 진단(RAG) API 라우터
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.database import get_db
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse
from src.app.domain.diagnosis.services.diagnostician import diagnose_with_rag
from src.app.domain.diagnosis.services.diagnostician import save_diagnosis_result
from src.app.utils.redis_client import get_from_redis
import json

router = APIRouter(prefix="/api/v1/diagnosis", tags=["diagnosis"])

@router.post("/rag", response_model=DiagnosisResponse, summary="RAG 기반 피부 질환 진단 API")
async def diagnose_rag(
    request: DiagnosisIdInput,
    db: AsyncSession = Depends(get_db)
) -> DiagnosisResponse:
    """
    입력 증상/부위/설명 기반으로 RAG 시스템을 활용해 진단 결과를 반환합니다.
    """
    response = await diagnose_with_rag(request)
    if not response or not response.diagnosis:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="진단 실패")

    # Redis에서 각 데이터 조회
    diagnosis_id = request.diagnosis_id
    image_description = get_from_redis(f"image_description:{diagnosis_id}")
    symptoms = get_from_redis(f"symptoms:{diagnosis_id}")
    other_symptom = get_from_redis(f"other_symptom:{diagnosis_id}")
    classification = get_from_redis(f"classification:{diagnosis_id}")

    # 딕셔너리나 리스트 형태의 데이터를 JSON 문자열로 변환
    def convert_to_json_string(data):
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False)
        return data

    # 진단 결과 저장
    await save_diagnosis_result(
        db=db,
        diagnosis_id=diagnosis_id,
        image_description=convert_to_json_string(image_description),
        symptoms=convert_to_json_string(symptoms),
        other_symptom=convert_to_json_string(other_symptom),
        classification=convert_to_json_string(classification),
        diagnosis=response.diagnosis
    )
    return response