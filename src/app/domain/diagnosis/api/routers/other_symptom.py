# 기타 증상 입력 api
from fastapi import APIRouter
from src.app.domain.diagnosis.schemas.other_symptom import OtherSymptomInput, OtherSymptomResult
from src.app.domain.diagnosis.services.other_symptom import process_other_symptom

router = APIRouter(prefix="/api/v1/diagnosis", tags=["진단"])

@router.post("/other-symptom", response_model=OtherSymptomResult,  summary="기타 증상 입력")
async def input_other_symptom(data: OtherSymptomInput) -> OtherSymptomResult:
    """
    진단 ID를 입력받고 기타 증상 string을 처리합니다.
    (이후 검증, 요약, 번역을 수행합니다.)

    Args:
        data: 진단 ID와 기타 증상 string
        
    Returns:
        OtherSymptomResult: 처리 결과
    """
    return await process_other_symptom(data)