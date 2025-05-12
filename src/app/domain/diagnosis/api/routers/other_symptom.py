# 기타 증상 입력 api
from fastapi import APIRouter
from src.app.domain.diagnosis.schemas.other_symptom import OtherSymptomInput, OtherSymptomResult
from src.app.domain.diagnosis.services.other_symptom import process_other_symptom

router = APIRouter(prefix="/other-symptom", tags=["Other Symptom"])

@router.post("/", response_model=OtherSymptomResult)
async def input_other_symptom(data: OtherSymptomInput) -> OtherSymptomResult:
    """
    진단 ID를 입력받고 기타 증상 string을 처리합니다.
    
    Args:
        data: 진단 ID와 기타 증상 string
        
    Returns:
        OtherSymptomResult: 처리 결과
    """
    return await process_other_symptom(data)