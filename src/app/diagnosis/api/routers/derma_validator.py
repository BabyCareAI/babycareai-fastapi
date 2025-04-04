# 이미지 검증 api
from fastapi import APIRouter, HTTPException
from src.app.diagnosis.schemas.derma_validator import DiagnosisIdInput, ValidationResult
from src.app.diagnosis.services.derma_validator import derma_validator_service

router = APIRouter(prefix="/api/v1/babyderm", tags=["Validation"])


@router.post("/validate", response_model=ValidationResult)
async def validate_skin_image(input_data: DiagnosisIdInput):
    """
    진단 ID로 S3에서 이미지를 가져와 피부 관련 이미지인지 검증합니다.
    
    Args:
        input_data: 진단 ID (UUID)
        
    Returns:
        ValidationResult: 검증 결과 (피부 관련 이미지 여부)
    """
    try:
        result = await derma_validator_service.validate_skin_image(input_data.diagnosis_id)
        return ValidationResult(
            is_skin_related=result["is_skin_related"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 검증 중 오류가 발생했습니다: {str(e)}")
