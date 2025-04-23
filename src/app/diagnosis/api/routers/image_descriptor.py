# 이미지 상태 설명 API
from fastapi import APIRouter, HTTPException
from src.app.diagnosis.schemas.image_descriptor import DiagnosisIdInput, ImageDescriptionResult
from src.app.diagnosis.services.image_descriptor import image_descriptor_service

router = APIRouter(prefix="/api/v1/diagnosis", tags=["Image Description"])


@router.post("/image-description", response_model=ImageDescriptionResult)
async def describe_skin_image(input_data: DiagnosisIdInput):
    """
    진단 ID로 S3에서 이미지를 가져와 피부 상태를 설명합니다.
    
    Args:
        input_data: 진단 ID (UUID)
        
    Returns:
        ImageDescriptionResult: 이미지 설명 결과
    """
    try:
        result = await image_descriptor_service.describe_skin_image(input_data.diagnosis_id)
        return ImageDescriptionResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 상태 설명 중 오류가 발생했습니다: {str(e)}")