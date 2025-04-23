# 이미지 상태 설명 API
from pydantic import BaseModel, Field


class DiagnosisIdInput(BaseModel):
    diagnosis_id: str = Field(..., description="진단 ID (UUID)")


class ImageDescriptionResult(BaseModel):
    description: str = Field(..., description="이미지에 나타난 피부의 상태나 특징에 대한 설명")