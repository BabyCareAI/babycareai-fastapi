# 이미지 검증 api
from pydantic import BaseModel, Field


class DiagnosisIdInput(BaseModel):
    diagnosis_id: str = Field(..., description="진단 ID (UUID)")


class ValidationResult(BaseModel):
    is_skin_related: bool = Field(..., description="피부 관련 이미지 여부")