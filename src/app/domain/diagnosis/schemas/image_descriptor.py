# 이미지 상태 설명 API
from pydantic import BaseModel, Field


class DiagnosisIdInput(BaseModel):
    diagnosis_id: str = Field(..., description="진단 ID (UUID)")


class ImageDescriptionResult(BaseModel):
    description: str = Field(..., description="이미지에 나타난 피부의 상태나 특징에 대한 설명")
    body_part: str | None = Field(None, alias="bodyPart", description="이미지의 신체 부위 정보 (S3 메타데이터)")

    class Config:
        allow_population_by_field_name = True