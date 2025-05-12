# 기타 증상 입력 api
from pydantic import BaseModel, Field

class OtherSymptomInput(BaseModel):
    diagnosis_id: str = Field(..., description="진단 ID")
    other_symptom_text: str = Field(..., description="기타 증상 내용")

class OtherSymptomResult(BaseModel):
    diagnosis_id: str
    summarized_translation: str
    is_valid: bool
    message: str = ""