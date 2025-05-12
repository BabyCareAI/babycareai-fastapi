# 진단(RAG) API 스키마
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class DiagnosisIdInput(BaseModel):
    diagnosis_id: str = Field(..., description="진단 ID (UUID)")

class DiseaseInfo(BaseModel):
    disease: str
    symptoms: List[str]
    skin_site: List[str]
    disease_information: str
    similarity: Optional[float] = Field(None, description="입력과의 유사도 점수")

class DiagnosisResponse(BaseModel):
    diagnosis: str = Field(..., description="최종 LLM 진단 결과")
    top_k_diseases: List[DiseaseInfo] = Field(..., description="유사 질병 Top-K 정보")
    input_embedding: Optional[Any] = Field(None, description="(디버그용) 입력 임베딩")
    retrieved_embeddings: Optional[Any] = Field(None, description="(디버그용) 검색된 임베딩")

class DiagnosisResultsCreate(BaseModel):
    id: str = Field(..., description="진단 ID (UUID)")
    image_description: Optional[str] = Field(None, description="이미지 설명(재료)")
    symptoms: Optional[str] = Field(None, description="증상 정보(재료)")
    other_symptom: Optional[str] = Field(None, description="기타 증상 정보(재료)")
    classification: Optional[str] = Field(None, description="딥러닝 모델 이미지 분류 결과(재료)")
    diagnosis: str = Field(..., description="최종 진단 결과")

class DiagnosisResultsRead(DiagnosisResultsCreate):
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True