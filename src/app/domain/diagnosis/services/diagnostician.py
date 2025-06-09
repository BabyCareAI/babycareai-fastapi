# 진단(RAG) 서비스
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse, DiseaseInfo
from src.app.utils.llm_client import get_text_embedding, create_diagnosis_chain
from src.app.utils.pinecone_client import retrieve_similar_diseases
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from src.app.domain.diagnosis.crud.diagnostician import create_diagnosis_result
from src.app.domain.diagnosis.utils.data_processor import flatten_and_join, extract_top_classification, convert_to_json_string
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import asyncio
from typing import Dict, Any
import json

async def diagnose_with_rag(request: DiagnosisIdInput, top_k: int = 4, db: AsyncSession = None) -> DiagnosisResponse:
    """
    증상/부위/설명을 바탕으로 RAG 기반 진단을 수행합니다. (CoT 프롬프팅 적용)
    1. Redis에서 캐시된 데이터(이미지 설명, 증상, CV 분석 결과 등) 조회
    2. 모든 입력 정보를 종합하여 임베딩 생성 및 유사 질병 Top-K 검색
    3. LLM에 구조화된 정보와 명시적인 CoT 단계를 포함한 프롬프트로 질의
    4. 결과 반환 및 저장
    """
    try:
        # 1. diagnosis_id 기반 redis에서 데이터 조회
        from src.app.utils.redis_client import get_from_redis
        
        diagnosis_id = request.diagnosis_id
        
        # Redis에서 데이터 조회
        image_description = get_from_redis(f"image_description:{diagnosis_id}")
        symptoms = get_from_redis(f"symptoms:{diagnosis_id}")
        other_symptom = get_from_redis(f"other_symptom:{diagnosis_id}")
        classification_data = get_from_redis(f"classification:{diagnosis_id}")
        top_classification = extract_top_classification(classification_data)

        # 2. 임베딩 생성을 위한 텍스트 조합
        input_values = [image_description, symptoms, other_symptom, top_classification]
        input_text_for_embedding = flatten_and_join(input_values)
        input_text_for_embedding = str(input_text_for_embedding)

        # 2. 임베딩 생성
        input_embedding = await get_text_embedding(input_text_for_embedding)
        if not input_embedding or (hasattr(input_embedding, '__len__') and len(input_embedding) == 0):
            logging.error("임베딩 생성 실패: input_embedding is None or empty")
            return DiagnosisResponse(
                diagnosis="임베딩 생성 실패",
                top_k_diseases=[]
            )

        # 3. 유사 질병 검색
        similar_diseases = retrieve_similar_diseases(input_text_for_embedding, top_k=top_k, input_embedding=input_embedding)
        if not similar_diseases:
            return DiagnosisResponse(
                diagnosis="유사 질병을 찾을 수 없습니다.",
                top_k_diseases=[]
            )

        # 4. 컨텍스트 구성
        context_parts = []
        for i, d in enumerate(similar_diseases):
            try:
                # 메타데이터 추출
                metadata = {}
                if isinstance(d, dict):
                    metadata = d.get('metadata', {})
                elif hasattr(d, 'metadata'):
                    metadata = d.metadata

                # 필수 필드 추출
                disease = metadata.get('disease', '') if isinstance(metadata, dict) else ''
                symptoms = metadata.get('symptoms', []) if isinstance(metadata, dict) else []
                skin_site = metadata.get('skin_site', []) if isinstance(metadata, dict) else []
                text = metadata.get('text', '') if isinstance(metadata, dict) else ''

                context_parts.append(
                    f"[{i+1}] Disease Name: {disease}\n"
                    f"Symptoms: {', '.join(symptoms)}\n"
                    f"Skin Site: {', '.join(skin_site)}\n"
                    f"Disease Information: {text}\n"
                )
            except Exception as e:
                logging.error(f"컨텍스트 생성 중 오류 발생: {e}")
                continue
                
        context = "\n\n".join(context_parts)

        # 5. LCEL 체인 생성 및 실행
        chain = create_diagnosis_chain()
        
        # 입력 데이터 구성
        chain_input = {
            "image_description": image_description if image_description else "No information provided",
            "symptoms": symptoms if symptoms else "No information provided",
            "other_symptom": other_symptom if other_symptom else "No information provided",
            "classification": top_classification if top_classification else "Below threshold or no information provided",
            "context": context
        }

        # 체인 실행
        result = await chain.ainvoke(chain_input)
        
        # 6. 진단 결과 저장
        if db:
            await save_diagnosis_result(
                db=db,
                diagnosis_id=diagnosis_id,
                image_description=convert_to_json_string(image_description),
                symptoms=convert_to_json_string(symptoms),
                other_symptom=convert_to_json_string(other_symptom),
                classification=convert_to_json_string(classification_data),
                diagnosis=result["translation"],
                top_k_diseases=json.dumps(similar_diseases)
            )

        return DiagnosisResponse(
            diagnosis=result["translation"]
        )

    except Exception as e:
        logging.exception("진단 RAG 서비스 오류: %s", e)
        return DiagnosisResponse(
            diagnosis=f"진단 중 오류 발생: {e}"
        )

async def save_diagnosis_result(
    db: AsyncSession,
    diagnosis_id: str,
    image_description: str | None,
    symptoms: str | None,
    other_symptom: str | None,
    classification: str | None,
    diagnosis: str,
    top_k_diseases: str
):
    data = DiagnosisResultsCreate(
        diagnosis_id=diagnosis_id,
        image_description=image_description,
        symptoms=symptoms,
        other_symptom=other_symptom,
        classification=classification,
        diagnosis=diagnosis,
        top_k_diseases=top_k_diseases
    )
    return await create_diagnosis_result(db, data)