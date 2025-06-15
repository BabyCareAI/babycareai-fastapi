# 진단(RAG) 서비스
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse, DiseaseInfo
from src.app.utils.llm_client import get_text_embedding, create_llm_model
from src.app.utils.pinecone_client import retrieve_similar_diseases
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from src.app.domain.diagnosis.crud.diagnostician import create_diagnosis_result
from src.app.domain.diagnosis.utils.data_processor import flatten_and_join, extract_top_classification, convert_to_json_string
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import asyncio
from typing import Dict, Any, AsyncGenerator
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_diagnosis_chain():
    """
    진단을 위한 LCEL 체인을 생성합니다.
    """
    # 진단 모델
    diagnosis_model = create_llm_model(
        model_name="gpt-4.1-mini-2025-04-14", #gemini-2.0-flash-lite
        temperature=0,
        max_output_tokens=1000,
        provider="openai" #google
    )
    
    # 번역 모델
    translation_model = create_llm_model(
        model_name="gemini-2.0-flash-lite",
        temperature=0,
        max_output_tokens=1000,
        provider="google"
    )
    
    # 진단 프롬프트
    diagnosis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI diagnostic assistant powered by the knowledge of a pediatric specialist. Your mission is to synthesize user-provided information with retrieved medical knowledge to provide a careful, clear, and reassuring preliminary diagnosis for concerned parents."),
        ("human", """
        ### User-Provided Information
        - **Image Analysis:** {image_description}
        - **Key Symptoms:** {symptoms}
        - **Other Symptoms:** {other_symptom}
        - **Computer Vision Analysis Reference:** {classification}

        ### Retrieved Medical Knowledge from Vector DB
        {context}

        ---

        ### Analysis and Diagnosis Instructions
        
        **1. Chain of Thought - This is your internal thinking process. Do not include this section's title or steps in the final output.**
        Before generating the final response, perform an internal analysis by following the logical steps below.

        *   **Step 1: Information Synthesis:** Summarize the key features from the 'User-Provided Information'.
        *   **Step 2: Hypothesis & Comparison:** Compare the synthesized information against each disease candidate from the retrieved knowledge.
        *   **Step 3: Evaluation & Final Hypothesis:** Select the most probable disease and formulate the reasoning.

        ---

        **### Final Response Generation ###**
        Now, based on your internal 'Chain of Thought' conclusions, generate ONLY the final, parent-facing response. The response MUST strictly follow the format below and start EXACTLY with "- Final Diagnosis:".

        - **Final Diagnosis:** [The most likely diagnosis name]
        - **Diagnosis Reason:** [Logically explain the reasoning...]
        - **Severity:** [Assess the severity...]
        - **Need for Hospital Visit:** [Provide a clear guideline...]
        - **Home Care Instructions:** [Provide 2-3 specific methods...]

        **Guidelines for the Final Response:**
        - Maintain a warm, empathetic, yet professional tone.
        - Do not reference document numbers. Instead, use natural phrases like "According to medical information...".
        - Use language that reduces anxiety and builds trust.
        """)
    ])

    # 번역 프롬프트
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical translator specializing in Korean-English translation."),
        ("human", """
        다음은 AI 진단 결과입니다. 이 결과를 한국어로 번역해주세요.
        번역 시 다음 사항을 반드시 지켜주세요:
        1. 원본의 형식("- Final Diagnosis:", "- Diagnosis Reason:" 등)을 최대한 유지한 채 번역만 해주세요.
        2. 의학 용어는 정확하게 번역해주세요.
        4. 불필요한 설명이나 추가 내용은 포함하지 마세요.

        원본 진단 결과:
        {diagnosis_result}
        """)
    ])

    # 기본 진단 체인
    _diagnosis_chain = (
        diagnosis_prompt
        | diagnosis_model
        | StrOutputParser()
    )

    # 번역 체인
    # 이 체인은 'diagnosis_result' 키를 가진 딕셔너리를 입력으로 기대합니다.
    _translation_chain = (
        translation_prompt
        | translation_model
        | StrOutputParser()
    )

    # 최종 체인 구성:
    final_chain = _diagnosis_chain | RunnableParallel(
        diagnosis=RunnablePassthrough(),  # 원본 진단 결과 (문자열)
        translation=RunnablePassthrough() | (lambda text_input: {"diagnosis_result": text_input}) | _translation_chain # 번역된 결과 (문자열)
    )
    
    return final_chain

async def diagnose_with_rag(request: DiagnosisIdInput, top_k: int = 4, db: AsyncSession = None) -> AsyncGenerator[str, None]:
    """
    증상/부위/설명을 바탕으로 RAG 기반 진단을 수행합니다. (CoT 프롬프팅 적용)
    1. Redis에서 캐시된 데이터(이미지 설명, 증상, CV 분석 결과 등) 조회
    2. 모든 입력 정보를 종합하여 임베딩 생성 및 유사 질병 Top-K 검색
    3. LLM에 구조화된 정보와 명시적인 CoT 단계를 포함한 프롬프트로 질의
    4. 결과 스트리밍 반환 및 저장
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
            yield "임베딩 생성 실패"
            return

        # 3. 유사 질병 검색
        similar_diseases = retrieve_similar_diseases(input_text_for_embedding, top_k=top_k, input_embedding=input_embedding)
        if not similar_diseases:
            yield "유사 질병을 찾을 수 없습니다."
            return

        # 4. 컨텍스트 구성
        context_parts = []
        for i, d in enumerate(similar_diseases):
            try:
                metadata = {}
                if isinstance(d, dict):
                    metadata = d.get('metadata', {})
                elif hasattr(d, 'metadata'):
                    metadata = d.metadata

                disease = metadata.get('disease', '') if isinstance(metadata, dict) else ''
                disease_symptoms = metadata.get('symptoms', []) if isinstance(metadata, dict) else []
                skin_site = metadata.get('skin_site', []) if isinstance(metadata, dict) else []
                text = metadata.get('text', '') if isinstance(metadata, dict) else ''

                context_parts.append(
                    f"[{i+1}] Disease Name: {disease}\n"
                    f"Symptoms: {', '.join(disease_symptoms)}\n"
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

        # 전체 진단 결과를 저장할 변수
        full_diagnosis = ""

        # 체인 실행 및 스트리밍
        async for chunk in chain.astream(chain_input):
            if "translation" in chunk:
                full_diagnosis += chunk["translation"]
                yield chunk["translation"]

        # 6. 진단 결과 저장 (전체 결과를 저장)
        if db and full_diagnosis:
            await save_diagnosis_result(
                db=db,
                diagnosis_id=diagnosis_id,
                image_description=convert_to_json_string(image_description),
                symptoms=convert_to_json_string(symptoms),
                other_symptom=convert_to_json_string(other_symptom),
                classification=convert_to_json_string(classification_data),
                diagnosis=full_diagnosis,
                top_k_diseases=json.dumps(similar_diseases)
            )

    except Exception as e:
        logging.exception("진단 RAG 서비스 오류: %s", e)
        yield f"진단 중 오류 발생: {e}"

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