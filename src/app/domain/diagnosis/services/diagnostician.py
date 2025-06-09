# 진단(RAG) 서비스
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse, DiseaseInfo
from src.app.utils.llm_client import get_text_embedding, query_llm_with_context
from src.app.utils.pinecone_client import retrieve_similar_diseases
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from src.app.domain.diagnosis.crud.diagnostician import create_diagnosis_result
from src.app.domain.diagnosis.utils.data_processor import flatten_and_join, extract_top_classification
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import asyncio

async def diagnose_with_rag(request: DiagnosisIdInput, top_k: int = 4, db: AsyncSession = None) -> DiagnosisResponse:
    """
    증상/부위/설명을 바탕으로 RAG 기반 진단을 수행합니다. (CoT 프롬프팅 적용)
    1. Redis에서 캐시된 데이터(이미지 설명, 증상, CV 분석 결과 등) 조회
    2. 모든 입력 정보를 종합하여 임베딩 생성 및 유사 질병 Top-K 검색
    3. LLM에 구조화된 정보와 명시적인 CoT 단계를 포함한 프롬프트로 질의
    4. 결과 반환
    """
    try:
        # 1. diagnosis_id 기반 redis에서 데이터 조회
        from src.app.utils.redis_client import get_from_redis
        diagnosis_id = request.diagnosis_id
        
        # 각 데이터를 개별 변수로 명확하게 받습니다.
        image_description = get_from_redis(f"image_description:{diagnosis_id}")
        symptoms = get_from_redis(f"symptoms:{diagnosis_id}")
        other_symptom = get_from_redis(f"other_symptom:{diagnosis_id}")
        classification_data = get_from_redis(f"classification:{diagnosis_id}")

        # classification에서 가장 높은 확률의 클래스 추출
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
                # input_embedding=None,
                # retrieved_embeddings=None
            )

        # 3. 유사 질병 검색
        similar_diseases = retrieve_similar_diseases(input_text_for_embedding, top_k=top_k, input_embedding=input_embedding)
        if not similar_diseases:
            return DiagnosisResponse(
                diagnosis="유사 질병을 찾을 수 없습니다.",
                top_k_diseases=[]
                # input_embedding=input_embedding,
                # retrieved_embeddings=None
            )

         # 4. LLM에 컨텍스트와 함께 질의
        context = "\n\n".join([
            f"[{i+1}] Disease Name: {d.get('metadata', {}).get('disease', '')}\n"
            f"Symptoms: {', '.join(d.get('metadata', {}).get('symptoms', []))}\n"
            f"Skin Site: {', '.join(d.get('metadata', {}).get('skin_site', []))}\n"
            f"Disease Information: {', '.join(d.get('metadata', {}).get('text', []))}\n"
            for i, d in enumerate(similar_diseases)
        ])
        user_prompt = f"""
        You are an AI diagnostic assistant powered by the knowledge of a pediatric specialist. Your mission is to synthesize user-provided information with retrieved medical knowledge to provide a careful, clear, and reassuring preliminary diagnosis for concerned parents.

        ### User-Provided Information
        - **Image Analysis:** {image_description if image_description else "No information provided"}
        - **Key Symptoms:** {symptoms if symptoms else "No information provided"}
        - **Other Symptoms:** {other_symptom if other_symptom else "No information provided"}
        - **Computer Vision Analysis Reference:** {top_classification if top_classification else "Below threshold or no information provided"}

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
        """

        diagnosis = await query_llm_with_context(user_prompt)
        
        # 한국어 번역을 위한 프롬프트
        translation_prompt = f"""
        다음은 AI 진단 결과입니다. 이 결과를 한국어로 번역해주세요.
        번역 시 다음 사항을 반드시 지켜주세요:
        1. 원본의 형식("- Final Diagnosis:", "- Diagnosis Reason:" 등)을 그대로 유지해주세요.
        2. 의학 용어는 정확하게 번역해주세요.
        3. 친절하고 공감하는 말투로 번역해주세요.
        4. 불필요한 설명이나 추가 내용은 포함하지 마세요.

        원본 진단 결과:
        {diagnosis}
        """

        # 한국어 번역 수행
        korean_diagnosis = await query_llm_with_context(translation_prompt)
        
        # 5. 결과 조합 및 반환
        top_k_diseases = [
            DiseaseInfo(
                disease=d.get('metadata', {}).get('disease', ''),
                symptoms=d.get('metadata', {}).get('symptoms', []),
                skin_site=d.get('metadata', {}).get('skin_site', []),
                disease_information=d.get('metadata', {}).get('text', []),  #d.get('content', '')[:100],
                similarity=d.get('score', None),
            )
            for d in similar_diseases
        ]
        return DiagnosisResponse(
            diagnosis=korean_diagnosis,
            top_k_diseases=top_k_diseases
            # input_embedding=input_embedding,
            # retrieved_embeddings=[d.get('embedding') for d in similar_diseases]
        )
    except Exception as e:
        logging.exception("진단 RAG 서비스 오류: %s", e)
        return DiagnosisResponse(
            diagnosis=f"진단 중 오류 발생: {e}",
            top_k_diseases=[]
            # input_embedding=None,
            # retrieved_embeddings=None
        )

async def save_diagnosis_result(
    db: AsyncSession,
    diagnosis_id: str,
    image_description: str | None,
    symptoms: str | None,
    other_symptom: str | None,
    classification: str | None,
    diagnosis: str
):
    data = DiagnosisResultsCreate(
        diagnosis_id=diagnosis_id,
        image_description=image_description,
        symptoms=symptoms,
        other_symptom=other_symptom,
        classification=classification,
        diagnosis=diagnosis
    )
    return await create_diagnosis_result(db, data)