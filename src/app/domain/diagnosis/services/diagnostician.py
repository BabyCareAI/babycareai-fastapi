# 진단(RAG) 서비스
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse, DiseaseInfo
from src.app.utils.llm_client import get_text_embedding, query_llm_with_context
from src.app.utils.pinecone_client import retrieve_similar_diseases
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from src.app.domain.diagnosis.crud.diagnostician import create_diagnosis_result
from src.app.domain.diagnosis.utils.data_processor import flatten_and_join, extract_top_classification
from sqlalchemy.ext.asyncio import AsyncSession
import logging

async def diagnose_with_rag(request: DiagnosisIdInput, top_k: int = 10, db: AsyncSession = None) -> DiagnosisResponse:
    """
    증상/부위/설명을 바탕으로 RAG 기반 진단을 수행합니다.
    1. 입력을 임베딩하여 벡터스토어에서 유사 질병 Top-K 검색
    2. LLM에 컨텍스트와 함께 질의하여 진단 생성
    3. 결과 반환
    """
    try:
        # 1. diagnosis_id 기반 redis에서 데이터 조회
        from src.app.utils.redis_client import get_from_redis
        diagnosis_id = request.diagnosis_id
        redis_keys = [
            f"image_description:{diagnosis_id}",
            f"symptoms:{diagnosis_id}",
            f"other_symptom:{diagnosis_id}",
            f"classification:{diagnosis_id}"
        ]
        values = [get_from_redis(key) for key in redis_keys]

        # classification에서 가장 높은 확률의 클래스 추출
        classification_data = values[3]  # classification 데이터
        top_classification = extract_top_classification(classification_data)
        
        # classification 데이터를 추출된 클래스로 대체
        values[3] = top_classification
        # logging.info(values[3])

        input_text = flatten_and_join(values)
        input_text = str(input_text)

        if not input_text:
            logging.error(f"diagnosis_id={diagnosis_id}에 해당하는 입력 데이터가 없습니다.")
            return DiagnosisResponse(
                diagnosis="입력 데이터 없음",
                top_k_diseases=[],
                input_embedding=None,
                retrieved_embeddings=None
            )
        # logging.info(f"diagnosis_id={diagnosis_id}에 대한 입력 데이터: {input_text[:1000]}... (총 {len(input_text)}자)")

        # 2. 입력 임베딩 생성 (OpenAIEmbeddings 사용)
        input_embedding = await get_text_embedding(input_text)
        if input_embedding is None:
            logging.error("입력 임베딩 생성 실패")
            return DiagnosisResponse(
                diagnosis="입력 임베딩 생성 실패",
                top_k_diseases=[],
                input_embedding=None,
                retrieved_embeddings=None
            )
        
        # 3. retriever를 통한 유사 질병 Top-K 검색 (텍스트 기반)
        similar_diseases = retrieve_similar_diseases(input_text, top_k=top_k)
        if not similar_diseases:
            return DiagnosisResponse(
                diagnosis="유사 질병을 찾을 수 없습니다.",
                top_k_diseases=[],
                input_embedding=input_embedding,
                retrieved_embeddings=None
            )

        # 4. LLM에 컨텍스트와 함께 질의 (Chain-of-Thought)
        context = "\n\n".join([
            f"[{i+1}] Disease Name: {d.get('metadata', {}).get('disease', '')}\n"
            f"Symptoms: {', '.join(d.get('metadata', {}).get('symptoms', []))}\n"
            f"Skin Site: {', '.join(d.get('metadata', {}).get('skin_site', []))}\n"
            f"Skin Description: {d.get('content', '')}"
            for i, d in enumerate(similar_diseases)
        ])
        user_prompt = f"""
Below is data about infant skin diseases. The documents are sorted by relevance, with the first and last documents being the most relevant.

{context}

---

[User Input]
{input_text}

---

Based on the above data and user input, please follow these guidelines:
1. Primarily reference the first and last documents, analyze the similar disease information and input logically step-by-step (Chain-of-Thought) to derive the most likely skin condition.
2. Clearly state the diagnosis reason and evidence.
3. Provide the response in Korean, using a warm and friendly tone that will reassure worried parents.

Please structure your response in the following format:
- 최종 진단 (Final Diagnosis): 진단명
- 진단 이유 (Diagnosis Reason): 이미지나 증상을 통해 해당 진단에 도달한 이유
- 중증도 (Severity): 중등도를 판단하며, 간단한 설명 포함
- 병원 내원 필요 여부 (Need for Hospital Visit): 즉시 방문 필요 여부를 명확하게 안내
- 가정 내 처치 방법 (Home Care Instructions): 부모님이 쉽게 실천할 수 있는 구체적이고 실용적인 조언

Guidelines for the response:
1. Maintain accurate medical terms but add simple explanations in parentheses when needed
2. Use warm and empathetic expressions that can reassure worried parents
3. Write in formal Korean (존댓말)
4. Choose vocabulary that reduces anxiety and builds trust
5. Avoid directly referencing document numbers (e.g., "문서 1과 4"). Instead, refer to the source naturally (e.g., "참고 자료에 따르면", "의료 정보를 바탕으로", "관련 자료에서는" 등)
"""
        diagnosis = await query_llm_with_context(user_prompt)
        
        # 5. 결과 조합 및 반환
        top_k_diseases = [
            DiseaseInfo(
                disease=d.get('metadata', {}).get('disease', ''),
                symptoms=d.get('metadata', {}).get('symptoms', []),
                skin_site=d.get('metadata', {}).get('skin_site', []),
                disease_information=d.get('content', ''),
                similarity=d.get('score', None),
            )
            for d in similar_diseases
        ]
        return DiagnosisResponse(
            diagnosis=diagnosis,
            top_k_diseases=top_k_diseases,
            input_embedding=input_embedding,
            retrieved_embeddings=[d.get('embedding') for d in similar_diseases]
        )
    except Exception as e:
        logging.exception("진단 RAG 서비스 오류: %s", e)
        return DiagnosisResponse(
            diagnosis=f"진단 중 오류 발생: {e}",
            top_k_diseases=[],
            input_embedding=None,
            retrieved_embeddings=None
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
        id=diagnosis_id,
        image_description=image_description,
        symptoms=symptoms,
        other_symptom=other_symptom,
        classification=classification,
        diagnosis=diagnosis
    )
    return await create_diagnosis_result(db, data)