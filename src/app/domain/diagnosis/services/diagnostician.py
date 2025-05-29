# 진단(RAG) 서비스
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisIdInput, DiagnosisResponse, DiseaseInfo
from src.app.utils.llm_client import get_text_embedding, query_llm_with_context
from src.app.utils.pinecone_client import retrieve_similar_diseases
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from src.app.domain.diagnosis.crud.diagnostician import create_diagnosis_result
from src.app.domain.diagnosis.utils.data_processor import flatten_and_join
from sqlalchemy.ext.asyncio import AsyncSession
import logging

async def diagnose_with_rag(request: DiagnosisIdInput, top_k: int = 5, db: AsyncSession = None) -> DiagnosisResponse:
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
        logging.info(f"diagnosis_id={diagnosis_id}에 대한 입력 데이터: {input_text[:500]}... (총 {len(input_text)}자)")

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
        
        # 4. LLM에 컨텍스트와 함께 질의 (Chain-of-Thought & 맞춤형 조언, 한국어)
        context = "\n\n".join([
            f"질병명: {d.get('metadata', {}).get('disease', '')}\n"
            f"증상: {', '.join(d.get('metadata', {}).get('symptoms', []))}\n"
            f"피부 부위: {', '.join(d.get('metadata', {}).get('skin_site', []))}\n"
            f"설명: {d.get('content', '')}"
            for d in similar_diseases
        ])
        user_prompt = f"""
아래는 영유아 피부 질환에 대한 데이터입니다.

{context}

---

[사용자 입력]
{input_text}

---

위의 데이터와 사용자 입력을 바탕으로 다음 지침을 따라주세요:
1. 유사 질병 정보와 입력을 논리적으로 단계별(Chain-of-Thought)로 분석하여, 가장 가능성 높은 피부 질환을 도출하세요.
2. 진단 이유와 근거를 명확하게 서술하세요.
3. 진단 결과와 함께 보호자(부모)를 위한 맞춤형 조언(생활관리, 주의사항, 병원 방문 필요 여부 등)을 제시하세요.
4. 모든 답변은 반드시 한국어로 작성하세요.

[출력 예시]
1. 최종 진단: ...
2. 진단 이유: ... (step-by-step reasoning)
3. 맞춤형 조언: ...

---
답변:
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