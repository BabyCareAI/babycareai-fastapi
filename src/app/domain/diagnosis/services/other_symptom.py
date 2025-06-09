# 기타 증상 입력 api
from src.app.domain.diagnosis.schemas.other_symptom import OtherSymptomInput, OtherSymptomResult
from src.app.utils.llm_client import create_llm_model
from langchain.schema.messages import HumanMessage, SystemMessage
from src.app.utils.redis_client import save_to_redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_other_symptom_chain():
    """
    기타 증상 처리를 위한 LCEL 체인을 생성합니다.
    """
    model = create_llm_model(
        model_name="gemini-2.0-flash-lite",
        temperature=0,
        max_output_tokens=500
    )

    # 증상 검증 프롬프트
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine whether the input text is related to symptoms. If it is related, reply with 'Y'; otherwise, reply with 'N'."),
        ("human", "Input: {symptom_text}")
    ])

    # 요약 및 번역 프롬프트
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the entered symptom description in one concise sentence in Korean, then translate that summary naturally into English. Return only the English translation as plain text. Do not include any labels, headings, or the original Korean summary."),
        ("human", "Input: {symptom_text}")
    ])

    # 검증 체인
    validation_chain = (
        {"symptom_text": RunnablePassthrough()}
        | validation_prompt
        | model
        | StrOutputParser()
    )

    # 요약 및 번역 체인
    summary_chain = (
        {"symptom_text": RunnablePassthrough()}
        | summary_prompt
        | model
        | StrOutputParser()
    )

    # 병렬 실행을 위한 체인 구성
    return RunnableParallel(
        validation=validation_chain,
        summary=summary_chain
    )

async def process_other_symptom(data: OtherSymptomInput) -> OtherSymptomResult:
    """
    진단 ID를 입력받고 기타 증상 string을 처리합니다.
    Args:
        data: 진단 ID와 기타 증상 string
    Returns:
        OtherSymptomResult: 처리 결과
    """
    chain = create_other_symptom_chain()
    result = await chain.ainvoke(data.other_symptom_text)
    
    is_valid = result["validation"].strip().upper().startswith('Y')
    if not is_valid:
        return OtherSymptomResult(
            diagnosis_id=data.diagnosis_id,
            summarized_translation="",
            is_valid=False,
            message="입력하신 내용은 증상과 관련이 없습니다."
        )

    summarized_translation = result["summary"].strip()

    # Redis 저장
    redis_key = f"other_symptom:{data.diagnosis_id}"
    save_to_redis(redis_key, {
        "summarized_translation": summarized_translation
    })

    return OtherSymptomResult(
        diagnosis_id=data.diagnosis_id,
        summarized_translation=summarized_translation,
        is_valid=True,
        message="검증, 요약, 번역 처리가 완료되었습니다."
    )