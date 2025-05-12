# 기타 증상 입력 api
from src.app.domain.diagnosis.schemas.other_symptom import OtherSymptomInput, OtherSymptomResult
from src.app.utils.llm_client import create_llm_model
from langchain.schema.messages import HumanMessage, SystemMessage
from src.app.utils.redis_client import save_to_redis

async def process_other_symptom(data: OtherSymptomInput) -> OtherSymptomResult:
    """
    진단 ID를 입력받고 기타 증상 string을 처리합니다.
    Args:
        data: 진단 ID와 기타 증상 string
    Returns:
        OtherSymptomResult: 처리 결과
    """
    model = create_llm_model()
    # 검증
    system_prompt = "Determine whether the input text is related to symptoms. If it is related, reply with 'Y'; otherwise, reply with 'N'."
    user_prompt = f"Input: {data.other_symptom_text}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = await model.ainvoke(messages)
    is_valid = response.content.strip().upper().startswith('Y')
    if not is_valid:
        return OtherSymptomResult(
            diagnosis_id=data.diagnosis_id,
            summarized_translation="",
            is_valid=False,
            message="입력하신 내용은 증상과 관련이 없습니다."
        )

    # 요약+번역
    system_prompt = (
        "Summarize the entered symptom description in one concise sentence in Korean, then translate that summary naturally into English. "
        "Return only the English translation as plain text. Do not include any labels, headings, or the original Korean summary."
    )
    user_prompt = f"Input: {data.other_symptom_text}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = await model.ainvoke(messages)
    summarized_translation = response.content.strip()

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