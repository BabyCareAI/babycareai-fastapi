# LLM 클라이언트 유틸리티
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage
import base64

def create_llm_model(model_name: str = "gemini-2.0-flash-lite", temperature: float = 0, max_output_tokens: int = 500):
    """
    LLM 모델을 생성합니다.
    
    Args:
        model_name: 모델 이름
        temperature: 생성 다양성 (0: 결정적, 1: 다양성)
        max_output_tokens: 최대 출력 토큰 수
        
    Returns:
        ChatGoogleGenerativeAI: LLM 모델
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

def encode_image_to_base64(image_data: bytes) -> str:
    """
    이미지를 base64로 인코딩합니다.
    
    Args:
        image_data: 이미지 데이터 (bytes)
        
    Returns:
        str: base64로 인코딩된 이미지
    """
    return base64.b64encode(image_data).decode('utf-8')

async def process_image_with_llm(model, base64_image: str, system_prompt: str, user_prompt: str):
    """
    LLM 모델을 사용하여 이미지를 처리합니다.
    
    Args:
        model: LLM 모델
        base64_image: base64로 인코딩된 이미지
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 프롬프트
        
    Returns:
        str: LLM 응답
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ])
    ]
    
    response = await model.ainvoke(messages)
    return response.content.strip() 