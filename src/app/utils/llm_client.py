# LLM 클라이언트 유틸리티
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
import base64
import asyncio

# OpenAI 텍스트 임베딩 모델 (text-embedding-3-large)
_embeddings_model = None

# OpenAIEmbeddings는 비동기 지원이므로 직접 await 가능

def get_embeddings_model() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    return _embeddings_model

async def get_text_embedding(text: str) -> list:
    """
    텍스트를 임베딩 벡터로 변환합니다.
    """
    model = get_embeddings_model()
    # OpenAIEmbeddings.embed_query는 sync이므로 run_in_executor 사용
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, model.embed_query, text)

async def query_llm_with_context(user_prompt: str, model_name: str = "gemini-2.0-flash-lite", temperature: float = 0, max_output_tokens: int = 500) -> str:
    """
    LLM에 프롬프트를 입력하여 응답을 반환합니다.
    """
    model = create_llm_model(model_name, temperature, max_output_tokens)
    messages = [HumanMessage(content=user_prompt)]
    loop = asyncio.get_event_loop()
    # ChatGoogleGenerativeAI는 sync이므로 run_in_executor
    return await loop.run_in_executor(None, lambda: model(messages).content)

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