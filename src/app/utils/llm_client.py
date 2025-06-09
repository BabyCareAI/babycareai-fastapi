# LLM 클라이언트 유틸리티
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings

import base64
import asyncio
from typing import Union, Optional
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)

LLMModel = Union[ChatGoogleGenerativeAI, ChatOpenAI]

@lru_cache(maxsize=1)
def get_embeddings_model() -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 모델을 반환합니다. (lru_cache 사용)
    """
    return OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

async def get_text_embedding(text: str) -> Optional[list]:
    """
    텍스트를 임베딩 벡터로 변환합니다.
    Args:
        text (str): 임베딩할 텍스트
    Returns:
        Optional[list]: 임베딩 벡터, 실패 시 None
    """
    if not text:
        logging.error("get_text_embedding: 입력 텍스트가 비어 있습니다.")
        return None
    model = get_embeddings_model()
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, model.embed_query, text)
    except Exception as e:
        logging.error(f"입력 임베딩 생성 실패: {e}")
        return None

def create_llm_model(
    *,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0,
    max_output_tokens: int = 1000,
    provider: str = "google"
) -> LLMModel:
    """
    LLM 모델을 생성 및 초기화합니다.
    Args:
        model_name (str): 모델 이름
        temperature (float): 생성 다양성
        max_output_tokens (int): 최대 출력 토큰 수
        provider (str): LLM 제공자 ("google" 또는 "openai")
    Returns:
        LLMModel: LLM 모델 인스턴스
    """
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    if provider == "openai":
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens
        )
    logging.error(f"지원하지 않는 provider: {provider}")
    raise ValueError(f"지원하지 않는 provider: {provider}")

def encode_image_to_base64(image_data: bytes) -> str:
    """
    이미지를 base64로 인코딩합니다.

    Args:
        image_data: 이미지 데이터 (bytes)

    Returns:
        str: base64로 인코딩된 이미지
    """
    return base64.b64encode(image_data).decode('utf-8')

async def process_image_with_llm(
        model: LLMModel,
        base64_image: str,
        system_prompt: str,
        user_prompt: str,
        provider: str = "google"
) -> str:
    """
    LLM 모델을 사용하여 이미지를 처리합니다.

    Args:
        model: LLM 모델
        base64_image: base64로 인코딩된 이미지
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 프롬프트
        provider: LLM 제공자 ("google" 또는 "openai")

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