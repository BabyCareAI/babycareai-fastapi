# LLM 클라이언트 유틸리티
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import base64
import asyncio
from typing import Union, Optional, Dict, Any
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO)

# OpenAI 텍스트 임베딩 모델 (text-embedding-3-large)
_embeddings_model = None

# LLM 모델 타입 정의
LLMModel = Union[ChatGoogleGenerativeAI, ChatOpenAI]

def get_embeddings_model() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
        logging.info("OpenAIEmbeddings: NEW INSTANCE CREATED")
    else:
        logging.info("OpenAIEmbeddings: REUSING EXISTING INSTANCE")
    return _embeddings_model


async def get_text_embedding(text: str) -> list:
    """
    텍스트를 임베딩 벡터로 변환합니다.
    """
    model = get_embeddings_model()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, model.embed_query, text)
    except Exception as e:
        logging.error(f"입력 임베딩 생성 실패: {e}", exc_info=True)
        traceback.print_exc(file=sys.stdout)
        return None
    return result


async def query_llm_with_context(
        user_prompt: str,
        model_name: str = "gemini-2.0-flash",  # gpt-4o-mini-2024-07-18 # gemini-2.0-flash
        temperature: float = 0,
        max_output_tokens: int = 1000,
        provider: str = "google"  # openai google
) -> str:
    """
    LLM에 프롬프트를 입력하여 응답을 반환합니다.

    Args:
        user_prompt: 사용자 입력 프롬프트
        model_name: 사용할 LLM 모델 이름
        temperature: 생성 다양성 (0: 결정적, 1: 다양성)
        max_output_tokens: 최대 출력 토큰 수
        provider: LLM 제공자 ("google" 또는 "openai")

    Returns:
        str: LLM 응답 내용
    """
    model = create_llm_model(model_name, temperature, max_output_tokens, provider)
    messages = [HumanMessage(content=user_prompt)]
    loop = asyncio.get_event_loop()

    if provider == "openai":
        return await loop.run_in_executor(None, lambda: model(messages).content)
    else:
        response = await model.ainvoke(messages)
        return response.content


def create_llm_model(
        model_name: str = "gemini-2.0-flash",  # gpt-4o-mini-2024-07-18 # gemini-2.0-flash
        temperature: float = 0,
        max_output_tokens: int = 1000,
        provider: str = "google"  # openai google
) -> LLMModel:
    """
    LLM 모델을 생성 및 초기화합니다.

    Args:
        model_name: 모델 이름
        temperature: 생성 다양성 (0: 결정적, 1: 다양성)
        max_output_tokens: 최대 출력 토큰 수
        provider: LLM 제공자 ("google" 또는 "openai")

    Returns:
        LLMModel: LLM 모델 인스턴스
    """
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    else:  # OpenAI
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens
        )


def create_diagnosis_chain(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0,
    max_output_tokens: int = 1000,
    provider: str = "google"
) -> RunnableParallel:
    """
    진단을 위한 LCEL 체인을 생성합니다.
    """
    model = create_llm_model(model_name, temperature, max_output_tokens, provider)
    
    # 진단 프롬프트 템플릿
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

    # 번역 프롬프트 템플릿
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

    # 진단 체인
    diagnosis_chain = (
        diagnosis_prompt 
        | model 
        | StrOutputParser()
    )

    # 번역 체인
    translation_chain = (
        {"diagnosis_result": diagnosis_chain}
        | translation_prompt 
        | model 
        | StrOutputParser()
    )

    # 병렬 실행을 위한 체인 구성
    return RunnableParallel(
        diagnosis=diagnosis_chain,
        translation=translation_chain
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