# 이미지 상태 설명 API
from src.app.utils.s3_client import get_image_from_s3
from src.app.utils.redis_client import save_to_redis
from src.app.utils.llm_client import create_llm_model, encode_image_to_base64, process_image_with_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 이미지 상태 설명 서비스
class ImageDescriptorService:
    def __init__(self):
        self.model = create_llm_model(
            model_name="gemini-2.0-flash-lite",
            temperature=0,
            max_output_tokens=500
        )
        self.chain = self._create_description_chain()
    
    def _create_description_chain(self):
        """
        이미지 설명을 위한 LCEL 체인을 생성합니다.
        """
        description_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI assistant system that supports the diagnosis of infant skin conditions. 
            Please objectively and thoroughly describe the skin condition as seen in the image provided by the user.

            Focus on the following elements, and respond clearly and specifically:

            [Skin Condition]: Describe the overall visible condition of the skin (e.g., widespread red spots, localized blisters).

            [Key Characteristics]:
            - Presence and type of rash: size, number, clarity of borders, and other visually identifiable features
            - Color changes: differences in color compared to normal skin, including intensity and distribution
            - Surface features: texture changes such as roughness, scaling, blisters, or ulcers
            - Distribution pattern: whether symptoms are localized or widespread, symmetrical or asymmetrical, and concentrated in specific areas

            [Objective Observations]: Describe all features that can be confirmed visually from the image without subjective interpretation.

            Please strictly follow these guidelines:
            1. Do not mention any specific diagnosis. Only describe features that are visually observable.
            2. Avoid speculative expressions such as "appears to be" or "might be".
            3. Do not recommend seeing a medical professional or make diagnostic judgments.
            4. Do not mention or infer information that cannot be seen in the image.
            5. Do not assess severity or urgency of the symptoms.
            """),
            ("human", "Please describe the objective and observable characteristics of this infant skin image.")
        ])

        return (
            {"base64_image": RunnablePassthrough()}
            | description_prompt
            | self.model
            | StrOutputParser()
        )
    
    async def describe_skin_image(self, diagnosis_id: str):
        """
        S3에서 이미지를 가져와 피부 상태를 설명합니다.
        Args:
            diagnosis_id: 진단 ID (이미지 파일명)
        Returns:
            dict: 이미지 설명 결과 및 부위 정보
        """
        # S3에서 이미지와 메타데이터(bodyPart) 가져오기
        image_data, body_part = get_image_from_s3(diagnosis_id)
        if not image_data:
            raise Exception(f"이미지를 찾을 수 없습니다. 진단 ID: {diagnosis_id}")

        # 이미지를 base64로 인코딩
        base64_image = encode_image_to_base64(image_data)

        # 이미지 설명 생성
        description = await self.chain.ainvoke(base64_image)

        # Redis에 결과 저장
        self._save_to_redis(diagnosis_id, description, body_part)

        return {"description": description, "bodyPart": body_part}

    def _save_to_redis(self, diagnosis_id: str, description: str, body_part: str | None):
        """이미지 설명 결과 및 부위 정보를 Redis에 저장합니다."""
        redis_key = f"image_description:{diagnosis_id}"
        save_to_redis(redis_key, {"bodyPart": body_part, "description": description})

# 서비스 인스턴스 생성
image_descriptor_service = ImageDescriptorService()