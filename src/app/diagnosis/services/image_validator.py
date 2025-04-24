# 이미지 검증 api
from src.app.utils.s3_client import get_image_from_s3
from src.app.utils.llm_client import create_llm_model, encode_image_to_base64, process_image_with_llm

# 이미지 검증 서비스
class DermaValidatorService:
    def __init__(self):
        self.model = create_llm_model(
            model_name="gemini-2.0-flash-lite",
            temperature=0,
            max_output_tokens=200
        )
    
    async def validate_skin_image(self, diagnosis_id: str):
        """
        S3에서 이미지를 가져와 피부 관련 이미지인지 검증합니다.
        
        Args:
            diagnosis_id: 진단 ID (이미지 파일명)
            
        Returns:
            dict: 검증 결과 (피부 관련 이미지 여부)
        """
        # S3에서 이미지 가져오기
        image_data, _ = get_image_from_s3(diagnosis_id)
        if not image_data:
            raise Exception(f"이미지를 찾을 수 없습니다. 진단 ID: {diagnosis_id}")
        
        # 이미지를 base64로 인코딩
        base64_image = encode_image_to_base64(image_data)
        
        # 이미지 검증
        is_skin_related = await self._validate_with_llm(base64_image)
        return {"is_skin_related": is_skin_related}
    
    async def _validate_with_llm(self, base64_image: str) -> bool:
        """LLM을 사용하여 이미지가 피부 관련 이미지인지 검증합니다."""
        system_prompt = """
        You are an expert in medical image analysis. 
        You need to determine whether the image provided by the user is related to skin conditions.
        A skin-related image refers to images showing skin diseases, skin conditions, or parts of the skin.
        
        If the image is related to skin, output only 'YES'.
        If the image is not related to skin, output only 'NO'.
        Do not provide any explanation.
        """
        
        user_prompt = "Please determine whether this image is related to skin conditions."
        # "이 이미지가 피부 관련 이미지인지 판단해주세요."
        
        content = await process_image_with_llm(self.model, base64_image, system_prompt, user_prompt)
        
        # 응답 확인 (YES 또는 NO)
        return 'YES' in content.upper()

# 서비스 인스턴스 생성
derma_validator_service = DermaValidatorService()