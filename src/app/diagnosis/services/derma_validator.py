# 이미지 검증 api
import boto3
import os
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import base64

load_dotenv()

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
)

# S3 버킷 이름
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# 이미지 검증 서비스
class DermaValidatorService:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=1,
            max_tokens=1000
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
        image_data = self._get_image_from_s3(diagnosis_id)
        if not image_data:
            raise Exception(f"이미지를 찾을 수 없습니다. 진단 ID: {diagnosis_id}")
        
        # 이미지를 base64로 인코딩
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 이미지 검증
        is_skin_related = await self._validate_with_llm(base64_image)
        return {"is_skin_related": is_skin_related}
        
    def _get_image_from_s3(self, diagnosis_id: str) -> bytes:
        """S3에서 이미지를 가져옵니다."""
        try:
            response = s3_client.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{diagnosis_id}"  # S3에 저장된 파일 이름이 diagnosis_id
            )
            image_data = response['Body'].read()
            return image_data
        except Exception as e:
            print(f"S3에서 이미지를 가져오는 중 오류 발생: {e}")
            return None
    
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
        # (한국어 버전)
        # """
        # 당신은 의료 이미지 분석 전문가입니다.
        # 사용자가 제공한 이미지가 피부(skin) 관련 이미지인지 판단해야 합니다.
        # 피부 관련 이미지란 피부 질환, 피부 상태, 피부의 일부분을 보여주는 이미지를 의미합니다.
        #
        # 분석 결과로 이미지가 피부 관련 이미지인 경우 'YES'만 출력하고,
        # 피부 관련 이미지가 아닌 경우 'NO'만 출력하세요.
        # 판단 이유는 작성하지 마세요.
        # """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Please determine whether this image is related to skin conditions."    # "이 이미지가 피부 관련 이미지인지 판단해주세요."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ])
        ]
        
        response = await self.model.ainvoke(messages)
        
        # 응답 확인 (YES 또는 NO)
        content = response.content.strip().upper()
        return 'YES' in content

# 서비스 인스턴스 생성
derma_validator_service = DermaValidatorService()