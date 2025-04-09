# 이미지 상태 설명 API
import boto3
import os
import redis
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage
import base64
import json

load_dotenv()

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
)

# Redis 클라이언트 초기화
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', ''),
    db=int(os.getenv('REDIS_DB', 0))
)

# S3 버킷 이름
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# 이미지 상태 설명 서비스
class ImageDescriptorService:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.2,
            max_output_tokens=500
        )
    
    async def describe_skin_image(self, diagnosis_id: str):
        """
        S3에서 이미지를 가져와 피부 상태를 설명합니다.
        
        Args:
            diagnosis_id: 진단 ID (이미지 파일명)
            
        Returns:
            dict: 이미지 설명 결과
        """
        # S3에서 이미지 가져오기
        image_data = self._get_image_from_s3(diagnosis_id)
        if not image_data:
            raise Exception(f"이미지를 찾을 수 없습니다. 진단 ID: {diagnosis_id}")
        
        # 이미지를 base64로 인코딩
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 이미지 설명 생성
        description = await self._describe_with_llm(base64_image)
        
        # Redis에 결과 저장
        self._save_to_redis(diagnosis_id, description)
        
        return {"description": description}
        
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
    
    async def _describe_with_llm(self, base64_image: str) -> str:
        """LLM을 사용하여 이미지의 피부 상태를 설명합니다."""
        system_prompt = """
        당신은 영유아 피부 질환 전문가입니다.
        사용자가 제공한 이미지에서 영유아의 피부 상태나 특징을 자세히 설명해주세요.
        
        다음 사항을 포함하여 설명해주세요:
        1. 피부의 전반적인 상태 (건강한지, 문제가 있는지)
        2. 발진, 염증, 상처, 피가 보이는지 등 구체적인 특징
        3. 피부 질환이 의심되는 경우 어떤 질환일 가능성이 있는지
        4. 피부의 색상, 질감, 형태 등 물리적 특징
        
        설명은 한국어로 작성해주세요.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "이 이미지에서 영유아의 피부 상태나 특징을 자세히 설명해주세요."
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
        return response.content.strip()
    
    def _save_to_redis(self, diagnosis_id: str, description: str):
        """이미지 설명 결과를 Redis에 저장합니다."""
        try:
            redis_key = f"image_description:{diagnosis_id}"
            redis_client.set(redis_key, json.dumps({"description": description}))
            # 24시간 후 만료되도록 설정
            redis_client.expire(redis_key, 86400)
        except Exception as e:
            print(f"Redis에 저장하는 중 오류 발생: {e}")

# 서비스 인스턴스 생성
image_descriptor_service = ImageDescriptorService()