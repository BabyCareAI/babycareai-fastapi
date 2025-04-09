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
            temperature=0,
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
        당신은 영유아 피부질환 진단 보조 시스템입니다. 사용자가 제공한 이미지에서 보이는 영유아 피부의 객관적 특징만을 정확하게 설명해주세요.

        다음 형식에 맞춰 간결하고 명확하게 응답하세요:

        [피부 상태]: 이미지에서 관찰되는 피부의 전반적인 상태를 객관적으로 기술 (예: 붉은 발진이 보임, 물집이 관찰됨)

        [주요 특징]:
        - 발진 여부 및 형태: 크기, 분포, 경계 특성 기술 
        - 색상 변화: 정상 피부와 비교하여 관찰되는 색상 차이 
        - 표면 특성: 융기, 함몰, 각질, 물집 등 표면 상태 
        - 분포 패턴: 집중/분산 형태, 대칭/비대칭, 특정 부위 집중 여부

        [객관적 관찰사항]: 의학적 판단은 제외하고 관찰되는 사실만 추가 기술 (최대 2-3줄)

        다음 사항을 엄격히 준수하세요:
        1. 정확한 진단명은 언급하지 말고, 관찰되는 특징만 객관적으로 기술하세요.
        2. "~일 가능성이 있습니다"와 같은 추측성 표현은 사용하지 마세요.
        3. 전문 의료인의 진찰을 권장하는 문구는 포함하지 마세요. (시스템에서 별도 처리함)
        4. 이미지에서 보이는 객관적 특징만 기술하고, 보이지 않는 내용은 추정하지 마세요.
        5. 증상에 대한 경중이나 긴급성에 대한 판단은 하지 마세요.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "이 영유아 피부 이미지에서 관찰되는 객관적인 특징들을 설명해주세요."
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