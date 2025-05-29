# S3 클라이언트 유틸리티
import boto3
import os
from dotenv import load_dotenv
from sqlalchemy.testing.plugin.plugin_base import logging

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

def get_image_from_s3(diagnosis_id: str) -> tuple[bytes, str | None]:
    """
    S3에서 이미지를 가져오고, 메타데이터(bodyPart)를 함께 반환합니다.
    
    Args:
        diagnosis_id: 진단 ID (이미지 파일명)
        
    Returns:
        (이미지 데이터, body_part)
    """
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{diagnosis_id}"
        )
        image_data = response['Body'].read()
        metadata = response.get('Metadata', {})
        body_part = metadata.get('bodypart') or metadata.get('bodyPart')
        return image_data, body_part
    except Exception as e:
        logging.error(f"S3에서 이미지를 가져오는 중 오류 발생: {e}")
        return None, None