# Redis 클라이언트 유틸리티
import redis
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Redis 클라이언트 초기화
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', ''),
    db=int(os.getenv('REDIS_DB', 0))
)

def save_to_redis(key: str, value: dict, expire_seconds: int = 86400):
    """
    Redis에 데이터를 저장합니다.
    
    Args:
        key: Redis 키
        value: 저장할 데이터 (딕셔너리)
        expire_seconds: 만료 시간 (초), 기본값 24시간
    """
    try:
        redis_client.set(key, json.dumps(value))
        # 만료 시간 설정
        redis_client.expire(key, expire_seconds)
    except Exception as e:
        print(f"Redis에 저장하는 중 오류 발생: {e}")

def get_from_redis(key: str) -> dict:
    """
    Redis에서 데이터를 가져옵니다.
    
    Args:
        key: Redis 키
        
    Returns:
        dict: 저장된 데이터
    """
    try:
        data = redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"Redis에서 데이터를 가져오는 중 오류 발생: {e}")
        return None 