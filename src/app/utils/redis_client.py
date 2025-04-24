# Redis 클라이언트 유틸리티
import redis
import os
import json
from dotenv import load_dotenv
import numpy as np
import asyncio
from typing import List, Dict, Any

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
    """
    try:
        data = redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"Redis에서 데이터를 가져오는 중 오류 발생: {e}")
        return None

# --- 벡터스토어 관련 함수 ---

async def search_similar_diseases(input_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Redis에 저장된 질병 임베딩 중 입력 임베딩과 가장 유사한 Top-K 질병 반환
    (질병 정보는 key: 'disease_info:<index>'로 저장, 임베딩은 key: 'disease_embedding:<index>'로 저장되어 있다고 가정)
    """
    loop = asyncio.get_event_loop()
    # Redis에서 모든 임베딩 및 질병 정보 조회
    keys = redis_client.keys('disease_embedding:*')
    if not keys:
        return []
    all_embeddings = []
    all_infos = []
    for key in keys:
        idx = key.decode().split(':')[-1]
        emb = redis_client.get(key)
        if emb is None:
            continue
        emb = json.loads(emb)
        all_embeddings.append(emb)
        info = redis_client.get(f'disease_info:{idx}')
        if info:
            info = json.loads(info)
            info['embedding'] = emb
            all_infos.append(info)
    if not all_embeddings or not all_infos:
        return []
    # numpy 배열로 변환
    input_vec = np.array(input_embedding)
    emb_mat = np.array(all_embeddings)
    # cosine similarity 계산
    sim = emb_mat @ input_vec / (np.linalg.norm(emb_mat, axis=1) * np.linalg.norm(input_vec) + 1e-8)
    top_indices = np.argsort(sim)[::-1][:top_k]
    result = []
    for i in top_indices:
        info = all_infos[i]
        info['similarity'] = float(sim[i])
        result.append(info)
    return result