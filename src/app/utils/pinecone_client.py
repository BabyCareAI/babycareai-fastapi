"""
Pinecone 벡터스토어 클라이언트 유틸리티
- Pinecone index 연결, 벡터 upsert, top-k 검색 함수 제공
- 환경변수: PINECONE_API_KEY, PINECONE_INDEX_NAME
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.app.utils.llm_client import get_embeddings_model
import logging

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# OpenAI 임베딩 모델
embeddings = get_embeddings_model()

# Pinecone index 객체 생성
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def retrieve_similar_diseases(query: str, top_k: int = 4, input_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    입력 텍스트(query)에 대해 top-k 유사 질병 정보를 반환합니다.
    
    Args:
        query: 검색할 텍스트
        top_k: 반환할 결과 수
        input_embedding: 외부에서 생성된 임베딩 (선택적)
    """
    try:
        # 임베딩이 제공되지 않은 경우에만 새로 생성
        if input_embedding is None:
            input_embedding = embeddings.embed_query(query)
    except Exception as e:
        logging.error(f"[pinecone_client][임베딩 생성 오류]: {e}", exc_info=True)
        return []

    try:
        # Pinecone 직접 쿼리
        results = index.query(
            vector=input_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 결과 변환
        similar_diseases = []
        for match in results.matches:
            similar_diseases.append({
                "metadata": match.metadata,
                "content": match.metadata.get("content", ""),
                "score": match.score
            })
        
        return similar_diseases
    except Exception as e:
        logging.error(f"[pinecone_client][검색 오류]: {e}", exc_info=True)
        return []