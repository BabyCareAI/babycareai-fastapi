"""
Pinecone 벡터스토어 클라이언트 유틸리티
- Pinecone index 연결, 벡터 upsert, top-k 검색 함수 제공
- 환경변수: PINECONE_API_KEY, PINECONE_INDEX_NAME
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.app.utils.llm_client import get_embeddings_model
from langchain_core.runnables import RunnablePassthrough
from functools import lru_cache
from pinecone import Pinecone
import logging
import json


load_dotenv()

@lru_cache(maxsize=1)
def get_pinecone_index():
    """
    Pinecone Index 인스턴스 반환 (lru_cache)
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise RuntimeError("PINECONE_API_KEY 또는 PINECONE_INDEX_NAME 환경변수가 누락되었습니다.")
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

@lru_cache(maxsize=1)
def get_embeddings():
    return get_embeddings_model()

def create_retrieval_chain() -> RunnablePassthrough:
    """
    질병 검색을 위한 LCEL 체인을 생성합니다.
    """
    def retrieve_diseases(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = input_data.get("query", "")
        top_k = input_data.get("top_k", 4)
        input_embedding = input_data.get("input_embedding")
        if not query and input_embedding is None:
            logging.error("retrieve_diseases: query 또는 input_embedding 중 하나는 필수입니다.")
            return []
        try:
            emb = input_embedding or get_embeddings().embed_query(query)
            results = get_pinecone_index().query(
                vector=emb,
                top_k=top_k,
                include_metadata=True
            )
            return _convert_pinecone_matches_to_diseases(results.matches)
        except Exception as e:
            logging.error(f"[pinecone_client][검색 오류]: {e}", exc_info=True)
            return []
    return RunnablePassthrough(retrieve_diseases)

def _convert_pinecone_matches_to_diseases(matches) -> List[Dict[str, Any]]:
    """
    Pinecone match 객체를 유사 질병 리스트로 변환
    """
    similar_diseases = []
    for match in matches:
        try:
            metadata = {}
            if isinstance(match.metadata, str):
                try:
                    metadata = json.loads(match.metadata)
                except json.JSONDecodeError:
                    logging.error(f"JSON 파싱 실패: {match.metadata}")
                    continue
            elif isinstance(match.metadata, dict):
                metadata = match.metadata
            else:
                logging.error(f"지원하지 않는 메타데이터 타입: {type(match.metadata)}")
                continue
            disease_info = {
                "metadata": {
                    "disease": metadata.get("disease", ""),
                    "symptoms": metadata.get("symptoms", []),
                    "skin_site": metadata.get("skin_site", []),
                    "text": metadata.get("text", [])
                },
                "score": float(match.score)
            }
            # 데이터 유효성 검사
            if not isinstance(disease_info["metadata"]["symptoms"], list):
                disease_info["metadata"]["symptoms"] = []
            if not isinstance(disease_info["metadata"]["skin_site"], list):
                disease_info["metadata"]["skin_site"] = []
            if not isinstance(disease_info["metadata"]["text"], list):
                disease_info["metadata"]["text"] = []
            similar_diseases.append(disease_info)
        except Exception as e:
            logging.error(f"결과 변환 중 오류 발생: {e}")
            continue
    return similar_diseases

def retrieve_similar_diseases(query: str, top_k: int = 4, input_embedding: Optional[list] = None) -> List[Dict[str, Any]]:
    """
    입력 텍스트(query)에 대해 top-k 유사 질병 정보를 반환합니다.
    Args:
        query (str): 검색할 텍스트
        top_k (int): 반환할 결과 수
        input_embedding (Optional[list]): 외부에서 생성된 임베딩
    Returns:
        List[Dict[str, Any]]: 유사 질병 정보 리스트
    """
    if not query and input_embedding is None:
        logging.error("retrieve_similar_diseases: query 또는 input_embedding 중 하나는 필수입니다.")
        return []
    try:
        emb = input_embedding or get_embeddings().embed_query(query)
        results = get_pinecone_index().query(
            vector=emb,
            top_k=top_k,
            include_metadata=True
        )
        return _convert_pinecone_matches_to_diseases(results.matches)
    except Exception as e:
        logging.error(f"[pinecone_client][검색 오류]: {e}", exc_info=True)
        return []