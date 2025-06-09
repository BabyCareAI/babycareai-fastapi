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
from langchain_core.output_parsers import StrOutputParser
import logging
import json

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# OpenAI 임베딩 모델
embeddings = get_embeddings_model()

# Pinecone index 객체 생성
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def create_retrieval_chain() -> RunnablePassthrough:
    """
    질병 검색을 위한 LCEL 체인을 생성합니다.
    """
    def retrieve_diseases(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = input_data.get("query", "")
        top_k = input_data.get("top_k", 4)
        input_embedding = input_data.get("input_embedding")
        
        try:
            # 임베딩이 제공되지 않은 경우에만 새로 생성
            if input_embedding is None:
                input_embedding = embeddings.embed_query(query)
                
            # Pinecone 직접 쿼리
            results = index.query(
                vector=input_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # 결과 변환
            similar_diseases = []
            for match in results.matches:
                try:
                    # 메타데이터 처리
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

                    # 필수 필드 확인 및 기본값 설정
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
            
        except Exception as e:
            logging.error(f"[pinecone_client][검색 오류]: {e}", exc_info=True)
            return []

    return RunnablePassthrough(retrieve_diseases)

def retrieve_similar_diseases(query: str, top_k: int = 4, input_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    입력 텍스트(query)에 대해 top-k 유사 질병 정보를 반환합니다.
    
    Args:
        query: 검색할 텍스트
        top_k: 반환할 결과 수
        input_embedding: 외부에서 생성된 임베딩 (선택적)
    """
    try:
        # Pinecone 직접 쿼리
        results = index.query(
            vector=input_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 결과 변환
        similar_diseases = []
        for i, match in enumerate(results.matches):
            try:
                # 메타데이터 처리
                metadata = {}
                if isinstance(match.metadata, str):
                    try:
                        metadata = json.loads(match.metadata)
                    except json.JSONDecodeError as e:
                        logging.error(f"[Pinecone] JSON 파싱 실패 - match {i}: {e}")
                        continue
                elif isinstance(match.metadata, dict):
                    metadata = match.metadata
                else:
                    logging.error(f"[Pinecone] 지원하지 않는 메타데이터 타입 - match {i}: {type(match.metadata)}")
                    continue

                # 필수 필드 확인 및 기본값 설정
                disease_info = {
                    "metadata": {
                        "disease": metadata.get("disease", ""),
                        "symptoms": metadata.get("symptoms", []),
                        "skin_site": metadata.get("skin_site", []),
                        "text": metadata.get("text", "")
                    },
                    "score": float(match.score)
                }
                similar_diseases.append(disease_info)
                
            except Exception as e:
                logging.error(f"[Pinecone] match {i} 처리 중 오류 발생: {e}")
                continue

        return similar_diseases
            
    except Exception as e:
        logging.error(f"[Pinecone] 검색 오류: {e}", exc_info=True)
        return []