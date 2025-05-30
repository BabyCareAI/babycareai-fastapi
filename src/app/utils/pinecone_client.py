"""
Pinecone 벡터스토어 클라이언트 유틸리티
- Pinecone index 연결, 벡터 upsert, top-k 검색 함수 제공
- 환경변수: PINECONE_API_KEY, PINECONE_INDEX_NAME
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_transformers import LongContextReorder
import logging
import numpy as np

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# OpenAI 임베딩 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

# Pinecone index 객체 생성
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# PineconeVectorStore 생성 (api_key, environment 인자 제거, index 인자 사용)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# as_retriever로 retriever 객체 제공
disease_retriever = vectorstore.as_retriever()

def retrieve_similar_diseases(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    입력 텍스트(query)에 대해 top-k 유사 질병 정보를 반환합니다.
    LongContextReorder를 사용하여 검색 결과를 재정렬합니다.
    """
    try:
        # 임베딩 생성
        input_embedding = embeddings.embed_query(query)
    except Exception as e:
        logging.error(f"[pinecone_client][체크리스트] 임베딩 생성 오류: {str(e)}")
        return []

    # 검색
    docs = vectorstore.similarity_search(query, k=top_k)
    
    # LongContextReorder 적용
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    
    return [
        {"metadata": doc.metadata, "content": doc.page_content} 
        for doc in reordered_docs
    ]

