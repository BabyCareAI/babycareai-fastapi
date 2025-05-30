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
import logging

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

# retriever를 통해 유사 질병 검색 (sync)
def retrieve_similar_diseases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    입력 텍스트(query)에 대해 top-k 유사 질병 정보를 반환합니다.
    """

    # 임베딩 생성
    try:
        input_embedding = embeddings.embed_query(query)
    except Exception as e:
        logging.error(f"[pinecone_client][체크리스트] 임베딩 생성 오류: {str(e)}")
        return []

    # 5. 검색
    docs = vectorstore.similarity_search(query, k=top_k)
    # logging.info(f"[pinecone_client][체크리스트] 검색된 문서 수: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        sim = doc.metadata.get('similarity', 'N/A')
        # logging.info(f"[{i}] {doc.metadata.get('disease', 'N/A')}: {sim}")
        # logging.info(f"[{i}] doc.metadata: {doc.metadata}")
        # logging.info(f"[{i}] doc.page_content[:100]: {doc.page_content[:100]}")
    return [
        {"metadata": doc.metadata, "content": doc.page_content} for doc in docs
    ]

