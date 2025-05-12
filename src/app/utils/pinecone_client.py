"""
Pinecone 벡터스토어 클라이언트 유틸리티
- Pinecone index 연결, 벡터 upsert, top-k 검색 함수 제공
- 환경변수: PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# OpenAI 임베딩 모델 (동일하게 사용)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

print("[Pinecone] INDEX_NAME:", PINECONE_INDEX_NAME)
print("[Pinecone] ENV:", PINECONE_ENV)
print("[Pinecone] Embedding model:", getattr(embeddings, 'model', None))
print("[Pinecone] Embedding dimension:", getattr(embeddings, 'dimensions', None))
print("[Pinecone] VectorStore 생성 시 네임스페이스:", "default")

# Pinecone index 객체 생성
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# PineconeVectorStore 생성 (api_key, environment 인자 제거, index 인자 사용)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="default"  # 필요시 네임스페이스 지정
)

# as_retriever로 retriever 객체 제공
disease_retriever = vectorstore.as_retriever()

# retriever를 통해 유사 질병 검색 (sync)
def retrieve_similar_diseases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    입력 텍스트(query)에 대해 top-k 유사 질병 정보를 반환합니다.
    """
    print("[pinecone_client] query repr:", repr(query), flush=True)
    print("[pinecone_client] query type:", type(query), flush=True)
    if not isinstance(query, str):
        query = str(query)

    # 임베딩 생성 로그
    try:
        input_embedding = embeddings.embed_query(query)
        print("[pinecone_client] input_embedding[:10]:", input_embedding[:10], flush=True)
        print("[pinecone_client] input_embedding dim:", len(input_embedding), flush=True)
    except Exception as e:
        print("[pinecone_client] 임베딩 생성 오류:", str(e), flush=True)

    # Pinecone 인덱스 벡터 개수(간단 조회)
    try:
        import pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index = pinecone.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        print("[Pinecone] Index stats:", stats, flush=True)
    except Exception as e:
        print("[Pinecone] Index stats 조회 실패:", str(e), flush=True)

    # 검색
    docs = disease_retriever.invoke(query)
    print("[retriever 검색 결과]", flush=True)
    for i, doc in enumerate(docs, 1):
        sim = doc.metadata.get('similarity', 'N/A')
        print(f"[{i}] {doc.metadata.get('disease', 'N/A')}: {sim}", flush=True)
        print(f"[{i}] doc.metadata:", doc.metadata, flush=True)
        print(f"[{i}] doc.page_content[:100]:", doc.page_content[:100], flush=True)
    return [
        {"metadata": doc.metadata, "content": doc.page_content} for doc in docs
    ]
