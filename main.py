from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# 입력 데이터 모델 정의
class InputData(BaseModel):
    disease_name: str  # 질병 이름
    fever_status: str  # 발열 여부
    blooding_status: str  # 출혈 여부
    age: str  # 나이
    symptoms: str  # 증상

# FastAPI 앱 초기화
app = FastAPI(root_path="/fastapi")
# app = FastAPI()
# 글로벌 변수 초기화
embeddings = None
vectorstores = {}
llm = None
prompt = None

@app.on_event("startup")
async def startup_event():
    global embeddings, vectorstores, llm, prompt

    # API 키 로드
    load_dotenv()

    # 의료 사전 로드
    with open("medical_dic.json", 'r', encoding='utf-8') as f:
        medical_info = json.load(f)

    # 임베딩 초기화
    embeddings = OpenAIEmbeddings()

    # 벡터 저장소 초기화
    for disease_name, disease_info in medical_info.items():
        doc = Document(page_content=disease_info)
        vs = FAISS.from_documents([doc], embeddings)
        vectorstores[disease_name] = vs
    # 이부분 서버끄고 킬때마다 실행되면 토큰 계속 사용됨. 그래서 medical_dic.json을 수정할 때만 실행되도록 수정해야함

    # LLM 초기화
    llm = ChatOpenAI(model_name='gpt-4o-mini-2024-07-18', temperature=0)

    # 프롬프트 정의
    prompt_template_str = """\
    당신은 의료 전문가입니다. 다음 정보를 기반으로 정확한 의료적 설명을 제공하세요. 답변은 의료 지식이 부족한 일반인도 이해할 수 있도록 쉽고 간단하게 작성하고, 부드러운 말투로 작성하세요.

    질환명: {disease_name}
    발열 여부: {fever_status}
    출혈 여부: {blooding_status}
    나이: {age}
    증상: {symptoms}

    컨텍스트에 제공된 정보를 사용하여 답변을 작성하세요. 컨텍스트에 존재하지 않는 정보를 제공하면 처벌받을 수 있습니다. 신중히 답변하세요.

    컨텍스트:
    {context}

    답변양식:
    선택한 사진을 기반으로 예측한 피부질환은 {disease_name}입니다. 아래의 정보는 예측한 진단명과, 입력하신 증상과 정보를 토대로 작성한 답변입니다.

    1. 위험성

    2. 전염성

    3. 응급 처치 방법

    4. 가정내 조치 방법

    5. 병원 방문 필요의 긴급성

    작성된 답변은 예측한 질환에 대해 서울아산병원의 건강정보를 참조하여 작성된 답변입니다. 답변 내용은 참고하시되, 가능한 병원을 방문해주세요.

    답변 작성시 주의사항:
    답변의 각 항목에는 * 등의 기호를 사용하지 마세요.
    """
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["disease_name", "fever_status", "blooding_status", "age", "symptoms", "context"]
    )

def get_vectorstore(disease_name: str):
    vectorstore = vectorstores.get(disease_name)
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="Disease not found")
    return vectorstore


@app.post("/generate")
async def generate_response(input_data: InputData):

    # 입력 데이터 추출
    disease_name = input_data.disease_name
    fever_status = input_data.fever_status
    blooding_status = input_data.blooding_status
    age = input_data.age
    symptoms = input_data.symptoms

    # 질병 이름에 따른 벡터 저장소 가져오기
    vectorstore = get_vectorstore(disease_name)

    # 문서 검색기 초기화
    retriever = vectorstore.as_retriever()

    # 입력된 증상을 기반으로 컨텍스트 문서 검색
    query = symptoms
    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])


    inputs = {
        "disease_name": disease_name,
        "fever_status": fever_status,
        "blooding_status": blooding_status,
        "age": age,
        "symptoms": symptoms,
        "context": context
    }

    # 프롬프트와 LLM을 체인으로 연결
    chain = prompt | llm

    # 응답 생성
    response = chain.invoke(inputs)


    # 응답 반환
    return {"response": response}
