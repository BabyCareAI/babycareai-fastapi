# 모찌케어-FastAPI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3.22-purple)
![OpenAI](https://img.shields.io/badge/OpenAI-1.51.1-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-6.0.2-red)

> 본 README.md는 모찌케어 FastAPI 서버에 대한 주요 정보만을 담고 있습니다.  
> [Blog](https://baxdailygit.github.io/) 👈 AI 기반 영유아 피부 질환 진단 서비스 개발 과정을 블로그에 작성하였습니다.  
> [DeepWiki](https://deepwiki.com/BabyCareAI/babycareai-fastapi) 👈딥위키를 통해 해당 레포지토리의 궁금한 점을 물어보세요.  

## 📋 프로젝트 개요

**모찌케어 AI**는 아기 피부 병변 사진과 증상 정보를 종합적으로 분석하여 질환명 예측, 중증도 분석, 병원 내원 필요 여부 판단, 가정 내 처치 방법 안내를 제공하는 AI 기반 서비스입니다.

### 주요 기능  
- 🔍 **이미지 검증**: 피부 질환 진단에 적합한 이미지인지 검증  
- 🖼️ **이미지 분석**: 피부 상태를 자동으로 설명 생성  
- 📝 **기타 증상 처리**: 사용자 입력 증상을 검증/요약/번역하고 Redis에 임시 저장  
- 🧠 **최종 진단**: 벡터 검색과 LLM 추론을 통한 최종 진단


## 🏗️ 시스템 아키텍처
<img width="500" alt="image" src="https://github.com/user-attachments/assets/802c8119-086c-4449-995c-d6d38d49ff3f" />

## 🚀 빠른 시작

### 필수 요구사항
- Python 3.10+
- Docker (선택사항)
- OpenAI API Key
- Google AI API Key
- Pinecone API Key
- AWS 계정 (S3)
- Redis 서버
- MySQL 데이터베이스

### 설치 및 실행

1. **저장소 클론**
```bash
git clone https://github.com/BabyCareAI/babycareai-fastapi.git
cd babycareai-fastapi
```

2. **환경 변수 설정**
```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
DATABASE_URL=your_mysql_url
REDIS_HOST=your_redis_host
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_bucket_name
```

3. **의존성 설치 및 실행**
```bash
pip install -r requirements.txt
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. **API 문서 확인**
```
http://localhost:8000/docs
```

## 📡 API 엔드포인트

### 진단 워크플로우

| 순서 | 엔드포인트 | 메서드 | 설명 |
|------|------------|--------|------|
| 1 | `/api/v1/diagnosis/validate` | POST | 피부 이미지 유효성 검증 |
| 2 | `/api/v1/diagnosis/image-description` | POST | 피부 상태 설명 생성 |
| 3 | `/api/v1/diagnosis/other-symptom` | POST | 기타 증상 입력 처리 |
| 4 | `/api/v1/diagnosis/rag` | POST | RAG 기반 최종 진단 |

### 예시 요청

**1. 이미지 검증**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/validate" \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_id": "uuid-from-upload"}'
```

**2. 이미지 설명 생성**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/image-description" \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_id": "uuid-from-upload"}'
```

**3. 기타 증상 입력**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/other-symptom" \
  -H "Content-Type: application/json" \
  -d '{
    "diagnosis_id": "uuid-from-upload",
    "other_symptom": "아이가 잠을 못자고, 자꾸 칭얼거``
```
**4. 최종 진단**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/RAG" \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_id": "uuid-from-upload"}'
```

4. **API 문서 확인**
```
http://localhost:8000/docs
```

## 📡 API 엔드포인트

### 진단 워크플로우

| 순서 | 엔드포인트 | 메서드 | 설명 |
|------|------------|--------|------|
| 1 | `/api/v1/diagnosis/validate` | POST | 피부 이미지 유효성 검증 |
| 2 | `/api/v1/diagnosis/image-description` | POST | 피부 상태 설명 생성 |
| 3 | `/api/v1/diagnosis/other-symptom` | POST | 기타 증상 입력 처리 |
| 4 | `/api/v1/diagnosis/rag` | POST | RAG 기반 최종 진단 |

### 예시 요청

**1. 이미지 검증**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/validate" \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_id": "uuid-from-upload"}'
```

**2. 이미지 설명 생성**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/image-description" \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_id": "uuid-from-upload"}'
```

**3. 기타 증상 입력**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/other-symptom" \
  -H "Content-Type: application/json" \
  -d '{
    "diagnosis_id": "uuid-from-upload",
    "other_symptom": "아이가 잠을 못자고, 자꾸 칭얼거려요."
  }'
```

**4. 최종 진단**
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/rag" \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_id": "uuid-from-upload"}'
```

## 🔧 기술 스택

- **Framework**: FastAPI 0.115.0
- **Language**: Python 3.10
- **ASGI Server**: Uvicorn 0.31.0
- **Documentation**: OpenAPI 3.0 (Swagger)

### AI/ML 서비스
- **LangChain**: 0.3.22 - LLM 오케스트레이션
- **OpenAI**: 1.51.1 - GPT 모델 및 임베딩
- **Google AI**: Generative Language API
- **Pinecone**: 6.0.2 - 벡터 데이터베이스
- **FAISS**: 1.9.0 - 로컬 벡터 검색

### 데이터베이스 & 캐싱
- **SQLAlchemy**: 2.0.35 - 비동기 ORM
- **MySQL**: aiomysql 연결
- **Redis**: aioredis 2.0.1 - 캐싱 레이어

### 클라우드 서비스
- **AWS S3**: boto3 - 이미지 저장소

### 🔄 진단 프로세스 [3](#1-2) 
<img width="400" alt="image" src="https://github.com/user-attachments/assets/21cec315-2e86-4668-9ce3-22601490749a" />

### RAG 파이프라인
<img width="400" alt="image" src="https://github.com/user-attachments/assets/2a9ef08d-46a5-4cfd-913e-9f754abb43f3" />


### 모니터링
- **Prometheus**: 메트릭 수집 및 계측
- **LangSmith**: LLM 호출 추적 및 성능 모니터링
<img width="400" alt="image" src="https://github.com/user-attachments/assets/9e432bfa-83f6-4d78-b45e-c668bde0b50d" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/b99cb911-664c-4f13-8e8c-5e971bbdf70c" />

## 📈 진단 정확도

테스트 데이터셋 기준 진단 정확도:
- **수두(Chickenpox)**: 60.87% ~ 82.61%
- **수족구병(HFMD)**: 89.13% ~ 100%
- **습진(Eczema)**: 89.36% ~ 91.49%
- **대상포진(Shingles)**: 61.54% ~ 76.92%

## 🚀 배포 자동화 파이프라인
<img width="400" alt="image" src="https://github.com/user-attachments/assets/583daa95-26cf-439a-b0d5-f81d89fd53a9" />

### 배포 명령
```bash
# main 브랜치에 푸시하면 자동 배포
git push origin release/v0.4.1
```

## 📁 프로젝트 구조

```
📦 
.github
workflows
deploy.yml
.gitignore
.idea
│  ├─ .gitignore
│  ├─ inspectionProfiles
│  │  └─ profiles_settings.xml
│  ├─ misc.xml
│  ├─ modules.xml
│  ├─ myapi.iml
│  └─ vcs.xml
├─ Dockerfile
LICENSE
README.md
appspec.yml
├─ diagnosis_id_6.csv
pytest.ini
├─ report.html
├─ report.json
├─ requirements.txt
├─ scripts
│  └─ start-server.sh
├─ src
│  ├─ __init__.py
│  └─ app
│     ├─ __init__.py
│     ├─ database.py
│     ├─ domain
│     │  ├─ __init__.py
│     │  └─ diagnosis
│     │     ├─ __init__.py
│     │     ├─ api
│     │     │  ├─ __init__.py
│     │     │  └─ routers
│     │     │     ├─ __init__.py
│     │     │     ├─ diagnostician.py
│     │     │     ├─ image_descriptor.py
│     │     │     ├─ image_validator.py
│     │     │     └─ other_symptom.py
│     │     ├─ crud
│     │     │  ├─ __init__.py
│     │     │  └─ diagnostician.py
│     │     ├─ schemas
│     │     │  ├─ __init__.py
│     │     │  ├─ diagnostician.py
│     │     │  ├─ image_descriptor.py
│     │     │  ├─ image_validator.py
│     │     │  └─ other_symptom.py
│     │     ├─ services
│     │     │  ├─ __init__.py
│     │     │  ├─ diagnostician.py
│     │     │  ├─ image_descriptor.py
│     │     │  ├─ image_validator.py
│     │     │  └─ other_symptom.py
│     │     └─ utils
│     │        └─ data_processor.py
│     ├─ main.py
│     ├─ models.py
│     └─ utils
│        ├─ __init__.py
│        ├─ llm_client.py
│        ├─ pinecone_client.py
│        ├─ redis_client.py
│        └─ s3_client.py
├─ test-config.yml
└─ test
   ├─ __init__.py
   ├─ conftest.py
   └─ domain
      ├─ __init__.py
      └─ diagnosis
         ├─ __init__.py
         └─ api
            ├─ __init__.py
            └─ routers
               ├─ __init__.py
               └─ test_image_validator.py
```
## 📄 라이센스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
