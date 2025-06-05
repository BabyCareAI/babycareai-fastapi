# 모찌케어-FastAPI

README.md 작성중

## AI 기반 GitHub 안내서
deepwiki 문서: https://deepwiki.com/BabyCareAI/babycareai-fastapi

## 프로젝트 구조

```
📦 
├─ .github
│  └─ workflows
│     └─ deploy.yml
├─ .gitignore
├─ .idea
│  ├─ .gitignore
│  ├─ inspectionProfiles
│  │  └─ profiles_settings.xml
│  ├─ misc.xml
│  ├─ modules.xml
│  ├─ myapi.iml
│  └─ vcs.xml
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ appspec.yml
├─ diagnosis_id_6.csv
├─ pytest.ini
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
