# 모찌케어-FastAPI

초보 부모들은 아기가 병원에 가봐야 하는 상황인지조차 파악하기 어려울 때가 많다. 특히 영유아기에 자주 발생하는 크고 작은 피부 질환들은 때로는 심각한 피부 질환과 구분하기 어려울 수 있다. '모찌케어'는 아기의 피부 병변 이미지와 추가적인 증상들을 종합적으로 분석하여 피부 질환을 분석하고 가정 내 처치 방법, 증상의 중증도, 병원 내원 필요 여부 등의 정보를 제공한다.

https://baby-care-ai-app.vercel.app

https://github.com/user-attachments/assets/09b2692a-c152-4230-9bdc-8dd68905ebef
(현재는 피부질환 예측 기능같은 경우 Amazon SageMaker 실시간 추론에서 서버리스추론으로 전환하였습니다.)

# 기능 
> ### 피부질환분석 (MVP)
> - 이미지 업로드 
> - 피부질환 예측 
> - 추가 정보 입력 (발열 여부, 가려움 등) 
> - 최종 진단 및 가이드 생성 

<br>

# 시스템 아키텍처
![image](https://github.com/user-attachments/assets/326d5366-600b-4e24-a440-593697053262)

<br>

# API 명세서
|Spring Boot|FastAPI|
|---|---|
|![screencapture-cautious-kale-62b-notion-site-api-1c4b9a739f36433d89e76ec05c27853a-2024-12-19-22_55_12](https://github.com/user-attachments/assets/1a1a10f6-8ec0-4f37-8c4a-02e83a822431)|![screencapture-cautious-kale-62b-notion-site-api-1c4b9a739f36433d89e76ec05c27853a-2024-12-19-23_10_51](https://github.com/user-attachments/assets/462461d8-ec90-4d13-8036-f5becf5da8c5)|

# Swagger
|Spring Boot|FastAPI|
|---|---|
|![screencapture-api-babycareai-net-swagger-ui-index-html-2024-12-19-22_35_32](https://github.com/user-attachments/assets/047c1c55-ae56-481b-963f-727df2fa403b)|![screencapture-api-babycareai-net-fastapi-docs-2024-12-19-23_12_17](https://github.com/user-attachments/assets/5a3299ff-4d4a-4199-a924-3ed066fe6651)|
