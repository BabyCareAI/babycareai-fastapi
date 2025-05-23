import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.app.domain.diagnosis.api.routers.image_validator import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.mark.asyncio
async def test_validate_skin_image_success():
    # 테스트용 진단 ID
    test_diagnosis_id = "123e4567-e89b-12d3-a456-426614174000"
    
    # Mock 응답 데이터
    mock_result = {"is_skin_related": True}
    
    # derma_validator_service.validate_skin_image를 모킹
    with patch('src.app.domain.diagnosis.api.routers.image_validator.derma_validator_service.validate_skin_image', 
               new_callable=AsyncMock) as mock_validate:
        mock_validate.return_value = mock_result
        
        # API 호출
        response = client.post(
            "/api/v1/diagnosis/validate",
            json={"diagnosis_id": test_diagnosis_id}
        )
        
        # 응답 검증
        assert response.status_code == 200
        assert response.json() == {"is_skin_related": True}
        
        # 모킹된 함수가 올바른 인자로 호출되었는지 검증
        mock_validate.assert_called_once_with(test_diagnosis_id)

@pytest.mark.asyncio
async def test_validate_skin_image_error():
    # 테스트용 진단 ID
    test_diagnosis_id = "123e4567-e89b-12d3-a456-426614174000"
    
    # derma_validator_service.validate_skin_image를 모킹하여 예외 발생
    with patch('src.app.domain.diagnosis.api.routers.image_validator.derma_validator_service.validate_skin_image', 
               new_callable=AsyncMock) as mock_validate:
        mock_validate.side_effect = Exception("테스트 에러")
        
        # API 호출
        response = client.post(
            "/api/v1/diagnosis/validate",
            json={"diagnosis_id": test_diagnosis_id}
        )
        
        # 에러 응답 검증
        assert response.status_code == 500
        assert "이미지 검증 중 오류가 발생했습니다" in response.json()["detail"] 