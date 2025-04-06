import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from src.app.diagnosis.api.routers.image_validator import router
from fastapi import FastAPI

# FastAPI 앱 생성
app = FastAPI()
app.include_router(router)

# 테스트 클라이언트 생성
client = TestClient(app)

@pytest.mark.parametrize("test_name,expected_result", [
    ("피부 관련 이미지 API 검증 성공", True),
])
@pytest.mark.asyncio
async def test_validate_skin_image_success(test_name, expected_result):
    """이미지 검증 API 성공 테스트"""
    # 서비스 모킹
    with patch('src.app.diagnosis.api.routers.derma_validator.derma_validator_service') as mock_service:
        # 비동기 함수 모킹
        mock_service.validate_skin_image = AsyncMock(return_value={"is_skin_related": expected_result})
        
        response = client.post("/validate/skin-image", json={"diagnosis_id": "test_diagnosis_id"})
        
        assert response.status_code == 200
        assert response.json() == {"is_skin_related": expected_result}
        mock_service.validate_skin_image.assert_called_once_with("test_diagnosis_id")

@pytest.mark.parametrize("test_name,expected_result", [
    ("피부 관련 이미지가 아닌 경우 API 검증", False),
])
@pytest.mark.asyncio
async def test_validate_skin_image_not_skin_related(test_name, expected_result):
    """이미지 검증 API - 피부 관련 이미지가 아닌 경우 테스트"""
    # 서비스 모킹
    with patch('src.app.diagnosis.api.routers.derma_validator.derma_validator_service') as mock_service:
        # 비동기 함수 모킹
        mock_service.validate_skin_image = AsyncMock(return_value={"is_skin_related": expected_result})
        
        response = client.post("/validate/skin-image", json={"diagnosis_id": "test_diagnosis_id"})
        
        assert response.status_code == 200
        assert response.json() == {"is_skin_related": expected_result}
        mock_service.validate_skin_image.assert_called_once_with("test_diagnosis_id")

@pytest.mark.parametrize("test_name,error_message", [
    ("이미지 검증 API 오류 발생", "이미지 검증 중 오류가 발생했습니다"),
])
@pytest.mark.asyncio
async def test_validate_skin_image_error(test_name, error_message):
    """이미지 검증 API 오류 테스트"""
    # 서비스 모킹
    with patch('src.app.diagnosis.api.routers.derma_validator.derma_validator_service') as mock_service:
        # 비동기 함수 모킹
        mock_service.validate_skin_image = AsyncMock(side_effect=Exception(error_message))
        
        response = client.post("/validate/skin-image", json={"diagnosis_id": "test_diagnosis_id"})
        
        assert response.status_code == 500
        assert error_message in response.json()["detail"]
        mock_service.validate_skin_image.assert_called_once_with("test_diagnosis_id") 