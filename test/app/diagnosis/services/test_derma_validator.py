import pytest
import base64
from unittest.mock import patch, MagicMock, AsyncMock
from src.app.diagnosis.services.derma_validator import DermaValidatorService

# 테스트용 이미지 데이터 (더미 데이터)
DUMMY_IMAGE_DATA = b"dummy_image_data"
DUMMY_BASE64_IMAGE = base64.b64encode(DUMMY_IMAGE_DATA).decode('utf-8')

@pytest.fixture
def derma_validator_service():
    """DermaValidatorService 인스턴스를 생성하는 fixture"""
    return DermaValidatorService()

@pytest.mark.parametrize("test_name,expected_result", [
    ("피부 관련 이미지 검증 성공", True),
])
@pytest.mark.asyncio
async def test_validate_skin_image_success(derma_validator_service, test_name, expected_result):
    """이미지 검증 성공 테스트"""
    # S3에서 이미지 가져오기 모킹
    with patch.object(derma_validator_service, '_get_image_from_s3', return_value=DUMMY_IMAGE_DATA):
        # LLM 검증 모킹
        with patch.object(derma_validator_service, '_validate_with_llm', return_value=expected_result):
            result = await derma_validator_service.validate_skin_image("test_diagnosis_id")
            assert result == {"is_skin_related": expected_result}

@pytest.mark.parametrize("test_name,expected_result", [
    ("피부 관련 이미지가 아닌 경우 검증", False),
])
@pytest.mark.asyncio
async def test_validate_skin_image_not_skin_related(derma_validator_service, test_name, expected_result):
    """피부 관련 이미지가 아닌 경우 테스트"""
    # S3에서 이미지 가져오기 모킹
    with patch.object(derma_validator_service, '_get_image_from_s3', return_value=DUMMY_IMAGE_DATA):
        # LLM 검증 모킹
        with patch.object(derma_validator_service, '_validate_with_llm', return_value=expected_result):
            result = await derma_validator_service.validate_skin_image("test_diagnosis_id")
            assert result == {"is_skin_related": expected_result}

@pytest.mark.parametrize("test_name,error_message", [
    ("이미지를 찾을 수 없는 경우 예외 발생", "이미지를 찾을 수 없습니다"),
])
@pytest.mark.asyncio
async def test_validate_skin_image_not_found(derma_validator_service, test_name, error_message):
    """이미지가 없는 경우 테스트"""
    # S3에서 이미지 가져오기 모킹
    with patch.object(derma_validator_service, '_get_image_from_s3', return_value=None):
        with pytest.raises(Exception) as excinfo:
            await derma_validator_service.validate_skin_image("test_diagnosis_id")
        assert error_message in str(excinfo.value)

@pytest.mark.parametrize("test_name", [
    ("S3에서 이미지 가져오기 성공"),
])
@pytest.mark.asyncio
async def test_get_image_from_s3_success(derma_validator_service, test_name):
    """S3에서 이미지 가져오기 성공 테스트"""
    # S3 클라이언트 모킹
    mock_s3_client = MagicMock()
    mock_s3_client.get_object.return_value = {"Body": MagicMock(read=lambda: DUMMY_IMAGE_DATA)}
    
    with patch('src.app.diagnosis.services.derma_validator.s3_client', mock_s3_client):
        result = derma_validator_service._get_image_from_s3("test_diagnosis_id")
        assert result == DUMMY_IMAGE_DATA
        mock_s3_client.get_object.assert_called_once()

@pytest.mark.parametrize("test_name", [
    ("S3에서 이미지 가져오기 실패"),
])
@pytest.mark.asyncio
async def test_get_image_from_s3_failure(derma_validator_service, test_name):
    """S3에서 이미지 가져오기 실패 테스트"""
    # S3 클라이언트 모킹
    mock_s3_client = MagicMock()
    mock_s3_client.get_object.side_effect = Exception("S3 오류")
    
    with patch('src.app.diagnosis.services.derma_validator.s3_client', mock_s3_client):
        result = derma_validator_service._get_image_from_s3("test_diagnosis_id")
        assert result is None

@pytest.mark.parametrize("test_name,model_response,expected_result", [
    ("LLM 검증 - 피부 관련 이미지", "YES", True),
])
@pytest.mark.asyncio
async def test_validate_with_llm_skin_related(derma_validator_service, test_name, model_response, expected_result):
    """LLM 검증 - 피부 관련 이미지 테스트"""
    # LLM 모델 모킹
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = MagicMock(content=model_response)
    
    with patch.object(derma_validator_service, 'model', mock_model):
        result = await derma_validator_service._validate_with_llm(DUMMY_BASE64_IMAGE)
        assert result is expected_result
        mock_model.ainvoke.assert_called_once()

@pytest.mark.parametrize("test_name,model_response,expected_result", [
    ("LLM 검증 - 피부 관련 이미지가 아닌 경우", "NO", False),
])
@pytest.mark.asyncio
async def test_validate_with_llm_not_skin_related(derma_validator_service, test_name, model_response, expected_result):
    """LLM 검증 - 피부 관련 이미지가 아닌 경우 테스트"""
    # LLM 모델 모킹
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = MagicMock(content=model_response)
    
    with patch.object(derma_validator_service, 'model', mock_model):
        result = await derma_validator_service._validate_with_llm(DUMMY_BASE64_IMAGE)
        assert result is expected_result
        mock_model.ainvoke.assert_called_once() 