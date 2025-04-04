import pytest
import os
from dotenv import load_dotenv

# 테스트 환경 변수 설정
@pytest.fixture(autouse=True)
def setup_test_env():
    """테스트 환경 변수 설정"""
    # .env 파일 로드
    load_dotenv()
    
    # 테스트용 환경 변수 설정
    os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
    os.environ["AWS_REGION"] = "ap-northeast-2"
    os.environ["S3_BUCKET_NAME"] = "test-bucket"
    
    yield
