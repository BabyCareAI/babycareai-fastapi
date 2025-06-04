from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
from src.app.models import DiagnosisResults
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from datetime import datetime
import logging
import re

def clean_uuid(uuid_str: str) -> str:
    """
    UUID 문자열에서 보이지 않는 특수문자와 공백을 제거합니다.
    """
    # UUID 형식에 맞는 문자만 추출 (하이픈 포함)
    cleaned = re.sub(r'[^0-9a-fA-F-]', '', uuid_str)
    return cleaned

async def create_diagnosis_result(
    db: AsyncSession, data: DiagnosisResultsCreate
) -> DiagnosisResults:
    try:
        data_dict = data.dict()
         # ID 정제
        data_dict['id'] = clean_uuid(data_dict['id'])
        current_time = datetime.utcnow()

        
        # UPSERT 쿼리 실행
        stmt = text("""
            INSERT INTO diagnosis_results 
            (id, image_description, symptoms, other_symptom, classification, diagnosis, updated_at)
            VALUES (:id, :image_description, :symptoms, :other_symptom, :classification, :diagnosis, :updated_at)
            ON DUPLICATE KEY UPDATE
            image_description = VALUES(image_description),
            symptoms = VALUES(symptoms),
            other_symptom = VALUES(other_symptom),
            classification = VALUES(classification),
            diagnosis = VALUES(diagnosis),
            updated_at = VALUES(updated_at)
        """)
        
        # UPSERT 실행
        await db.execute(stmt, {**data_dict, 'updated_at': current_time})
        
        # 업데이트된 레코드를 ORM 모델로 조회
        query = select(DiagnosisResults).where(DiagnosisResults.id == data_dict['id'])
        result = await db.execute(query)
        diagnosis_result = result.scalar_one_or_none()
        
        if diagnosis_result is None:
            logging.error(f"[진단 결과 조회 실패] ID: {data_dict['id']}")
            raise RuntimeError("진단 결과를 저장했지만 조회할 수 없습니다.")
            
        # 모든 작업이 성공적으로 완료된 후 커밋
        await db.commit()
        
        return diagnosis_result
        
    except SQLAlchemyError as e:
        logging.error(f"[DB 에러] ID: {data_dict['id']}, 에러: {str(e)}")
        await db.rollback()
        raise RuntimeError(f"DB error: {e}")
    except Exception as e:
        logging.error(f"[예상치 못한 에러] ID: {data_dict['id']}, 에러: {str(e)}")
        await db.rollback()
        raise RuntimeError(f"Unexpected error: {e}")
