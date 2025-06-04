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
        # diagnosis_id 정제
        data_dict['diagnosis_id'] = clean_uuid(data_dict['diagnosis_id'])
        current_time = datetime.utcnow()
        
        # INSERT 쿼리 실행 (UPSERT 대신 INSERT만 수행)
        stmt = text("""
            INSERT INTO diagnosis_results 
            (diagnosis_id, image_description, symptoms, other_symptom, classification, diagnosis, updated_at)
            VALUES (:diagnosis_id, :image_description, :symptoms, :other_symptom, :classification, :diagnosis, :updated_at)
        """)
        
        # INSERT 실행
        result = await db.execute(stmt, {**data_dict, 'updated_at': current_time})
        
        # 생성된 레코드의 id 가져오기
        inserted_id = result.lastrowid
        
        # 생성된 레코드를 ORM 모델로 조회
        query = select(DiagnosisResults).where(DiagnosisResults.id == inserted_id)
        result = await db.execute(query)
        diagnosis_result = result.scalar_one_or_none()
        
        if diagnosis_result is None:
            logging.error(f"[진단 결과 조회 실패] id: {inserted_id}")
            raise RuntimeError("진단 결과를 저장했지만 조회할 수 없습니다.")
            
        # 모든 작업이 성공적으로 완료된 후 커밋
        await db.commit()
        
        return diagnosis_result
        
    except SQLAlchemyError as e:
        logging.error(f"[DB 에러] diagnosis_id: {data_dict['diagnosis_id']}, 에러: {str(e)}")
        await db.rollback()
        raise RuntimeError(f"DB error: {e}")
    except Exception as e:
        logging.error(f"[예상치 못한 에러] diagnosis_id: {data_dict['diagnosis_id']}, 에러: {str(e)}")
        await db.rollback()
        raise RuntimeError(f"Unexpected error: {e}")
