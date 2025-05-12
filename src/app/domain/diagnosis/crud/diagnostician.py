from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
from src.app.models import DiagnosisResults
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from datetime import datetime

async def create_diagnosis_result(
    db: AsyncSession, data: DiagnosisResultsCreate
) -> DiagnosisResults:
    try:
        data_dict = data.dict()
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
            raise RuntimeError("진단 결과를 저장했지만 조회할 수 없습니다.")
            
        # 모든 작업이 성공적으로 완료된 후 커밋
        await db.commit()
        
        return diagnosis_result
        
    except SQLAlchemyError as e:
        await db.rollback()
        raise RuntimeError(f"DB error: {e}")
    except Exception as e:
        await db.rollback()
        raise RuntimeError(f"Unexpected error: {e}")
