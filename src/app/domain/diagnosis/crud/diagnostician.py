from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.app.models import DiagnosisResults
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate

async def create_diagnosis_result(
    db: AsyncSession, data: DiagnosisResultsCreate
) -> DiagnosisResults:
    try:
        data_dict = data.dict()
        # ON DUPLICATE KEY UPDATE 구문 사용하여
        # 1. id가 이미 존재하는 경우, 기존 레코드를 업데이트
        # 2. id가 존재하지 않는 경우, 새로운 레코드를 삽입
        stmt = text("""
            INSERT INTO diagnosis_results 
            (id, image_description, symptoms, other_symptom, classification, diagnosis)
            VALUES (:id, :image_description, :symptoms, :other_symptom, :classification, :diagnosis)
            ON DUPLICATE KEY UPDATE
            image_description = VALUES(image_description),
            symptoms = VALUES(symptoms),
            other_symptom = VALUES(other_symptom),
            classification = VALUES(classification),
            diagnosis = VALUES(diagnosis)
        """)
        
        await db.execute(stmt, data_dict) # SQLAlchemy에서 실행
        await db.commit() # 트랜잭션 커밋
        
        # 업데이트된 레코드 조회
        result = await db.execute(
            text("SELECT * FROM diagnosis_results WHERE id = :id"),
            {"id": data_dict['id']}
        )
        return result.fetchone()
        
    except SQLAlchemyError as e:
        await db.rollback()
        raise RuntimeError(f"DB error: {e}")
