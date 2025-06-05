from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from src.app.models import DiagnosisResults
from src.app.domain.diagnosis.schemas.diagnostician import DiagnosisResultsCreate
from datetime import datetime


async def create_diagnosis_result(
        db: AsyncSession, data: DiagnosisResultsCreate
) -> DiagnosisResults:
    try:
        data_dict = data.dict()
        current_time = datetime.utcnow()

        # 새 레코드 생성
        diagnosis_result = DiagnosisResults(
            diagnosis_id=data_dict['diagnosis_id'],
            image_description=data_dict['image_description'],
            symptoms=data_dict['symptoms'],
            other_symptom=data_dict['other_symptom'],
            classification=data_dict['classification'],
            diagnosis=data_dict['diagnosis'],
            updated_at=current_time
        )
        db.add(diagnosis_result)
        await db.commit()
        await db.refresh(diagnosis_result)

        return diagnosis_result

    except SQLAlchemyError as e:
        await db.rollback()
        raise RuntimeError(f"DB error: {e}")
    except Exception as e:
        await db.rollback()
        raise RuntimeError(f"Unexpected error: {e}")