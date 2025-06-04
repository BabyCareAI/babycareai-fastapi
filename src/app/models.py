from sqlalchemy import Column, String, Text, DateTime, Integer
from sqlalchemy.sql import func

from .database import Base

class DiagnosisResults(Base):
    __tablename__ = "diagnosis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    diagnosis_id = Column(String(36), nullable=False)
    image_description = Column(Text, nullable=True)
    symptoms = Column(Text, nullable=True)
    other_symptom = Column(Text, nullable=True)
    classification = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

