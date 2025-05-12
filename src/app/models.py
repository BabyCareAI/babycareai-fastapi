from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func

from .database import Base

class DiagnosisResults(Base):
    __tablename__ = "diagnosis_results"

    id = Column(String(36), primary_key=True, nullable=False)  # UUID
    image_description = Column(Text, nullable=True)
    symptoms = Column(Text, nullable=True)
    other_symptom = Column(Text, nullable=True)
    classification = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

class Question(Base):
    __tablename__ = "question"

    id = Column(Integer, primary_key=True)
    subject = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
