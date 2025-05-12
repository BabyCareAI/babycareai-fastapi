from src.app.models import Question
from sqlalchemy.orm import Session
from datetime import datetime

from src.app.domain.diagnosis.schemas.question import QuestionCreate

# 본 파일(question_crud.py)은 질문에 관련된 CRUD 함수를 정의한 파일이다.
# question_router.py 파일에서 사용된다.
# question_crud.py과 question_router.py 파일을 분리한 이유는 역할과 책임을 분리하기 위해서이다.
# question_crud.py 파일의 역할과 책임은 질문과 관련된 CRUD 함수를 정의하는 것이다.

# get_question_list 함수는 데이터베이스에 저장된 질문 목록을 조회하는 함수이다.
def get_question_list(db: Session):
    question_list = db.query(Question)\
        .order_by(Question.create_date.desc())\
        .all()
    return question_list

# get_question 함수는 데이터베이스에 저장된 질문 상세 정보를 조회하는 함수이다.
def get_question(db: Session, question_id: int):
    question = db.query(Question).get(question_id)
    return question

# create_question 함수는 질문을 생성하는 함수이다.
def create_question(db: Session, question_create: QuestionCreate):
    db_question = Question(subject=question_create.subject,
                           content=question_create.content,
                           create_date=datetime.now())
    db.add(db_question)
    db.commit()