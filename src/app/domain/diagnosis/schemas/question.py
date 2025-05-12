import datetime

from pydantic import BaseModel, field_validator


# Question 클래스는 Question 모델의 스키마를 정의한 클래스이다.
class Question(BaseModel):
    id: int
    subject: str
    content: str
    create_date: datetime.datetime

# QuestionCreate 클래스는 QuestionCreate 모델의 스키마를 정의한 클래스이다.
class QuestionCreate(BaseModel):
    subject: str
    content: str

    # not_empty 함수를 추가하여 subject, content 필드가 빈 값이면 에러를 발생
    @field_validator('subject', 'content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v