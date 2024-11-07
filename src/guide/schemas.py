from pydantic import BaseModel

class InputData(BaseModel):
    disease_name: str  # 질병 이름
    fever_status: str  # 발열 여부
    blooding_status: str  # 출혈 여부
    age: str            # 나이
    symptoms: str       # 증상