from fastapi import APIRouter
from .schemas import InputData
from .service import generate_guide_from_prediction

router = APIRouter()

@router.post("/generate")
async def generate_response_endpoint(input_data: InputData):
    response = await generate_guide_from_prediction(input_data)
    return {"response": response}