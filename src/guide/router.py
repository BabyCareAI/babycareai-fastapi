from fastapi import APIRouter
from .schemas import InputData
from .service import generate_response

router = APIRouter()

@router.post("/generate")
async def generate_response_endpoint(input_data: InputData):
    response = await generate_response(input_data)
    return {"response": response}