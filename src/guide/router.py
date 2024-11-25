from fastapi import APIRouter, HTTPException, Body
import os
from dotenv import load_dotenv
import redis.asyncio as redis
import json
from .service import generate_guide_from_prediction
from .schemas import InputData

router = APIRouter()

# Initialize Redis client (assuming default connection parameters)
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

@router.post("/api/diagnosis/generate")
async def generate_diagnosis(diagnosis_id: str = Body(..., embed=True)):
    try:
        # Construct Redis keys
        prediction_key = f"prediction:{diagnosis_id}"
        symptoms_key = f"symptoms:{diagnosis_id}"

        # Fetch data from Redis
        prediction_data = await redis_client.get(prediction_key)
        symptoms_data = await redis_client.get(symptoms_key)

        if not prediction_data or not symptoms_data:
            raise HTTPException(status_code=404, detail="Diagnosis data not found")

        # Parse the prediction data
        prediction_result = json.loads(prediction_data)

        # Use symptoms_data as a string without parsing
        symptoms = symptoms_data  # Accept as is

        # Create InputData object
        input_data = InputData(
            symptoms=symptoms,
            prediction_result=prediction_result
        )

        # Call the service function to generate the guide
        guide = await generate_guide_from_prediction(input_data)

        return {"guide": guide}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
