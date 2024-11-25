from fastapi import APIRouter, HTTPException, Body
import redis.asyncio as redis
import json
from .service import generate_guide_from_prediction
from .schemas import InputData

router = APIRouter()

# Initialize Redis client (assuming default connection parameters)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


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
