import asyncio
import redis.asyncio as redis
import json
from .service import generate_guide_from_prediction
from .schemas import InputData
from redis.exceptions import ResponseError


'''
redis stream에서 이미지 예측 결과를 받아서 가이드를 생성하기 위한 비동기 함수
redis 키: diagnosis:prediction:result:stream
redis 소비자 그룹: guidance_service_group
producers: PredictService.java
consumers: redis_listener.py
'''
async def redis_stream_listener():
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    stream_name = "diagnosis:prediction:result:stream"
    group_name = "guidance_service_group"
    consumer_name = "guidance_service_consumer"

    # Create consumer group if it doesn't exist
    try:
        await redis_client.xgroup_create(stream_name, group_name, id='0-0', mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            print("Consumer group already exists")
        else:
            raise e

    while True:
        try:
            # Read messages from the stream
            result = await redis_client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: '>'},
                count=1,
                block=0
            )
            if result:
                # Process messages
                for stream, messages in result:
                    for message_id, message in messages:
                        message_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in message.items()}
                        await process_message(
                            message_id.decode('utf-8'),
                            message_dict,
                            redis_client,
                            stream.decode('utf-8'),
                            group_name
                        )
        except Exception as e:
            print(f"Error in redis_stream_listener: {e}")
            await asyncio.sleep(1)


async def process_message(message_id, message, redis_client, stream_name, group_name):
    # Extract fields from the message
    image_url = message.get('imageUrl')
    prediction_result_str = message.get('predictionResult')

    # Parse prediction result
    prediction_result = json.loads(prediction_result_str)

    # Extract necessary information
    predicted_classes = prediction_result.get('predictionResult')
    probabilities = prediction_result.get('probabilities')

    disease_name = predicted_classes[0] if predicted_classes else "Unknown"

    # Prepare input data
    input_data = InputData(
        # disease_name=disease_name,
        prediction_result=prediction_result
    )

    # Generate guide
    response = await generate_guide_from_prediction(input_data)

    # Handle the generated guide (e.g., save to database)
    print("Generated Guide:", response)

    # Acknowledge the message
    await redis_client.xack(stream_name, group_name, message_id)