import asyncio
from fastapi import FastAPI
from guide.router import router as nlp_router
from guide.service import initialize_service
from guide.redis_listener import redis_stream_listener

app = FastAPI(root_path="/fastapi")

app.include_router(nlp_router)

@app.on_event("startup")
async def startup_event():
    await initialize_service()
    asyncio.create_task(redis_stream_listener())