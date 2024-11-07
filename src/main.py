from fastapi import FastAPI
from guide.router import router as nlp_router
from guide.service import initialize_service

app = FastAPI(root_path="/fastapi")

app.include_router(nlp_router)

@app.on_event("startup")
async def startup_event():
    await initialize_service()