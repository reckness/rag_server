from fastapi import FastAPI
from app.api.convert_api import router

app = FastAPI()

app.include_router(router)