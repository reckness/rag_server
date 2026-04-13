from fastapi import APIRouter, UploadFile
from app.services.converter import convert_to_pdf
import shutil

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Doc2PDF Service is running", "version": "1.0.0", "endpoint": "/convert"}

@router.post("/convert")
async def convert(file: UploadFile):

    path = f"data/upload/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pdf = convert_to_pdf(path)

    return {"pdf": pdf}