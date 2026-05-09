from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
from dotenv import load_dotenv; load_dotenv()

from backend.utils.ocr_engine import extract_text_from_image

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

class QueryRequest(BaseModel):
    question: str

@app.post("/upload_pdf/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    UPLOAD_DIR = os.path.join("uploaded_data", "files")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        return {"message": f"Successfully uploaded {len(files)} files."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/build_db/")
async def build_db():
    try:
        from scripts.populate_database import main as build_database
        build_database()
        return {"message": "Knowledge base built successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: QueryRequest):
    try:
        from scripts.query_data import query_rag
        response = query_rag(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat generation failed: {e}")

@app.post("/ocr/")
async def ocr(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        ocr_result = extract_text_from_image(image_bytes)

        extracted_text = (ocr_result.get("text") or "").strip()
        if not extracted_text:
            if ocr_result.get("error"):
                raise HTTPException(status_code=400, detail=f"OCR failed: {ocr_result['error']}")
            return {
                "extracted_text": "No readable text found in the image.",
                "confidence": ocr_result.get("confidence", 0.0),
                "method": ocr_result.get("method", "unknown"),
            }

        return {
            "extracted_text": extracted_text,
            "confidence": ocr_result.get("confidence", 0.0),
            "method": ocr_result.get("method", "unknown"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
